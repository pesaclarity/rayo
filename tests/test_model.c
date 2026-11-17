/* format/pcwt, the plan, the golden outputs of the test models, and memory. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "rayo/rayo.h"
#include "session.h"
#include "test.h"

static unsigned char *model_file(const char *name, size_t *size) {
  char path[512];
  snprintf(path, sizeof path, "%s/%s", test_models_dir(), name);
  unsigned char *p = test_read_file(path, size);
  if (!p) fprintf(stderr, "cannot read %s\n", path);
  return p;
}

static void put_u32(unsigned char *p, uint32_t v) {
  for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}

void test_pcwt(void) {
  size_t size;
  unsigned char *file = model_file("reader.weights", &size);
  EXPECT(file != NULL);
  if (!file) return;
  rayo_model *m = NULL;
  rayo_message msg;
  EXPECT(rayo_model_create(file, size, &m, &msg) == RAYO_OK);
  EXPECT(strcmp(rayo_model_name(m), "test-reader") == 0);
  EXPECT(strcmp(rayo_model_metadata_json(m), "{\"labels\":[\"a\",\"b\",\"c\"],\"count\":3}") == 0);
  EXPECT(rayo_model_input_count(m) == 2 && rayo_model_output_count(m) == 2);
  rayo_tensor_view v;
  EXPECT(rayo_model_input_info(m, 1, &v) == RAYO_OK && strcmp(v.name, "shapes") == 0 &&
         v.dtype == RAYO_DTYPE_I32 && v.shape.rank == 1 && v.shape.dims[0] == 12);
  EXPECT(rayo_model_input_info(m, 2, &v) == RAYO_ERR_INVALID_ARGUMENT);
  rayo_model_destroy(m);
  rayo_model_destroy(NULL);
  /* Every truncation fails cleanly, never reads past the end. */
  for (size_t cut = 0; cut < size; cut += 7) {
    unsigned char *t = malloc(cut ? cut : 1);
    memcpy(t, file, cut);
    EXPECT(rayo_model_create(t, cut, &m, NULL) != RAYO_OK && m == NULL);
    free(t);
  }
  /* A wrong magic, version or header length. */
  unsigned char *bad = malloc(size);
  memcpy(bad, file, size);
  bad[0] = 'X';
  EXPECT(rayo_model_create(bad, size, &m, &msg) == RAYO_ERR_FORMAT);
  memcpy(bad, file, size);
  put_u32(bad + 4, 2);
  EXPECT(rayo_model_create(bad, size, &m, &msg) == RAYO_ERR_UNSUPPORTED);
  memcpy(bad, file, size);
  put_u32(bad + 8, 0xFFFFFFF0u);
  EXPECT(rayo_model_create(bad, size, &m, &msg) == RAYO_ERR_FORMAT);
  /* Random byte flips: anything but a crash. */
  uint32_t s = 7;
  for (int i = 0; i < 2000; i++) {
    memcpy(bad, file, size);
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    bad[s % size] = (unsigned char)(s >> 8);
    if (rayo_model_create(bad, size, &m, NULL) == RAYO_OK) {
      rayo_session *se = NULL;
      if (rayo_session_create(m, &se, NULL) == RAYO_OK) {
        rayo_session_run(se, NULL);
        rayo_session_destroy(se);
      }
      rayo_model_destroy(m);
    }
  }
  free(bad);
  EXPECT(rayo_model_create(NULL, 10, &m, NULL) == RAYO_ERR_INVALID_ARGUMENT);
  free(file);
}

/* Planned activations never overlap while both are alive. */
void test_plan(void) {
  size_t size;
  unsigned char *file = model_file("reader.weights", &size);
  if (!file) return;
  rayo_model *m;
  rayo_session *s;
  EXPECT(rayo_model_create(file, size, &m, NULL) == RAYO_OK);
  EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
  const rayo_graph *g = &m->graph;
  int32_t last[64];
  for (int32_t i = 0; i < g->node_count; i++) {
    last[i] = i;
    for (int32_t a = 0; a < g->nodes[i].input_count; a++) {
      if (g->nodes[i].inputs[a].kind == RAYO_REF_NODE) last[g->nodes[i].inputs[a].index] = i;
    }
  }
  for (int32_t o = 0; o < g->output_count; o++) last[g->outputs[o].node] = g->node_count;
  for (int32_t i = 0; i < g->node_count; i++) {
    for (int32_t j = i + 1; j < g->node_count; j++) {
      if (last[i] < j) continue; /* i is dead before j is made */
      size_t ai = s->plan.node_offset[i], bi = ai + rayo_tensor_bytes(&s->nodes[i]);
      size_t aj = s->plan.node_offset[j], bj = aj + rayo_tensor_bytes(&s->nodes[j]);
      EXPECT(bi <= aj || bj <= ai);
    }
    EXPECT(s->plan.node_offset[i] + rayo_tensor_bytes(&s->nodes[i]) <= s->plan.scratch_offset);
  }
  rayo_memory_info mem;
  rayo_session_memory(s, &mem);
  EXPECT(mem.arena == s->plan.arena_bytes && mem.weights == size);
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  free(file);
}

static double compare(const rayo_session *s, int output, const rayo_json *want) {
  rayo_tensor_view v;
  rayo_session_output(s, output, &v);
  const float *got = v.data;
  if (v.bytes / 4 != want->length) return INFINITY;
  double worst = 0;
  for (size_t i = 0; i < want->length; i++)
    worst = fmax(worst, fabs(got[i] - want->items[i]->number));
  return worst;
}

static void fill(rayo_session *s, int input, const rayo_json *values) {
  rayo_tensor_view v;
  rayo_session_input(s, input, &v);
  int32_t *p = v.data;
  for (size_t i = 0; i < values->length; i++) p[i] = (int32_t)values->items[i]->number;
}

void test_golden(void) {
  size_t size, esize;
  unsigned char *expected = model_file("expected.json", &esize);
  if (!expected) {
    EXPECT(0);
    return;
  }
  rayo_arena a = {0};
  rayo_json *root;
  EXPECT(rayo_json_parse((const char *)expected, esize, &a, &root, NULL) == RAYO_OK);
  const char *names[] = {"reader", "float"};
  for (int k = 0; k < 2; k++) {
    const rayo_json *spec = rayo_json_get(root, names[k]);
    unsigned char *file = model_file(rayo_json_as_string(rayo_json_get(spec, "file")), &size);
    rayo_model *m;
    rayo_session *s;
    rayo_message msg;
    if (rayo_model_create(file, size, &m, &msg) != RAYO_OK) {
      fprintf(stderr, "%s: %s\n", names[k], msg.text);
      EXPECT(0);
      continue;
    }
    EXPECT(rayo_session_create(m, &s, &msg) == RAYO_OK);
    const rayo_json *cases = rayo_json_get(spec, "cases");
    for (size_t c = 0; c < cases->length; c++) {
      const rayo_json *cs = cases->items[c];
      int64_t t;
      if (k == 0) {
        rayo_json_as_int(rayo_json_get(cs, "T"), &t);
        EXPECT(rayo_session_bind(s, "T", t, &msg) == RAYO_OK);
        fill(s, 0, rayo_json_get(cs, "tokens"));
        fill(s, 1, rayo_json_get(cs, "shapes"));
      } else {
        rayo_json_as_int(rayo_json_get(cs, "N"), &t);
        EXPECT(rayo_session_bind(s, "N", t, &msg) == RAYO_OK);
        fill(s, 0, rayo_json_get(cs, "ids"));
      }
      EXPECT(rayo_session_run(s, &msg) == RAYO_OK);
      if (k == 0) {
        EXPECT(compare(s, 0, rayo_json_get(cs, "slots")) < 1e-5);
        EXPECT(compare(s, 1, rayo_json_get(cs, "intent")) < 1e-5);
      } else {
        EXPECT(compare(s, 0, rayo_json_get(cs, "y")) < 1e-5);
      }
    }
    /* Out-of-range sizes and ids are refused, not run. */
    EXPECT(rayo_session_bind(s, k == 0 ? "T" : "N", 0, NULL) == RAYO_ERR_SHAPE);
    EXPECT(rayo_session_bind(s, k == 0 ? "T" : "N", 1000, NULL) == RAYO_ERR_SHAPE);
    EXPECT(rayo_session_bind(s, "Q", 1, NULL) == RAYO_ERR_INVALID_ARGUMENT);
    rayo_tensor_view v;
    rayo_session_input(s, 0, &v);
    ((int32_t *)v.data)[0] = 99999;
    EXPECT(rayo_session_run(s, &msg) == RAYO_ERR_SHAPE);
    rayo_session_destroy(s);
    rayo_model_destroy(m);
    free(file);
  }
  rayo_arena_free(&a);
  free(expected);
}

/* A run allocates nothing, however many times it runs. */
void test_memory(void) {
  size_t size;
  unsigned char *file = model_file("reader.weights", &size);
  if (!file) return;
  rayo_model *m;
  rayo_session *s;
  EXPECT(rayo_model_create(file, size, &m, NULL) == RAYO_OK);
  EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
  long before = test_allocations();
  for (int i = 0; i < 1000; i++) {
    rayo_session_bind(s, "T", 1 + i % 12, NULL);
    EXPECT(rayo_session_run(s, NULL) == RAYO_OK);
  }
  EXPECT(test_allocations() == before);
  printf("  allocations in 1000 runs: %ld\n", test_allocations() - before);
  rayo_session_destroy(s);
  rayo_session_destroy(NULL);
  rayo_model_destroy(m);
  free(file);
}
