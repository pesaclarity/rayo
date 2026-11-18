/* format/pcwt, the plan, the golden outputs of the test models, memory,
 * and a decoder with a cache. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "session.h"
#include "test.h"

static void put_u32(unsigned char *p, rayo_u32 v) {
  int i;
  for (i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static rayo_u32 xorshift(rayo_u32 s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

static void open_or_fail(rayo_model *m) {
  rayo_session *se = NULL;
  if (rayo_session_create(m, &se, NULL) == RAYO_OK) {
    rayo_session_run(se, NULL);
    rayo_session_destroy(se);
  }
  rayo_model_destroy(m);
}

static void damaged_files(const unsigned char *file, size_t size) {
  unsigned char *bad = malloc(size);
  rayo_model *m = NULL;
  rayo_message msg;
  rayo_u32 s = 7;
  size_t cut;
  int i;
  if (!bad) return;
  /* Every truncation fails cleanly, never reads past the end. */
  for (cut = 0; cut < size; cut += 7) {
    unsigned char *t = malloc(cut ? cut : 1);
    memcpy(t, file, cut);
    EXPECT(rayo_model_create(t, cut, &m, NULL) != RAYO_OK && m == NULL);
    free(t);
  }
  /* A wrong magic, version or header length. */
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
  for (i = 0; i < 2000; i++) {
    memcpy(bad, file, size);
    s = xorshift(s);
    bad[s % size] = (unsigned char)(s >> 8);
    if (rayo_model_create(bad, size, &m, NULL) == RAYO_OK) open_or_fail(m);
  }
  free(bad);
}

void test_pcwt(void) {
  size_t size;
  unsigned char *file = test_model_file("reader.weights", &size);
  rayo_model *m = NULL;
  rayo_message msg;
  rayo_tensor_view v;
  EXPECT(file != NULL);
  if (!file) return;
  EXPECT(rayo_model_create(file, size, &m, &msg) == RAYO_OK);
  EXPECT(strcmp(rayo_model_name(m), "test-reader") == 0);
  EXPECT(strcmp(rayo_model_metadata_json(m), "{\"labels\":[\"a\",\"b\",\"c\"],\"count\":3}") == 0);
  EXPECT(rayo_model_input_count(m) == 2 && rayo_model_output_count(m) == 2);
  EXPECT(rayo_model_input_info(m, 1, &v) == RAYO_OK && strcmp(v.name, "shapes") == 0 &&
         v.dtype == RAYO_DTYPE_I32 && v.shape.rank == 1 && v.shape.dims[0] == 12);
  EXPECT(rayo_model_input_info(m, 2, &v) == RAYO_ERR_INVALID_ARGUMENT);
  rayo_model_destroy(m);
  rayo_model_destroy(NULL);
  damaged_files(file, size);
  EXPECT(rayo_model_create(NULL, 10, &m, NULL) == RAYO_ERR_INVALID_ARGUMENT);
  free(file);
}

/* Planned activations never overlap while both are alive. */
void test_plan(void) {
  size_t size;
  unsigned char *file = test_model_file("reader.weights", &size);
  rayo_model *m;
  rayo_session *s;
  const rayo_graph *g;
  rayo_i32 last[64], i, j, a, o;
  rayo_memory_info mem;
  if (!file) return;
  EXPECT(rayo_model_create(file, size, &m, NULL) == RAYO_OK);
  EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
  g = &m->graph;
  for (i = 0; i < g->node_count; i++) {
    last[i] = i;
    for (a = 0; a < g->nodes[i].input_count; a++) {
      if (g->nodes[i].inputs[a].kind == RAYO_REF_NODE) last[g->nodes[i].inputs[a].index] = i;
    }
  }
  for (o = 0; o < g->output_count; o++) last[g->outputs[o].node] = g->node_count;
  for (i = 0; i < g->node_count; i++) {
    for (j = i + 1; j < g->node_count; j++) {
      size_t ai = s->plan.node_offset[i], bi = ai + rayo_tensor_bytes(&s->values.nodes[i]);
      size_t aj = s->plan.node_offset[j], bj = aj + rayo_tensor_bytes(&s->values.nodes[j]);
      if (last[i] < j) continue; /* i is dead before j is made */
      EXPECT(bi <= aj || bj <= ai);
    }
    EXPECT(s->plan.node_offset[i] + rayo_tensor_bytes(&s->values.nodes[i]) <=
           s->plan.scratch_offset);
  }
  rayo_session_memory(s, &mem);
  EXPECT(mem.arena == s->plan.arena_bytes && mem.weights == size);
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  free(file);
}

static double compare(const rayo_session *s, int output, const rayo_json *want) {
  rayo_tensor_view v;
  const float *got;
  double worst = 0;
  size_t i;
  rayo_session_output(s, output, &v);
  got = v.data;
  if (v.bytes / 4 != want->length) return 1e30;
  for (i = 0; i < want->length; i++) {
    double d = fabs(got[i] - want->items[i]->number);
    worst = d > worst ? d : worst;
  }
  return worst;
}

static void fill(rayo_session *s, int input, const rayo_json *values) {
  rayo_tensor_view v;
  rayo_i32 *p;
  size_t i;
  rayo_session_input(s, input, &v);
  p = v.data;
  for (i = 0; i < values->length; i++) p[i] = (rayo_i32)values->items[i]->number;
}

static rayo_json *expected_json(rayo_arena *a, unsigned char **text) {
  size_t size;
  rayo_json *root = NULL;
  *text = test_model_file("expected.json", &size);
  if (*text && rayo_json_parse((const char *)*text, size, (size_t)-1, a, &root, NULL) != RAYO_OK)
    root = NULL;
  return root;
}

static void golden_case(rayo_session *s, rayo_bool reader, const rayo_json *cs) {
  rayo_message msg;
  rayo_i32 t;
  if (reader) {
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
  if (reader) {
    EXPECT(compare(s, 0, rayo_json_get(cs, "slots")) < 1e-5);
    EXPECT(compare(s, 1, rayo_json_get(cs, "intent")) < 1e-5);
  } else {
    EXPECT(compare(s, 0, rayo_json_get(cs, "y")) < 1e-5);
  }
}

void test_golden(void) {
  static const char *const names[] = {"reader", "float"};
  rayo_arena a;
  unsigned char *text;
  rayo_json *root;
  int k;
  a.head = NULL;
  a.bytes = 0;
  root = expected_json(&a, &text);
  EXPECT(root != NULL);
  for (k = 0; root && k < 2; k++) {
    const rayo_json *spec = rayo_json_get(root, names[k]);
    const rayo_json *cases = rayo_json_get(spec, "cases");
    const char *symbol = k == 0 ? "T" : "N";
    size_t size, c;
    unsigned char *file = test_model_file(rayo_json_as_string(rayo_json_get(spec, "file")), &size);
    rayo_model *m;
    rayo_session *s;
    rayo_message msg;
    rayo_tensor_view v;
    if (!file || rayo_model_create(file, size, &m, &msg) != RAYO_OK) {
      EXPECT(0);
      free(file);
      continue;
    }
    EXPECT(rayo_session_create(m, &s, &msg) == RAYO_OK);
    test_choose_kernels(s);
    for (c = 0; c < cases->length; c++) golden_case(s, k == 0, cases->items[c]);
    /* Out-of-range sizes and ids are refused, not run. */
    EXPECT(rayo_session_bind(s, symbol, 0, NULL) == RAYO_ERR_SHAPE);
    EXPECT(rayo_session_bind(s, symbol, 1000, NULL) == RAYO_ERR_SHAPE);
    EXPECT(rayo_session_bind(s, "Q", 1, NULL) == RAYO_ERR_INVALID_ARGUMENT);
    rayo_session_input(s, 0, &v);
    ((rayo_i32 *)v.data)[0] = 99999;
    EXPECT(rayo_session_run(s, &msg) == RAYO_ERR_SHAPE);
    rayo_session_destroy(s);
    rayo_model_destroy(m);
    free(file);
  }
  rayo_arena_free(&a);
  free(text);
}

/* A run allocates nothing, however many times it runs. */
void test_memory(void) {
  size_t size;
  unsigned char *file = test_model_file("reader.weights", &size);
  rayo_model *m;
  rayo_session *s;
  long before;
  int i;
  if (!file) return;
  if (test_allocations() < 0) {
    printf("  allocation counting not built in (no --wrap): skipped\n");
    free(file);
    return;
  }
  EXPECT(rayo_model_create(file, size, &m, NULL) == RAYO_OK);
  EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
  test_choose_kernels(s);
  before = test_allocations();
  for (i = 0; i < 1000; i++) {
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

static int run_tokens(rayo_session *s, const rayo_i32 *tokens, rayo_i32 past, rayo_i32 count) {
  rayo_tensor_view v;
  if (rayo_session_bind(s, "P", past, NULL) != RAYO_OK) return 0;
  if (rayo_session_bind(s, "T", count, NULL) != RAYO_OK) return 0;
  rayo_session_input(s, 0, &v);
  memcpy(v.data, tokens + past, (size_t)count * sizeof(rayo_i32));
  return rayo_session_run(s, NULL) == RAYO_OK;
}

static void decode(rayo_session *s, const rayo_i32 *tokens, rayo_i32 n, const rayo_json *want) {
  long before;
  rayo_i32 p;
  /* All at once: the logits after the last token. */
  rayo_session_reset(s);
  EXPECT(run_tokens(s, tokens, 0, n));
  EXPECT(compare(s, 0, want->items[n - 1]) < 1e-5);
  /* Two tokens, then one at a time, allocating nothing. */
  rayo_session_reset(s);
  EXPECT(run_tokens(s, tokens, 0, 2));
  EXPECT(compare(s, 0, want->items[1]) < 1e-5);
  before = test_allocations();
  for (p = 2; p < n; p++) {
    EXPECT(run_tokens(s, tokens, p, 1));
    EXPECT(compare(s, 0, want->items[p]) < 1e-5);
  }
  EXPECT(test_allocations() == before);
}

/* A decoder with a key/value cache: the whole sequence at once, and token by
 * token, give NumPy's logits; a run that does not fit the cache is refused;
 * steps allocate nothing. */
void test_decoder(void) {
  rayo_arena a;
  unsigned char *text, *file;
  rayo_json *root;
  const rayo_json *spec, *toks;
  rayo_i32 tokens[16], n, i;
  rayo_model *m;
  rayo_session *s;
  rayo_message msg;
  size_t size;
  a.head = NULL;
  a.bytes = 0;
  root = expected_json(&a, &text);
  file = test_model_file("decoder.weights", &size);
  EXPECT(root && file);
  if (!root || !file) {
    free(text);
    free(file);
    rayo_arena_free(&a);
    return;
  }
  spec = rayo_json_get(root, "decoder");
  toks = rayo_json_get(spec, "tokens");
  n = (rayo_i32)toks->length;
  for (i = 0; i < n && i < 16; i++) tokens[i] = (rayo_i32)toks->items[i]->number;
  EXPECT(rayo_model_create(file, size, &m, &msg) == RAYO_OK);
  EXPECT(rayo_session_create(m, &s, &msg) == RAYO_OK);
  test_choose_kernels(s);
  decode(s, tokens, n, rayo_json_get(spec, "logits"));
  decode(s, tokens, n, rayo_json_get(spec, "logits"));
  /* Past the cache's length. */
  EXPECT(rayo_session_bind(s, "P", n - 1, NULL) == RAYO_OK);
  EXPECT(rayo_session_bind(s, "T", 2, &msg) == RAYO_ERR_SHAPE);
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  rayo_arena_free(&a);
  free(text);
  free(file);
}
