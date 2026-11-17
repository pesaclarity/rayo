#define _POSIX_C_SOURCE 199309L
/* Runs the Ask reader on the recorded TFLite readings and times it:
 *
 *   ask_reader_check MODEL.weights ask_reader_decodes.json [repeats [generic|simd]]
 *
 * "generic" runs the portable C kernels, "simd" the build's SIMD ones
 * without the run-time dot-product choice; the default is what the session
 * picks.
 *
 * Exits 1 unless every intent and token label matches and every
 * probability is within 1e-4. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/arena.h"
#include "core/json.h"
#include "rayo/rayo.h"
#include "session.h"

static unsigned char *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *p = malloc((size_t)n + 1);
  if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
    free(p);
    p = NULL;
  }
  fclose(f);
  *size = (size_t)n;
  return p;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static size_t top(const float *p, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; i++) best = p[i] > p[best] ? i : best;
  return best;
}

static size_t top_json(const rayo_json *row) {
  size_t best = 0;
  for (size_t i = 1; i < row->length; i++)
    best = row->items[i]->number > row->items[best]->number ? i : best;
  return best;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL FIXTURE [repeats]\n", argv[0]);
    return 2;
  }
  int repeats = argc > 3 ? atoi(argv[3]) : 50;
  size_t model_size, fixture_size;
  unsigned char *bytes = read_file(argv[1], &model_size);
  unsigned char *fixture = read_file(argv[2], &fixture_size);
  if (!bytes || !fixture) {
    fprintf(stderr, "cannot read the files\n");
    return 2;
  }
  rayo_message msg = {{0}};
  double t0 = now_ms();
  rayo_model *model;
  rayo_status s = rayo_model_create(bytes, model_size, &model, &msg);
  double t1 = now_ms();
  if (s != RAYO_OK) {
    fprintf(stderr, "load: %s: %s\n", rayo_status_str(s), msg.text);
    return 1;
  }
  rayo_session *session;
  s = rayo_session_create(model, &session, &msg);
  double t2 = now_ms();
  if (argc > 4 && strcmp(argv[4], "generic") == 0) rayo_kernels_generic(&session->kernels);
#if defined(RAYO_KERNELS_X86) && (defined(__AVX2__) || defined(__SSE4_1__))
  if (argc > 4 && strcmp(argv[4], "simd") == 0) {
    session->kernels.name = "x86";
    session->kernels.a_offset = 0;
    session->kernels.gemm_i8 = rayo_gemm_i8_x86;
  }
#endif
  if (s != RAYO_OK) {
    fprintf(stderr, "session: %s\n", msg.text);
    return 1;
  }
  rayo_arena arena = {0};
  rayo_json *root;
  if (rayo_json_parse((const char *)fixture, fixture_size, &arena, &root, &msg) != RAYO_OK) {
    fprintf(stderr, "fixture: %s\n", msg.text);
    return 2;
  }
  const rayo_json *readings = rayo_json_get(root, "readings");
  rayo_tensor_view tokens, shapes, slots, intent;
  rayo_session_input(session, 0, &tokens);
  rayo_session_input(session, 1, &shapes);
  int labels = 0, total = 0;
  double worst = 0.0;
  size_t n = readings->length;
  double *times = malloc(sizeof(double) * n * (size_t)repeats);
  for (int rep = 0; rep < repeats; rep++) {
    for (size_t r = 0; r < n; r++) {
      const rayo_json *input = rayo_json_get(readings->items[r], "input");
      int32_t *tk = tokens.data, *sh = shapes.data;
      for (size_t i = 0; i < 96; i++) {
        tk[i] = (int32_t)input->items[i]->number;
        sh[i] = (int32_t)input->items[96 + i]->number;
      }
      double a = now_ms();
      s = rayo_session_run(session, &msg);
      times[(size_t)rep * n + r] = now_ms() - a;
      if (s != RAYO_OK) {
        fprintf(stderr, "run: %s\n", msg.text);
        return 1;
      }
      if (rep) continue;
      rayo_session_output(session, 0, &slots);
      rayo_session_output(session, 1, &intent);
      const rayo_json *want_i = rayo_json_get(readings->items[r], "intent");
      const rayo_json *want_s = rayo_json_get(readings->items[r], "slots");
      const float *pi = intent.data, *ps = slots.data;
      int ok = top(pi, want_i->length) == top_json(want_i);
      for (size_t i = 0; i < want_i->length; i++)
        worst = fmax(worst, fabs(pi[i] - want_i->items[i]->number));
      for (size_t t = 0; t < want_s->length; t++) {
        const rayo_json *row = want_s->items[t];
        ok &= top(ps + t * row->length, row->length) == top_json(row);
        for (size_t j = 0; j < row->length; j++) {
          worst = fmax(worst, fabs(ps[t * row->length + j] - row->items[j]->number));
        }
      }
      labels += ok;
      total++;
    }
  }
  qsort(times, n * (size_t)repeats, sizeof *times, cmp);
  rayo_memory_info mem;
  rayo_session_memory(session, &mem);
  printf("%s %s: labels %d/%d, max |p - TFLite| %.2e\n", rayo_version_string(),
         rayo_session_kernels(session), labels, total, worst);
  printf("load %.2f ms, session %.2f ms; per question median %.4f ms, p90 %.4f ms\n", t1 - t0,
         t2 - t1, times[n * (size_t)repeats / 2], times[n * (size_t)repeats * 9 / 10]);
  printf("memory: weights %zu, arena %zu, overhead %zu bytes\n", mem.weights, mem.arena,
         mem.overhead);
  /* Per op: the same loop as rayo_session_run, timed. */
  const rayo_graph *g = &model->graph;
  double *op_ms = calloc((size_t)g->node_count, sizeof *op_ms);
  const rayo_exec exec = {&session->kernels, session->memory + session->plan.scratch_offset,
                          session->symbols};
  for (int rep = 0; rep < repeats; rep++) {
    for (int32_t i = 0; i < g->node_count; i++) {
      rayo_tensor args[RAYO_MAX_INPUTS];
      rayo_graph_node_inputs(g, &model->pcwt, i, &session->values, args);
      double a = now_ms();
      g->nodes[i].op->run(g->nodes[i].params, args, &session->values.nodes[i], &exec);
      op_ms[i] += now_ms() - a;
    }
  }
  double sum = 0;
  for (int32_t i = 0; i < g->node_count; i++) sum += op_ms[i];
  for (int32_t i = 0; i < g->node_count; i++) {
    if (op_ms[i] / sum > 0.02) {
      printf("  %-8s %-12s %6.1f%%  %.4f ms\n", g->nodes[i].output, g->nodes[i].op->name,
             100 * op_ms[i] / sum, op_ms[i] / repeats);
    }
  }
  free(op_ms);
  free(times);
  rayo_arena_free(&arena);
  rayo_session_destroy(session);
  rayo_model_destroy(model);
  free(bytes);
  free(fixture);
  return labels == total && worst <= 1e-4 ? 0 : 1;
}
