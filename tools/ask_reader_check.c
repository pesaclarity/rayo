/* Runs the Ask reader on the recorded TFLite readings and times it:
 *
 *   ask_reader_check MODEL.weights ask_reader_decodes.json [repeats [generic|simd]]
 *
 * "generic" runs the portable C kernels, "simd" the build's SIMD ones
 * without the run-time dot-product choice; the default is what the session
 * picks. Prints the median and p90 per question and the time per op; exits
 * 1 unless every label matches and every probability is within 1e-4. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "core/json.h"

static double larger(double a, double b) {
  return a > b ? a : b;
}

static size_t top_f(const float *p, size_t n) {
  size_t i, best = 0;
  for (i = 1; i < n; i++) best = p[i] > p[best] ? i : best;
  return best;
}

static size_t top_j(const rayo_json *row) {
  size_t i, best = 0;
  for (i = 1; i < row->length; i++)
    best = row->items[i]->number > row->items[best]->number ? i : best;
  return best;
}

static void put(rayo_session *s, const rayo_json *input) {
  rayo_tensor_view tokens, shapes;
  size_t i;
  rayo_session_input(s, 0, &tokens);
  rayo_session_input(s, 1, &shapes);
  for (i = 0; i < 96; i++) {
    ((rayo_i32 *)tokens.data)[i] = (rayo_i32)input->items[i]->number;
    ((rayo_i32 *)shapes.data)[i] = (rayo_i32)input->items[96 + i]->number;
  }
}

/* Whether the outputs have TFLite's labels; widens *worst. */
static int check(const rayo_session *s, const rayo_json *r, double *worst) {
  const rayo_json *want_i = rayo_json_get(r, "intent"), *want_s = rayo_json_get(r, "slots");
  rayo_tensor_view slots, intent;
  const float *pi, *ps;
  int ok;
  size_t i, j;
  rayo_session_output(s, 0, &slots);
  rayo_session_output(s, 1, &intent);
  pi = intent.data;
  ps = slots.data;
  ok = top_f(pi, want_i->length) == top_j(want_i);
  for (i = 0; i < want_i->length; i++)
    *worst = larger(*worst, fabs(pi[i] - want_i->items[i]->number));
  for (i = 0; i < want_s->length; i++) {
    const rayo_json *row = want_s->items[i];
    ok &= top_f(ps + i * row->length, row->length) == top_j(row);
    for (j = 0; j < row->length; j++) {
      *worst = larger(*worst, fabs(ps[i * row->length + j] - row->items[j]->number));
    }
  }
  return ok;
}

int main(int argc, char **argv) {
  const int repeats = argc > 3 ? atoi(argv[3]) : 50;
  size_t model_size = 0, fixture_size = 0, n, r, total;
  unsigned char *bytes, *fixture;
  rayo_message msg;
  rayo_model *model;
  rayo_session *session;
  rayo_arena arena;
  rayo_json *root;
  const rayo_json *readings;
  rayo_memory_info mem;
  double t0, t1, t2, worst = 0.0, *times, *op_ms;
  int labels = 0, rep;
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL FIXTURE [repeats [generic|simd]]\n", argv[0]);
    return 2;
  }
  bytes = bench_read_file(argv[1], &model_size);
  fixture = bench_read_file(argv[2], &fixture_size);
  if (!bytes || !fixture) return 2;
  t0 = bench_now_ms();
  if (rayo_model_create(bytes, model_size, &model, &msg) != RAYO_OK) {
    fprintf(stderr, "load: %s\n", msg.text);
    return 1;
  }
  t1 = bench_now_ms();
  if (rayo_session_create(model, &session, &msg) != RAYO_OK) {
    fprintf(stderr, "session: %s\n", msg.text);
    return 1;
  }
  t2 = bench_now_ms();
  bench_choose_kernels(session, argc > 4 ? argv[4] : NULL);
  arena.head = NULL;
  arena.bytes = 0;
  if (rayo_json_parse((const char *)fixture, fixture_size, &arena, &root, &msg) != RAYO_OK) {
    fprintf(stderr, "fixture: %s\n", msg.text);
    return 2;
  }
  readings = rayo_json_get(root, "readings");
  n = readings->length;
  total = n * (size_t)repeats;
  times = malloc(sizeof(double) * total);
  op_ms = calloc((size_t)model->graph.node_count, sizeof *op_ms);
  for (rep = 0; rep < repeats; rep++) {
    for (r = 0; r < n; r++) {
      double a;
      put(session, rayo_json_get(readings->items[r], "input"));
      a = bench_now_ms();
      rayo_session_run(session, NULL);
      times[(size_t)rep * n + r] = bench_now_ms() - a;
      if (rep == 0) labels += check(session, readings->items[r], &worst);
      bench_run_ops(session, op_ms);
    }
  }
  rayo_session_memory(session, &mem);
  printf("%s %s: labels %d/%lu, max |p - TFLite| %.2e\n", rayo_version_string(),
         rayo_session_kernels(session), labels, (unsigned long)n, worst);
  printf("load %.2f ms, session %.2f ms; per question median %.4f ms, p90 %.4f ms\n", t1 - t0,
         t2 - t1, bench_quantile(times, total, 0.5), bench_quantile(times, total, 0.9));
  printf("memory: weights %lu, arena %lu, overhead %lu bytes\n", (unsigned long)mem.weights,
         (unsigned long)mem.arena, (unsigned long)mem.overhead);
  bench_print_ops(session, op_ms);
  free(times);
  free(op_ms);
  rayo_arena_free(&arena);
  rayo_session_destroy(session);
  rayo_model_destroy(model);
  free(bytes);
  free(fixture);
  return labels == (int)n && worst <= 1e-4 ? 0 : 1;
}
