/* Times the Ask writer decoding the recorded replies word by word, with its
 * key/value cache: the prefix in one run, then one run per reply token.
 *
 *   ask_writer_bench MODEL.weights ask_writer_decodes.json [repeats [generic|simd]]
 *
 * Prints the median and p90 per reply and per step, and the time per op. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "core/json.h"

static void run(rayo_session *s, const rayo_i32 *tokens, rayo_i32 past, rayo_i32 count,
                double *op_ms) {
  rayo_tensor_view v;
  rayo_session_bind(s, "P", past, NULL);
  rayo_session_bind(s, "T", count, NULL);
  rayo_session_input(s, 0, &v);
  memcpy(v.data, tokens + past, (size_t)count * sizeof(rayo_i32));
  if (op_ms) {
    bench_run_ops(s, op_ms);
  } else {
    rayo_session_run(s, NULL);
  }
}

/* The reply's tokens (prefix, then the recorded ids), at most 64. */
static rayo_i32 tokens_of(const rayo_json *d, rayo_i32 *tokens, rayo_i32 *n0) {
  const rayo_json *prefix = rayo_json_get(d, "prefix"), *ids = rayo_json_get(d, "ids");
  rayo_i32 len = 0;
  size_t i;
  for (i = 0; i < prefix->length && len < 64; i++)
    tokens[len++] = (rayo_i32)prefix->items[i]->number;
  *n0 = len;
  if (ids->kind == RAYO_JSON_ARRAY) {
    for (i = 0; i < ids->length && len < 64; i++) tokens[len++] = (rayo_i32)ids->items[i]->number;
  }
  return len;
}

int main(int argc, char **argv) {
  const int repeats = argc > 3 ? atoi(argv[3]) : 20;
  size_t model_size = 0, fixture_size = 0, n, r, total;
  unsigned char *bytes, *fixture;
  rayo_message msg;
  rayo_model *model;
  rayo_session *session;
  rayo_arena arena;
  rayo_json *root;
  const rayo_json *decodes;
  rayo_memory_info mem;
  double *reply, *step, *op_ms;
  long steps = 0;
  int rep;
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL FIXTURE [repeats [generic|simd]]\n", argv[0]);
    return 2;
  }
  bytes = bench_read_file(argv[1], &model_size);
  fixture = bench_read_file(argv[2], &fixture_size);
  if (!bytes || !fixture) return 2;
  if (rayo_model_create(bytes, model_size, &model, &msg) != RAYO_OK ||
      rayo_session_create(model, &session, &msg) != RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    return 1;
  }
  bench_choose_kernels(session, argc > 4 ? argv[4] : NULL);
  arena.head = NULL;
  arena.bytes = 0;
  if (rayo_json_parse((const char *)fixture, fixture_size, (size_t)-1, &arena, &root, &msg) !=
      RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    return 2;
  }
  decodes = rayo_json_get(root, "decodes");
  n = decodes->length;
  total = n * (size_t)repeats;
  reply = malloc(sizeof(double) * total);
  step = malloc(sizeof(double) * total);
  op_ms = calloc((size_t)model->graph.node_count, sizeof *op_ms);
  for (rep = 0; rep <= repeats; rep++) {
    for (r = 0; r < n; r++) {
      rayo_i32 tokens[64], n0, len, p;
      double *ops = rep == repeats ? op_ms : NULL, a, t;
      len = tokens_of(decodes->items[r], tokens, &n0);
      a = bench_now_ms();
      rayo_session_reset(session);
      run(session, tokens, 0, n0, ops);
      for (p = n0; p < len; p++) run(session, tokens, p, 1, ops);
      t = bench_now_ms() - a;
      if (rep < repeats) {
        reply[(size_t)rep * n + r] = t;
        step[(size_t)rep * n + r] = t / (double)(len - n0 + 1);
        steps += len - n0 + 1;
      }
    }
  }
  rayo_session_memory(session, &mem);
  printf("%s %s: %lu replies, %.1f steps each\n", rayo_version_string(),
         rayo_session_kernels(session), (unsigned long)n, (double)steps / (double)total);
  printf("per reply median %.3f ms, p90 %.3f ms; per step median %.4f ms\n",
         bench_quantile(reply, total, 0.5), bench_quantile(reply, total, 0.9),
         bench_quantile(step, total, 0.5));
  printf("memory: weights %lu, arena %lu, overhead %lu bytes\n", (unsigned long)mem.weights,
         (unsigned long)mem.arena, (unsigned long)mem.overhead);
  bench_print_ops(session, op_ms);
  free(reply);
  free(step);
  free(op_ms);
  rayo_arena_free(&arena);
  rayo_session_destroy(session);
  rayo_model_destroy(model);
  free(bytes);
  free(fixture);
  return 0;
}
