/* slm_bench: a decoder's speed and memory: a prompt of P random tokens (in
 * the model's chunks), then D tokens decoded greedily one at a time, with
 * the model mapped from its file.
 *
 *   slm_bench MODEL.weights [P] [D] [threads]
 *
 * Prints prompt and decode tokens a second and the peak resident memory. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "bench.h"

static double peak_mb(void) {
#if defined(__unix__) || defined(__APPLE__)
  struct rusage u;
  if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
  return (double)u.ru_maxrss / (1024.0 * 1024.0);
#else
  return (double)u.ru_maxrss / 1024.0;
#endif
#else
  return -1.0;
#endif
}

static rayo_i32 argmax(const float *v, rayo_i32 n) {
  rayo_i32 i, best = 0;
  for (i = 1; i < n; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

static int step(rayo_session *s, const rayo_i32 *ids, rayo_i32 p, rayo_i32 t) {
  rayo_tensor_view in;
  rayo_message msg;
  if (rayo_session_bind(s, "P", p, &msg) != RAYO_OK ||
      rayo_session_bind(s, "T", t, &msg) != RAYO_OK) {
    fprintf(stderr, "bind: %s\n", msg.text);
    return 0;
  }
  rayo_session_input(s, 0, &in);
  memcpy(in.data, ids, (size_t)t * sizeof *ids);
  if (rayo_session_run(s, &msg) != RAYO_OK) {
    fprintf(stderr, "run: %s\n", msg.text);
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  rayo_model *m = NULL;
  rayo_session *s = NULL;
  rayo_session_options o;
  rayo_tensor_view out, in;
  rayo_message msg;
  rayo_memory_info mem;
  rayo_i32 prompt = argc > 2 ? atoi(argv[2]) : 128, decode = argc > 3 ? atoi(argv[3]) : 64;
  rayo_i32 chunk, vocab, i, at = 0, *ids;
  double t0, t_load, t_prompt, t_decode, rss_load;
  if (argc < 2) {
    fprintf(stderr, "usage: slm_bench MODEL.weights [prompt] [decode] [threads]\n");
    return 2;
  }
  memset(&o, 0, sizeof o);
  o.threads = argc > 4 ? atoi(argv[4]) : 1;
  t0 = bench_now_ms();
  if (rayo_model_load(argv[1], NULL, &m, &msg) != RAYO_OK ||
      rayo_session_create_with(m, &o, &s, &msg) != RAYO_OK) {
    fprintf(stderr, "slm_bench: %s\n", msg.text);
    return 1;
  }
  t_load = bench_now_ms() - t0;
  rss_load = peak_mb();
  rayo_model_input_info(m, 0, &in);
  chunk = in.shape.dims[0];
  rayo_session_output(s, 0, &out);
  vocab = out.shape.dims[0];
  ids = malloc((size_t)(prompt + decode + 1) * sizeof *ids);
  if (!ids) return 1;
  srand(7);
  for (i = 0; i < prompt; i++) ids[i] = rand() % vocab;
  t0 = bench_now_ms();
  while (at < prompt) {
    const rayo_i32 c = prompt - at < chunk ? prompt - at : chunk;
    if (!step(s, ids + at, at, c)) return 1;
    at += c;
  }
  t_prompt = bench_now_ms() - t0;
  t0 = bench_now_ms();
  for (i = 0; i < decode; i++) {
    rayo_session_output(s, 0, &out);
    ids[at] = argmax(out.data, vocab);
    if (!step(s, ids + at, at, 1)) return 1;
    at++;
  }
  t_decode = bench_now_ms() - t0;
  if (getenv("RAYO_BENCH_OPS")) {
    /* Where a decode step's time goes, op by op. */
    double *op_ms = calloc((size_t)s->model->graph.node_count, sizeof *op_ms);
    const int whole = strcmp(getenv("RAYO_BENCH_OPS"), "prompt") == 0;
    for (i = 0; op_ms && i < (whole ? 2 : 16); i++) {
      rayo_session_bind(s, "P", whole ? 0 : at - 1, NULL);
      rayo_session_bind(s, "T", whole ? chunk : 1, NULL);
      bench_run_ops(s, op_ms);
    }
    printf("time by op, %s:\n", whole ? "a prompt chunk" : "a decode step");
    if (op_ms) bench_print_op_kinds(s, op_ms);
    free(op_ms);
  }
  rayo_session_memory(s, &mem);
  printf(
      "%s threads=%d kernels=%s load_ms=%.0f prompt=%ld prompt_tok_s=%.1f decode=%ld "
      "decode_tok_s=%.2f file_mb=%.1f arena_mb=%.1f rss_after_load_mb=%.1f peak_rss_mb=%.1f\n",
      rayo_model_name(m), rayo_session_threads(s), rayo_session_kernels(s), t_load, (long)prompt,
      prompt * 1000.0 / t_prompt, (long)decode, decode * 1000.0 / t_decode,
      (double)mem.weights / 1048576.0, (double)mem.arena / 1048576.0, rss_load, peak_mb());
  free(ids);
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  return 0;
}
