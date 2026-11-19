/* slm_bench: a decoder's speed and memory: a prompt of P random tokens (in
 * chunks), then D tokens decoded greedily one at a time, with the model
 * mapped from its file, sized as a generator sizes it.
 *
 *   slm_bench MODEL.weights [P] [D] [threads] [--kv int8|float32]
 *             [--context N] [--chunk N] [--access normal|sequential|willneed|random]
 *             [--kernels generic|simd|best] [--runs N] [--json FILE] [--label TEXT]
 *
 * --context (default 1024) and --chunk (default 32) are at most the
 * model's; --kernels (default best) picks the kernel set, as the Ask
 * benchmarks do (simd: what an ARMv8.0 phone runs). Prints prompt and decode tokens a second, the
 * peak private memory (the process's own) and mapped-file memory, and the peak RSS.
 *
 * --runs N (default 1) runs the prompt and the decode N times, the cache
 * reset between, and reports the medians. --json FILE also writes the
 * results as JSON (tool/bench_phone.sh reads it): the model's load time
 * (mapping and checking the file) and the session's, prompt (prefill) and
 * decode tokens a second, each run's too, and the memory; --label names the
 * device in it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "bench.h"

#define MAX_RUNS 64

/* `s` as a JSON string (quotes, backslashes and control characters escaped). */
static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; s++) {
    const unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') {
      fputc('\\', f);
      fputc(c, f);
    } else if (c < 0x20) {
      fprintf(f, "\\u%04x", (unsigned)c);
    } else {
      fputc(c, f);
    }
  }
  fputc('"', f);
}

static void json_list(FILE *f, const double *v, int n) {
  int i;
  fputc('[', f);
  for (i = 0; i < n; i++) fprintf(f, "%s%.2f", i ? ", " : "", v[i]);
  fputc(']', f);
}

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

static rayo_process_memory_info peak;

static void sample(void) {
  rayo_process_memory_info m;
  if (rayo_process_memory(&m) != RAYO_OK) return;
  if (m.private_bytes > peak.private_bytes) peak.private_bytes = m.private_bytes;
  if (m.file_bytes > peak.file_bytes) peak.file_bytes = m.file_bytes;
  if (m.resident_bytes > peak.resident_bytes) peak.resident_bytes = m.resident_bytes;
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
  sample();
  return 1;
}

int main(int argc, char **argv) {
  rayo_model *m = NULL;
  rayo_session *s = NULL;
  rayo_session_options o;
  rayo_model_options mo;
  rayo_symbol_limit limits[2];
  rayo_tensor_view out, in;
  rayo_message msg;
  rayo_memory_info mem;
  rayo_i32 num[4] = {0, 128, 64, 1};
  rayo_i32 prompt, decode, chunk = 32, context = 1024, vocab, i, at = 0, *ids;
  int a, positional = 0, runs = 1, r;
  const char *path = NULL, *kernels = NULL, *json = NULL, *label = "";
  double t0, t_load, t_model, t_prompt, t_decode, rss_load;
  double prompt_tok_s[MAX_RUNS], decode_tok_s[MAX_RUNS], sorted[MAX_RUNS];
  memset(&o, 0, sizeof o);
  memset(&mo, 0, sizeof mo);
  for (a = 1; a < argc; a++) {
    if (strcmp(argv[a], "--kv") == 0 && a + 1 < argc) {
      o.cache = strcmp(argv[++a], "int8") == 0 ? RAYO_CACHE_INT8 : RAYO_CACHE_FLOAT32;
    } else if (strcmp(argv[a], "--context") == 0 && a + 1 < argc) {
      context = atoi(argv[++a]);
    } else if (strcmp(argv[a], "--chunk") == 0 && a + 1 < argc) {
      chunk = atoi(argv[++a]);
    } else if (strcmp(argv[a], "--kernels") == 0 && a + 1 < argc) {
      kernels = argv[++a];
    } else if (strcmp(argv[a], "--runs") == 0 && a + 1 < argc) {
      runs = atoi(argv[++a]);
      if (runs < 1) runs = 1;
      if (runs > MAX_RUNS) runs = MAX_RUNS;
    } else if (strcmp(argv[a], "--json") == 0 && a + 1 < argc) {
      json = argv[++a];
    } else if (strcmp(argv[a], "--label") == 0 && a + 1 < argc) {
      label = argv[++a];
    } else if (strcmp(argv[a], "--access") == 0 && a + 1 < argc) {
      const char *v = argv[++a];
      mo.access = strcmp(v, "normal") == 0       ? RAYO_ACCESS_NORMAL
                  : strcmp(v, "sequential") == 0 ? RAYO_ACCESS_SEQUENTIAL
                  : strcmp(v, "willneed") == 0   ? RAYO_ACCESS_WILLNEED
                  : strcmp(v, "random") == 0     ? RAYO_ACCESS_RANDOM
                                                 : RAYO_ACCESS_DEFAULT;
    } else if (!path) {
      path = argv[a];
    } else if (positional < 3) {
      num[1 + positional++] = atoi(argv[a]);
    }
  }
  if (!path) {
    fprintf(stderr,
            "usage: slm_bench MODEL.weights [prompt] [decode] [threads] [--kv int8|float32] "
            "[--context N] [--chunk N] [--access normal|sequential|willneed|random] "
            "[--kernels generic|simd|best] [--runs N] [--json FILE] [--label TEXT]\n");
    return 2;
  }
  prompt = num[1];
  decode = num[2];
  o.threads = num[3];
  t0 = bench_now_ms();
  if (rayo_model_load(path, &mo, &m, &msg) != RAYO_OK) {
    fprintf(stderr, "slm_bench: %s\n", msg.text);
    return 1;
  }
  t_model = bench_now_ms() - t0;
  /* As a generator sizes its session: the cache for `context` tokens, the
   * working memory for a chunk at its end. */
  rayo_model_input_info(m, 0, &in);
  if (chunk > in.shape.dims[0]) chunk = in.shape.dims[0];
  for (i = 0; i < m->graph.symbol_count; i++) {
    if (strcmp(m->graph.symbols[i], "P") == 0 && context > m->graph.symbol_max[i] + 1) {
      context = m->graph.symbol_max[i] + 1;
    }
  }
  if (prompt + decode > context) {
    fprintf(stderr, "slm_bench: %ld tokens do not fit a context of %ld\n", (long)(prompt + decode),
            (long)context);
    return 2;
  }
  o.state_rows = context;
  limits[0].name = "T";
  limits[0].max = limits[0].plan = chunk;
  limits[1].name = "P";
  limits[1].max = context - 1;
  limits[1].plan = context - chunk;
  o.limits = limits;
  o.limit_count = 2;
  if (rayo_session_create_with(m, &o, &s, &msg) != RAYO_OK) {
    fprintf(stderr, "slm_bench: %s\n", msg.text);
    return 1;
  }
  bench_choose_kernels(s, kernels);
  t_load = bench_now_ms() - t0;
  rss_load = peak_mb();
  sample();
  rayo_session_output(s, 0, &out);
  vocab = out.shape.dims[0];
  ids = malloc((size_t)(prompt + decode + 1) * sizeof *ids);
  if (!ids) return 1;
  srand(7);
  for (i = 0; i < prompt; i++) ids[i] = rand() % vocab;
  for (r = 0; r < runs; r++) {
    rayo_session_reset(s);
    at = 0;
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
    prompt_tok_s[r] = prompt * 1000.0 / t_prompt;
    decode_tok_s[r] = decode * 1000.0 / t_decode;
  }
  memcpy(sorted, prompt_tok_s, (size_t)runs * sizeof *sorted);
  t_prompt = bench_quantile(sorted, (size_t)runs, 0.5);
  memcpy(sorted, decode_tok_s, (size_t)runs * sizeof *sorted);
  t_decode = bench_quantile(sorted, (size_t)runs, 0.5);
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
      "%s threads=%d kernels=%s kv=%s context=%ld chunk=%ld load_ms=%.0f prompt=%ld "
      "prompt_tok_s=%.1f decode=%ld decode_tok_s=%.2f file_mb=%.1f arena_mb=%.1f "
      "rss_after_load_mb=%.1f peak_private_mb=%.1f peak_file_resident_mb=%.1f "
      "peak_rss_mb=%.1f runs=%d\n",
      rayo_model_name(m), rayo_session_threads(s), rayo_session_kernels(s),
      o.cache == RAYO_CACHE_INT8 ? "int8" : "float32", (long)context, (long)chunk, t_load,
      (long)prompt, t_prompt, (long)decode, t_decode, (double)mem.weights / 1048576.0,
      (double)mem.arena / 1048576.0, rss_load, (double)peak.private_bytes / 1048576.0,
      (double)peak.file_bytes / 1048576.0, peak_mb(), runs);
  if (json) {
    FILE *f = fopen(json, "w");
    if (!f) {
      fprintf(stderr, "slm_bench: cannot write %s\n", json);
      return 1;
    }
    fputs("{\n  \"tool\": \"slm_bench\",\n  \"rayo\": ", f);
    json_string(f, rayo_version_string());
    fputs(",\n  \"label\": ", f);
    json_string(f, label);
    fputs(",\n  \"model\": ", f);
    json_string(f, rayo_model_name(m));
    fputs(",\n  \"model_path\": ", f);
    json_string(f, path);
    fputs(",\n  \"kernels\": ", f);
    json_string(f, rayo_session_kernels(s));
    fprintf(f,
            ",\n  \"threads\": %d,\n  \"fast_cores\": %d,\n  \"kv\": \"%s\",\n"
            "  \"context\": %ld,\n  \"chunk\": %ld,\n  \"prompt_tokens\": %ld,\n"
            "  \"decode_tokens\": %ld,\n  \"runs\": %d,\n"
            "  \"model_load_ms\": %.2f,\n  \"session_ms\": %.2f,\n  \"load_ms\": %.2f,\n"
            "  \"prefill_tok_s\": %.2f,\n  \"decode_tok_s\": %.2f,\n",
            rayo_session_threads(s), rayo_cpu_fast_cores(),
            o.cache == RAYO_CACHE_INT8 ? "int8" : "float32", (long)context, (long)chunk,
            (long)prompt, (long)decode, runs, t_model, t_load - t_model, t_load, t_prompt,
            t_decode);
    fputs("  \"prefill_tok_s_runs\": ", f);
    json_list(f, prompt_tok_s, runs);
    fputs(",\n  \"decode_tok_s_runs\": ", f);
    json_list(f, decode_tok_s, runs);
    fprintf(f,
            ",\n  \"file_mb\": %.2f,\n  \"arena_mb\": %.2f,\n"
            "  \"rss_after_load_mb\": %.2f,\n  \"peak_private_mb\": %.2f,\n"
            "  \"peak_file_resident_mb\": %.2f,\n  \"peak_rss_mb\": %.2f\n}\n",
            (double)mem.weights / 1048576.0, (double)mem.arena / 1048576.0, rss_load,
            (double)peak.private_bytes / 1048576.0, (double)peak.file_bytes / 1048576.0, peak_mb());
    if (fclose(f) != 0) return 1;
  }
  free(ids);
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  return 0;
}
