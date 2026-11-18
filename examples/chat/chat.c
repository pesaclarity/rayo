/* chat: talk to a small language model converted by rayo-convert, on the
 * CPU, offline. docs/RUNNING-SLMS.md shows the whole way, from Hugging Face
 * to a phone.
 *
 *   chat MODEL.weights                       a conversation, a line a turn
 *   chat MODEL.weights -p "Once upon a time" one raw prompt, continued
 *
 * Replies stream to standard output; prompts, notes and the numbers (prompt
 * and reply tokens a second, peak memory) go to standard error. In a
 * conversation, /reset starts again and /quit (or the end of input) stops.
 *
 * C89 with rayo's public API only. The clock and the memory peak use POSIX
 * (or Win32) where there is one. */
#if defined(__unix__) || defined(__APPLE__)
#define CHAT_POSIX 1
#define _POSIX_C_SOURCE 200112L
#elif defined(_WIN32)
#define CHAT_WIN32 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rayo/rayo.h"

#if defined(CHAT_POSIX)
#include <sys/resource.h>
#include <sys/time.h>
#elif defined(CHAT_WIN32)
#include <psapi.h>
#include <windows.h>
#endif

#define LINE 8192

static double now_s(void) {
#if defined(CHAT_POSIX)
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#elif defined(CHAT_WIN32)
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* The process's peak resident memory in MB, or -1. */
static double peak_mb(void) {
#if defined(CHAT_POSIX)
  struct rusage u;
  if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
  return (double)u.ru_maxrss / (1024.0 * 1024.0); /* bytes */
#else
  return (double)u.ru_maxrss / 1024.0; /* KB */
#endif
#elif defined(CHAT_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return -1.0;
  return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
#else
  return -1.0;
#endif
}

typedef struct stats {
  double prompt_s, reply_s;
  long prompt_tokens, reply_tokens;
} stats;

static void usage(void) {
  fprintf(stderr,
          "usage: chat MODEL.weights [options]\n"
          "  --tokenizer FILE     its tokenizer.json (default MODEL.tokenizer.json)\n"
          "  -p, --prompt TEXT    continue one raw prompt instead of a conversation\n"
          "  --system TEXT        a system text for the conversation\n"
          "  --threads N          threads (default: the fast cores, at most 4)\n");
  fprintf(stderr,
          "  --temp T             temperature (default 0.7; 0: greedy)\n"
          "  --top-k K            (default 40)    --top-p P   (default 0.9)\n"
          "  --min-p P            (default 0.05)  --repeat-penalty R (default 1.1)\n"
          "  --seed S             (default 1)\n"
          "  --max-tokens N       per reply (default 256)\n");
}

/* Streams the reply; its tokens and time go to `st`. */
static rayo_status reply(rayo_generator *g, const rayo_tokenizer *tok, stats *st,
                         rayo_message *msg) {
  rayo_i32 token;
  const double t0 = now_s();
  long n = 0;
  rayo_status s;
  for (;;) {
    size_t size;
    const char *piece;
    s = rayo_generator_next(g, &token, msg);
    if (s != RAYO_OK || token < 0) break;
    piece = rayo_tokenizer_piece(tok, token, &size);
    if (piece) fwrite(piece, 1, size, stdout);
    fflush(stdout);
    n++;
  }
  fputc('\n', stdout);
  fflush(stdout);
  st->reply_s += now_s() - t0;
  st->reply_tokens += n;
  return s;
}

static void print_stats(const rayo_generator *g, const rayo_model *m, const stats *st) {
  rayo_memory_info mem;
  rayo_model_memory(m, &mem);
  fprintf(stderr, "\n-- %s, %d thread%s, kernels %s\n", rayo_model_name(m),
          rayo_generator_threads(g), rayo_generator_threads(g) == 1 ? "" : "s",
          rayo_generator_kernels(g));
  if (st->prompt_tokens) {
    fprintf(stderr, "prompt: %ld tokens, %.1f tokens/s\n", st->prompt_tokens,
            (double)st->prompt_tokens / (st->prompt_s > 0 ? st->prompt_s : 1e-9));
  }
  if (st->reply_tokens) {
    fprintf(stderr, "reply:  %ld tokens, %.1f tokens/s\n", st->reply_tokens,
            (double)st->reply_tokens / (st->reply_s > 0 ? st->reply_s : 1e-9));
  }
  fprintf(stderr, "context: %ld of %ld tokens; model file %.1f MB (mapped)\n",
          (long)rayo_generator_position(g), (long)rayo_generator_context(g),
          (double)mem.weights / (1024.0 * 1024.0));
  if (peak_mb() >= 0) fprintf(stderr, "peak memory (RSS): %.1f MB\n", peak_mb());
}

static void trim_newline(char *s) {
  size_t n = strlen(s);
  while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

int main(int argc, char **argv) {
  const char *model_path = NULL, *tok_path = NULL, *prompt = NULL, *system = NULL;
  char tok_buf[1024], *line;
  rayo_generator_options o;
  rayo_model *m = NULL;
  rayo_tokenizer *tok = NULL;
  rayo_generator *g = NULL;
  rayo_message msg;
  rayo_status s;
  stats st;
  double t0;
  int i, threads = 0;
  memset(&o, 0, sizeof o);
  memset(&st, 0, sizeof st);
  o.sampling.temperature = 0.7f;
  o.sampling.top_k = 40;
  o.sampling.top_p = 0.9f;
  o.sampling.min_p = 0.05f;
  o.sampling.repetition_penalty = 1.1f;
  o.sampling.seed = 1;
  o.max_tokens = 256;
  for (i = 1; i < argc; i++) {
    const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
    if (a[0] != '-') {
      model_path = a;
      continue;
    }
    if (!v) {
      usage();
      return 2;
    }
    i++;
    if (strcmp(a, "--tokenizer") == 0) {
      tok_path = v;
    } else if (strcmp(a, "-p") == 0 || strcmp(a, "--prompt") == 0) {
      prompt = v;
    } else if (strcmp(a, "--system") == 0) {
      system = v;
    } else if (strcmp(a, "--threads") == 0) {
      threads = atoi(v);
    } else if (strcmp(a, "--temp") == 0) {
      o.sampling.temperature = (float)atof(v);
    } else if (strcmp(a, "--top-k") == 0) {
      o.sampling.top_k = atoi(v);
    } else if (strcmp(a, "--top-p") == 0) {
      o.sampling.top_p = (float)atof(v);
    } else if (strcmp(a, "--min-p") == 0) {
      o.sampling.min_p = (float)atof(v);
    } else if (strcmp(a, "--repeat-penalty") == 0) {
      o.sampling.repetition_penalty = (float)atof(v);
    } else if (strcmp(a, "--seed") == 0) {
      o.sampling.seed = (rayo_u32)strtoul(v, NULL, 10);
    } else if (strcmp(a, "--max-tokens") == 0) {
      o.max_tokens = atoi(v);
    } else {
      usage();
      return 2;
    }
  }
  if (!model_path) {
    usage();
    return 2;
  }
  if (!tok_path) {
    /* MODEL.weights -> MODEL.tokenizer.json, as rayo-convert writes it. */
    size_t n = strlen(model_path);
    const char *dot = strrchr(model_path, '.');
    if (dot && strcmp(dot, ".weights") == 0) n = (size_t)(dot - model_path);
    if (n + 16 > sizeof tok_buf) return 2;
    memcpy(tok_buf, model_path, n);
    strcpy(tok_buf + n, ".tokenizer.json");
    tok_path = tok_buf;
  }
  /* Few threads by default: a phone's big cores, at most 4, which is where
   * the memory bandwidth runs out and the battery starts to pay. */
  o.threads = threads > 0 ? threads : rayo_cpu_fast_cores();
  if (threads <= 0 && o.threads > 4) o.threads = 4;
  t0 = now_s();
  s = rayo_model_load(model_path, NULL, &m, &msg);
  if (s == RAYO_OK) s = rayo_tokenizer_load(tok_path, &tok, &msg);
  if (s == RAYO_OK) s = rayo_generator_create(m, tok, &o, &g, &msg);
  if (s != RAYO_OK) {
    fprintf(stderr, "chat: %s\n", msg.text);
    rayo_tokenizer_destroy(tok);
    rayo_model_destroy(m);
    return 1;
  }
  fprintf(stderr, "-- loaded in %.2f s; %s\n", now_s() - t0,
          prompt || rayo_generator_can_chat(g) ? "ready" : "no chat template: use -p");
  if (prompt) {
    t0 = now_s();
    s = rayo_generator_prompt(g, prompt, strlen(prompt), &msg);
    st.prompt_s += now_s() - t0;
    st.prompt_tokens += rayo_generator_position(g);
    fputs(prompt, stdout);
    if (s == RAYO_OK) s = reply(g, tok, &st, &msg);
  } else {
    line = malloc(LINE);
    while (line && s == RAYO_OK) {
      rayo_i32 before;
      fputs("> ", stderr);
      fflush(stderr);
      if (!fgets(line, LINE, stdin)) break;
      trim_newline(line);
      if (strcmp(line, "/quit") == 0) break;
      if (strcmp(line, "/reset") == 0) {
        rayo_generator_reset(g);
        fputs("-- started again\n", stderr);
        continue;
      }
      if (!*line) continue;
      before = rayo_generator_position(g);
      t0 = now_s();
      s = rayo_generator_chat(g, rayo_generator_position(g) == 0 ? system : NULL, line, &msg);
      st.prompt_s += now_s() - t0;
      st.prompt_tokens += rayo_generator_position(g) - before;
      if (s == RAYO_ERR_SHAPE) {
        fprintf(stderr, "-- %s; /reset to start again\n", msg.text);
        s = RAYO_OK;
        continue;
      }
      if (s == RAYO_OK) s = reply(g, tok, &st, &msg);
    }
    free(line);
  }
  if (s != RAYO_OK) fprintf(stderr, "chat: %s\n", msg.text);
  print_stats(g, m, &st);
  rayo_generator_destroy(g);
  rayo_tokenizer_destroy(tok);
  rayo_model_destroy(m);
  return s == RAYO_OK ? 0 : 1;
}
