/* Times the Ask writer decoding the recorded replies word by word, with its
 * key/value cache: the prefix in one run, then one run per reply token.
 *
 *   ask_writer_bench MODEL.weights ask_writer_decodes.json [repeats]
 *
 * Prints the median and p90 per reply and per step, and the time per op. */
#define _POSIX_C_SOURCE 199309L
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

static void run(rayo_session *s, const int32_t *tokens, int64_t past, int64_t count,
                double *op_ms) {
  rayo_session_bind(s, "P", past, NULL);
  rayo_session_bind(s, "T", count, NULL);
  rayo_tensor_view v;
  rayo_session_input(s, 0, &v);
  memcpy(v.data, tokens + past, (size_t)count * sizeof(int32_t));
  if (!op_ms) {
    rayo_session_run(s, NULL);
    return;
  }
  const rayo_graph *g = &s->model->graph;
  const rayo_exec exec = {&s->kernels, s->memory + s->plan.scratch_offset, s->symbols};
  for (int32_t i = 0; i < g->node_count; i++) {
    rayo_tensor args[RAYO_MAX_INPUTS];
    rayo_graph_node_inputs(g, &s->model->pcwt, i, &s->values, args);
    double a = now_ms();
    g->nodes[i].op->run(g->nodes[i].params, args, &s->values.nodes[i], &exec);
    op_ms[i] += now_ms() - a;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL FIXTURE [repeats]\n", argv[0]);
    return 2;
  }
  const int repeats = argc > 3 ? atoi(argv[3]) : 20;
  size_t model_size, fixture_size;
  unsigned char *bytes = read_file(argv[1], &model_size);
  unsigned char *fixture = read_file(argv[2], &fixture_size);
  if (!bytes || !fixture) return 2;
  rayo_message msg = {{0}};
  rayo_model *model;
  rayo_session *session;
  if (rayo_model_create(bytes, model_size, &model, &msg) != RAYO_OK ||
      rayo_session_create(model, &session, &msg) != RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    return 1;
  }
  rayo_arena arena = {0};
  rayo_json *root;
  if (rayo_json_parse((const char *)fixture, fixture_size, &arena, &root, &msg) != RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    return 2;
  }
  const rayo_json *decodes = rayo_json_get(root, "decodes");
  const size_t n = decodes->length;
  double *reply = malloc(sizeof(double) * n * (size_t)repeats);
  double *step = malloc(sizeof(double) * n * (size_t)repeats);
  double *op_ms = calloc((size_t)model->graph.node_count, sizeof *op_ms);
  long steps = 0;
  for (int rep = 0; rep <= repeats; rep++) {
    for (size_t r = 0; r < n; r++) {
      const rayo_json *prefix = rayo_json_get(decodes->items[r], "prefix");
      const rayo_json *ids = rayo_json_get(decodes->items[r], "ids");
      int32_t tokens[64];
      int64_t len = 0;
      for (size_t i = 0; i < prefix->length && len < 64; i++) {
        tokens[len++] = (int32_t)prefix->items[i]->number;
      }
      const int64_t n0 = len;
      if (ids->kind == RAYO_JSON_ARRAY) {
        for (size_t i = 0; i < ids->length && len < 64; i++) {
          tokens[len++] = (int32_t)ids->items[i]->number;
        }
      }
      double *ops = rep == repeats ? op_ms : NULL;
      double a = now_ms();
      rayo_session_reset(session);
      run(session, tokens, 0, n0, ops);
      for (int64_t p = n0; p < len; p++) run(session, tokens, p, 1, ops);
      double t = now_ms() - a;
      if (rep < repeats) {
        reply[(size_t)rep * n + r] = t;
        step[(size_t)rep * n + r] = t / (double)(len - n0 + 1);
        steps += len - n0 + 1;
      }
    }
  }
  const size_t total = n * (size_t)repeats;
  qsort(reply, total, sizeof *reply, cmp);
  qsort(step, total, sizeof *step, cmp);
  rayo_memory_info mem;
  rayo_session_memory(session, &mem);
  printf("%s %s: %zu replies, %.1f steps each\n", rayo_version_string(),
         rayo_session_kernels(session), n, (double)steps / (double)total);
  printf("per reply median %.3f ms, p90 %.3f ms; per step median %.4f ms\n", reply[total / 2],
         reply[total * 9 / 10], step[total / 2]);
  printf("memory: weights %zu, arena %zu, overhead %zu bytes\n", mem.weights, mem.arena,
         mem.overhead);
  double sum = 0;
  for (int32_t i = 0; i < model->graph.node_count; i++) sum += op_ms[i];
  for (int32_t i = 0; i < model->graph.node_count; i++) {
    if (op_ms[i] / sum > 0.03) {
      printf("  %-8s %-12s %5.1f%%\n", model->graph.nodes[i].output, model->graph.nodes[i].op->name,
             100 * op_ms[i] / sum);
    }
  }
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
