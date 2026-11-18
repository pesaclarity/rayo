#define _POSIX_C_SOURCE 199309L
#include "bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

unsigned char *bench_read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  long n;
  unsigned char *p;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  n = ftell(f);
  fseek(f, 0, SEEK_SET);
  p = n >= 0 ? (unsigned char *)malloc((size_t)n + 1) : NULL;
  if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
    free(p);
    p = NULL;
  }
  fclose(f);
  if (p) {
    p[n] = 0;
    *size = (size_t)n;
  }
  return p;
}

double bench_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

double bench_quantile(double *v, size_t n, double fraction) {
  qsort(v, n, sizeof *v, cmp);
  return v[(size_t)((double)n * fraction)];
}

void bench_run_ops(rayo_session *s, double *op_ms) {
  const rayo_graph *g = &s->model->graph;
  rayo_exec exec;
  rayo_i32 i;
  exec.kernels = &s->kernels;
  exec.scratch = s->memory + s->plan.scratch_offset;
  exec.symbols = s->symbols;
  exec.pool = s->pool;
  for (i = 0; i < g->node_count; i++) {
    double a;
    rayo_graph_node_inputs(g, &s->model->pcwt, i, &s->values, s->values.args);
    a = bench_now_ms();
    g->nodes[i].op->run(g->nodes[i].params, s->values.args, &s->values.nodes[i], &exec);
    op_ms[i] += bench_now_ms() - a;
  }
}

void bench_print_ops(const rayo_session *s, const double *op_ms) {
  const rayo_graph *g = &s->model->graph;
  double sum = 0;
  rayo_i32 i;
  for (i = 0; i < g->node_count; i++) sum += op_ms[i];
  for (i = 0; i < g->node_count; i++) {
    if (op_ms[i] / sum > 0.03) {
      printf("  %-14s %-12s %5.1f%%\n", g->nodes[i].output, g->nodes[i].op->name,
             100 * op_ms[i] / sum);
    }
  }
}

void bench_choose_kernels(rayo_session *s, const char *name) {
  if (name && strcmp(name, "generic") == 0) rayo_kernels_generic(&s->kernels);
  if (name && strcmp(name, "simd") == 0) rayo_kernels_simd(&s->kernels);
}

void bench_print_op_kinds(const rayo_session *s, const double *op_ms) {
  const rayo_graph *g = &s->model->graph;
  const char *seen[64];
  double ms[64], sum = 0;
  int n = 0, k;
  rayo_i32 i;
  for (i = 0; i < g->node_count; i++) {
    for (k = 0; k < n && strcmp(seen[k], g->nodes[i].op->name) != 0; k++) {
    }
    if (k == n && n < 64) {
      seen[n] = g->nodes[i].op->name;
      ms[n++] = 0;
    }
    if (k < n) ms[k] += op_ms[i];
    sum += op_ms[i];
  }
  for (k = 0; k < n; k++) printf("  %-12s %5.1f%%\n", seen[k], 100 * ms[k] / sum);
}
