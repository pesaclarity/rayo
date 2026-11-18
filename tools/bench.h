/* Shared by the tools: files, a clock, medians, and running a session op
 * by op (timed). C89 with POSIX clock_gettime. */
#ifndef RAYO_TOOLS_BENCH_H
#define RAYO_TOOLS_BENCH_H

#include <stddef.h>

#include "rayo/rayo.h"
#include "session.h"

/* A whole file, NUL-terminated; NULL if it cannot be read. free() it. */
unsigned char *bench_read_file(const char *path, size_t *size);

/* Milliseconds from a monotonic clock. */
double bench_now_ms(void);

/* Sorts `v` and gives its value at `fraction` (0.5: the median). */
double bench_quantile(double *v, size_t n, double fraction);

/* Runs `s` like rayo_session_run, adding each node's time to op_ms. */
void bench_run_ops(rayo_session *s, double *op_ms);

/* Prints the ops above 3% of `op_ms`'s total. */
void bench_print_ops(const rayo_session *s, const double *op_ms);

/* The share of `op_ms` each op (dense, attention, ...) takes. */
void bench_print_op_kinds(const rayo_session *s, const double *op_ms);

/* Sets `s`'s kernels by name ("generic", "simd"; else the session's own). */
void bench_choose_kernels(rayo_session *s, const char *name);

#endif
