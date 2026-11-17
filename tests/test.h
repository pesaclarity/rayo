/* A tiny test runner: each test is a function that reports failures with
 * EXPECT; main() in test_main.c runs them all. C89. */
#ifndef RAYO_TEST_H
#define RAYO_TEST_H

#include <stddef.h>
#include <stdio.h>

#include "rayo/rayo.h"

extern int rayo_test_failures;

#define EXPECT(cond)                                                      \
  do {                                                                    \
    if (!(cond)) {                                                        \
      fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
      rayo_test_failures++;                                               \
    }                                                                     \
  } while (0)

/* Reads a whole file; NULL if it cannot. free() it. */
unsigned char *test_read_file(const char *path, size_t *size);

/* A test model's file, from RAYO_TEST_MODELS or tests/models. free() it. */
unsigned char *test_model_file(const char *name, size_t *size);

/* Allocations made through malloc, calloc and realloc so far, or -1 when
 * the tests were built without counting them (alloc_count.c, linked with
 * --wrap, which only GNU-style linkers have). */
long test_allocations(void);

/* Makes `session` use the kernels RAYO_TEST_KERNELS names ("generic",
 * "simd"; anything else or unset: the session's own choice), so one build
 * can test each level. */
void test_choose_kernels(rayo_session *session);

#endif
