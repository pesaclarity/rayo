/* A tiny test runner: each test is a function that reports failures with
 * EXPECT; main() in test_main.c runs them all. */
#ifndef RAYO_TEST_H
#define RAYO_TEST_H

#include <stdio.h>

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

/* The directory of the test models, from RAYO_TEST_MODELS or tests/models. */
const char *test_models_dir(void);

/* Allocations made through malloc, calloc and realloc so far (counted by
 * alloc_count.c when the tests are linked with --wrap). */
long test_allocations(void);

#endif
