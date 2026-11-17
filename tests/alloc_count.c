/* Counts the library's allocations: tool/check links the tests with
 * -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc and RAYO_TEST_COUNT_ALLOCS, so
 * every call in rayo's objects comes here first. GNU-style linkers only. */
#include <stddef.h>

#if defined(RAYO_TEST_COUNT_ALLOCS)

void *__real_malloc(size_t size);
void *__real_calloc(size_t n, size_t size);
void *__real_realloc(void *p, size_t size);
void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t n, size_t size);
void *__wrap_realloc(void *p, size_t size);
long test_allocations(void);

static long allocations;

void *__wrap_malloc(size_t size) {
  allocations++;
  return __real_malloc(size);
}

void *__wrap_calloc(size_t n, size_t size) {
  allocations++;
  return __real_calloc(n, size);
}

void *__wrap_realloc(void *p, size_t size) {
  allocations++;
  return __real_realloc(p, size);
}

long test_allocations(void) {
  return allocations;
}

#else
typedef int rayo_alloc_count_unused;
#endif
