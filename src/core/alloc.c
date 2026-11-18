#include "core/alloc.h"

#include <stdlib.h>
#include <string.h>

#include "core/compat.h"

size_t rayo_align_up(size_t size) {
  if (size > RAYO_SIZE_MAX - (RAYO_ALIGN - 1)) return 0;
  return (size + RAYO_ALIGN - 1) & ~(size_t)(RAYO_ALIGN - 1);
}

void *rayo_align_pointer(void *p, size_t align) {
  unsigned char *b = (unsigned char *)p;
  size_t off = (size_t)b % align; /* a pointer's address, as C89 allows */
  return off ? b + (align - off) : b;
}

/* malloc'd block: [padding][original pointer][aligned data...] */
void *rayo_aligned_alloc(size_t size) {
  unsigned char *raw;
  unsigned char *p;
  if (size == 0 || size > RAYO_SIZE_MAX - RAYO_ALIGN - sizeof(void *)) return NULL;
  /* calloc, not malloc and memset: a large block comes zeroed from the
   * system, and its pages cost memory only once they are written. */
  raw = (unsigned char *)calloc(1, size + RAYO_ALIGN + sizeof(void *));
  if (!raw) return NULL;
  p = (unsigned char *)rayo_align_pointer(raw + sizeof(void *), RAYO_ALIGN);
  memcpy(p - sizeof(void *), &raw, sizeof raw);
  return p;
}

void rayo_aligned_free(void *p) {
  void *raw;
  if (!p) return;
  memcpy(&raw, (unsigned char *)p - sizeof(void *), sizeof raw);
  free(raw);
}
