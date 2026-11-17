#include "core/alloc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

size_t rayo_align_up(size_t size) {
  if (size > SIZE_MAX - (RAYO_ALIGN - 1)) return 0;
  return (size + RAYO_ALIGN - 1) & ~(size_t)(RAYO_ALIGN - 1);
}

/* malloc'd block: [padding][original pointer][aligned data...] */
void *rayo_aligned_alloc(size_t size) {
  if (size == 0 || size > SIZE_MAX - RAYO_ALIGN - sizeof(void *)) return NULL;
  unsigned char *raw = malloc(size + RAYO_ALIGN + sizeof(void *));
  if (!raw) return NULL;
  uintptr_t start = (uintptr_t)(raw + sizeof(void *));
  uintptr_t aligned = (start + RAYO_ALIGN - 1) & ~(uintptr_t)(RAYO_ALIGN - 1);
  unsigned char *p = (unsigned char *)aligned;
  memcpy(p - sizeof(void *), &raw, sizeof raw);
  memset(p, 0, size);
  return p;
}

void rayo_aligned_free(void *p) {
  if (!p) return;
  void *raw;
  memcpy(&raw, (unsigned char *)p - sizeof(void *), sizeof raw);
  free(raw);
}
