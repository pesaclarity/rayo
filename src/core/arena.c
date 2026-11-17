#include "core/arena.h"

#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"
#include "core/compat.h"

#define RAYO_ARENA_BLOCK 8192

/* A block: this header, then its 16-byte aligned memory. */
struct rayo_arena_block {
  rayo_arena_block *next;
  unsigned char *data;
  size_t used;
  size_t size;
};

void *rayo_arena_alloc(rayo_arena *a, size_t size) {
  rayo_arena_block *b;
  void *p;
  if (size > RAYO_SIZE_MAX - 64) return NULL;
  size = (size + 15) & ~(size_t)15;
  b = a->head;
  if (!b || b->size - b->used < size) {
    size_t cap = size > RAYO_ARENA_BLOCK ? size : RAYO_ARENA_BLOCK;
    b = (rayo_arena_block *)malloc(sizeof *b + cap + 15);
    if (!b) return NULL;
    b->data = (unsigned char *)rayo_align_pointer(b + 1, 16);
    b->next = a->head;
    b->used = 0;
    b->size = cap;
    a->head = b;
    a->bytes += sizeof *b + cap + 15;
  }
  p = b->data + b->used;
  b->used += size;
  memset(p, 0, size);
  return p;
}

void *rayo_arena_alloc_aligned(rayo_arena *a, size_t size) {
  unsigned char *p;
  if (size > RAYO_SIZE_MAX - RAYO_ALIGN) return NULL;
  p = (unsigned char *)rayo_arena_alloc(a, size + RAYO_ALIGN - 16);
  return p ? rayo_align_pointer(p, RAYO_ALIGN) : NULL;
}

void *rayo_arena_array(rayo_arena *a, size_t count, size_t size) {
  if (size != 0 && count > RAYO_SIZE_MAX / size) return NULL;
  return rayo_arena_alloc(a, count * size);
}

char *rayo_arena_strndup(rayo_arena *a, const char *s, size_t n) {
  char *p;
  if (n == RAYO_SIZE_MAX) return NULL;
  p = (char *)rayo_arena_alloc(a, n + 1);
  if (p && n) memcpy(p, s, n);
  return p;
}

void rayo_arena_free(rayo_arena *a) {
  rayo_arena_block *b = a->head;
  while (b) {
    rayo_arena_block *next = b->next;
    free(b);
    b = next;
  }
  a->head = NULL;
  a->bytes = 0;
}
