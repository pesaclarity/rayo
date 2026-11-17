#include "core/arena.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"

#define RAYO_ARENA_BLOCK 8192

struct rayo_arena_block {
  rayo_arena_block *next;
  size_t used;
  size_t size;
  _Alignas(16) unsigned char data[];
};

void *rayo_arena_alloc(rayo_arena *a, size_t size) {
  if (size > SIZE_MAX - 15) return NULL;
  size = (size + 15) & ~(size_t)15;
  rayo_arena_block *b = a->head;
  if (!b || b->size - b->used < size) {
    size_t cap = size > RAYO_ARENA_BLOCK ? size : RAYO_ARENA_BLOCK;
    b = malloc(sizeof *b + cap);
    if (!b) return NULL;
    b->next = a->head;
    b->used = 0;
    b->size = cap;
    a->head = b;
    a->bytes += sizeof *b + cap;
  }
  void *p = b->data + b->used;
  b->used += size;
  memset(p, 0, size);
  return p;
}

void *rayo_arena_alloc_aligned(rayo_arena *a, size_t size) {
  if (size > SIZE_MAX - RAYO_ALIGN) return NULL;
  unsigned char *p = rayo_arena_alloc(a, size + RAYO_ALIGN - 16);
  if (!p) return NULL;
  uintptr_t at = ((uintptr_t)p + RAYO_ALIGN - 1) & ~(uintptr_t)(RAYO_ALIGN - 1);
  return (void *)at;
}

void *rayo_arena_array(rayo_arena *a, size_t count, size_t size) {
  if (size != 0 && count > SIZE_MAX / size) return NULL;
  return rayo_arena_alloc(a, count * size);
}

char *rayo_arena_strndup(rayo_arena *a, const char *s, size_t n) {
  if (n == SIZE_MAX) return NULL;
  char *p = rayo_arena_alloc(a, n + 1);
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
