/* An arena: many small allocations with one owner, freed together. A model
 * keeps its parsed header and graph in one. */
#ifndef RAYO_CORE_ARENA_H
#define RAYO_CORE_ARENA_H

#include <stddef.h>

typedef struct rayo_arena_block rayo_arena_block;

typedef struct rayo_arena {
  rayo_arena_block *head;
  size_t bytes; /* held, for rayo_memory_info */
} rayo_arena;

/* Zeroed, 16-byte aligned memory of `size` bytes owned by `a`; NULL when out
 * of memory. */
void *rayo_arena_alloc(rayo_arena *a, size_t size);

/* Zeroed memory aligned to RAYO_ALIGN (64), for buffers SIMD kernels read. */
void *rayo_arena_alloc_aligned(rayo_arena *a, size_t size);

/* `count` zeroed elements of `size` bytes; NULL on overflow or out of
 * memory. */
void *rayo_arena_array(rayo_arena *a, size_t count, size_t size);

/* A NUL-terminated copy of the `n` bytes at `s`. */
char *rayo_arena_strndup(rayo_arena *a, const char *s, size_t n);

/* Frees every allocation (`a` may be zeroed or already freed). */
void rayo_arena_free(rayo_arena *a);

#endif
