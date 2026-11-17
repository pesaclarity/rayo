/* Aligned heap memory. Every buffer rayo's kernels read is aligned to
 * RAYO_ALIGN bytes. */
#ifndef RAYO_CORE_ALLOC_H
#define RAYO_CORE_ALLOC_H

#include <stddef.h>

#define RAYO_ALIGN 64

/* `size` rounded up to a multiple of RAYO_ALIGN; 0 if that overflows. */
size_t rayo_align_up(size_t size);

/* `size` bytes aligned to RAYO_ALIGN, zeroed; NULL when out of memory or
 * `size` is 0. Free with rayo_aligned_free. */
void *rayo_aligned_alloc(size_t size);

/* Frees memory from rayo_aligned_alloc (NULL is fine). */
void rayo_aligned_free(void *p);

#endif
