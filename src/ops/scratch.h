/* Carving an op's scratch memory into aligned pieces. `infer` adds up the
 * pieces' sizes with rayo_scratch_add; `run` takes them in the same order
 * with rayo_scratch_take. */
#ifndef RAYO_OPS_SCRATCH_H
#define RAYO_OPS_SCRATCH_H

#include <stddef.h>

#include "core/alloc.h"

/* Adds `bytes` (aligned) to `*total`, saturating at the largest size_t. */
void rayo_scratch_add(size_t *total, size_t bytes);

/* a * b * c, or the largest size_t when that overflows: for sizes that
 * are products of dimensions larger than any one tensor. */
size_t rayo_scratch_bytes(size_t a, size_t b, size_t c);

void *rayo_scratch_take(unsigned char **at, size_t bytes);

#endif
