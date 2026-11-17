/* Carving an op's scratch memory into aligned pieces. `infer` adds up the
 * pieces' sizes with rayo_scratch_add; `run` takes them in the same order
 * with rayo_scratch_take. */
#ifndef RAYO_OPS_SCRATCH_H
#define RAYO_OPS_SCRATCH_H

#include <stddef.h>

#include "core/alloc.h"

void rayo_scratch_add(size_t *total, size_t bytes);

void *rayo_scratch_take(unsigned char **at, size_t bytes);

#endif
