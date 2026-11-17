/* A tensor as ops see it: a dtype, a shape and memory (an activation in the
 * session's arena, or a weight in the model). */
#ifndef RAYO_CORE_TENSOR_H
#define RAYO_CORE_TENSOR_H

#include <stddef.h>

#include "format/pcwt.h"
#include "rayo/types.h"

typedef struct rayo_tensor {
  rayo_dtype dtype;
  rayo_shape shape;
  void *data;
  const rayo_weight *weight; /* set when this is a weight */
} rayo_tensor;

/* Bytes of one element of `dtype`. */
size_t rayo_dtype_size(rayo_dtype dtype);

/* The element count of `shape` (1 for rank 0). Shapes are checked when they
 * are made, so the count fits size_t. */
size_t rayo_shape_count(const rayo_shape *shape);

rayo_bool rayo_shape_equal(const rayo_shape *a, const rayo_shape *b);

/* The bytes of `t`'s elements. */
size_t rayo_tensor_bytes(const rayo_tensor *t);

/* The innermost dimension, and the product of the others (rank 0: 1 x 1). */
size_t rayo_shape_inner(const rayo_shape *shape);
size_t rayo_shape_outer(const rayo_shape *shape);

/* A tensor of a weight (its dtype, shape and data). */
rayo_tensor rayo_tensor_of_weight(const rayo_weight *w);

#endif
