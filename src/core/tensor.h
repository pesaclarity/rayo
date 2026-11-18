/* A tensor as ops see it: a dtype, a shape and memory (an activation in the
 * session's arena, or a weight in the model). */
#ifndef RAYO_CORE_TENSOR_H
#define RAYO_CORE_TENSOR_H

#include <stddef.h>

#include "format/pcwt.h"
#include "rayo/types.h"

/* An int8 activation may carry scales: `group` > 0 means each row's values
 * come in groups of `group` (along the innermost dimension), group g of row
 * r with the real values `scales[r * (inner / group) + g] * q` (a session's
 * int8 key/value cache). */
typedef struct rayo_tensor {
  rayo_dtype dtype;
  rayo_shape shape;
  void *data;
  const rayo_weight *weight; /* set when this is a weight */
  float *scales;             /* int8 with groups: see above */
  rayo_i32 group;            /* 0: no scales */
} rayo_tensor;

/* Bytes of one element of `dtype`. */
size_t rayo_dtype_size(rayo_dtype dtype);

/* The element count of `shape` (1 for rank 0). Shapes are checked when they
 * are made, so the count fits size_t. */
size_t rayo_shape_count(const rayo_shape *shape);

rayo_bool rayo_shape_equal(const rayo_shape *a, const rayo_shape *b);

/* The bytes of `t`'s elements. */
size_t rayo_tensor_bytes(const rayo_tensor *t);

/* The bytes `t` takes in memory: its elements, then (int8 with groups) its
 * scales, from a multiple of 4 bytes. */
size_t rayo_tensor_storage(const rayo_tensor *t);

/* Where a grouped int8 tensor's scales start, from its data. */
float *rayo_tensor_scales_at(void *data, const rayo_tensor *t);

/* The innermost dimension, and the product of the others (rank 0: 1 x 1). */
size_t rayo_shape_inner(const rayo_shape *shape);
size_t rayo_shape_outer(const rayo_shape *shape);

/* A tensor of a weight (its dtype, shape and data). */
rayo_tensor rayo_tensor_of_weight(const rayo_weight *w);

#endif
