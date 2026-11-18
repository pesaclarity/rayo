#include "core/tensor.h"

size_t rayo_dtype_size(rayo_dtype dtype) {
  switch (dtype) {
    case RAYO_DTYPE_F32:
    case RAYO_DTYPE_I32:
      return 4;
    case RAYO_DTYPE_I8:
      return 1;
    case RAYO_DTYPE_I4: /* weights only, never planned */
    case RAYO_DTYPE_I3:
      return 0;
  }
  return 0;
}

size_t rayo_shape_count(const rayo_shape *shape) {
  size_t n = 1;
  rayo_i32 i;
  for (i = 0; i < shape->rank; i++) n *= (size_t)shape->dims[i];
  return n;
}

rayo_bool rayo_shape_equal(const rayo_shape *a, const rayo_shape *b) {
  rayo_i32 i;
  if (a->rank != b->rank) return RAYO_FALSE;
  for (i = 0; i < a->rank; i++) {
    if (a->dims[i] != b->dims[i]) return RAYO_FALSE;
  }
  return RAYO_TRUE;
}

size_t rayo_tensor_bytes(const rayo_tensor *t) {
  return rayo_shape_count(&t->shape) * rayo_dtype_size(t->dtype);
}

size_t rayo_tensor_storage(const rayo_tensor *t) {
  const size_t n = rayo_tensor_bytes(t);
  if (t->dtype != RAYO_DTYPE_I8 || t->group <= 0) return n;
  return (n + 3) / 4 * 4 + rayo_shape_count(&t->shape) / (size_t)t->group * sizeof(float);
}

float *rayo_tensor_scales_at(void *data, const rayo_tensor *t) {
  return (float *)(void *)((unsigned char *)data + (rayo_tensor_bytes(t) + 3) / 4 * 4);
}

size_t rayo_shape_inner(const rayo_shape *shape) {
  return shape->rank == 0 ? 1 : (size_t)shape->dims[shape->rank - 1];
}

size_t rayo_shape_outer(const rayo_shape *shape) {
  size_t n = 1;
  rayo_i32 i;
  for (i = 0; i + 1 < shape->rank; i++) n *= (size_t)shape->dims[i];
  return n;
}

rayo_tensor rayo_tensor_of_weight(const rayo_weight *w) {
  rayo_tensor t;
  t.dtype = w->dtype;
  t.shape = w->shape;
  t.data = (void *)w->data; /* weights are read-only; ops never write them */
  t.weight = w;
  t.scales = NULL;
  t.group = 0;
  return t;
}
