#include "core/tensor.h"

size_t rayo_dtype_size(rayo_dtype dtype) {
  switch (dtype) {
    case RAYO_DTYPE_F32:
    case RAYO_DTYPE_I32:
      return 4;
    case RAYO_DTYPE_I8:
      return 1;
  }
  return 0;
}

int64_t rayo_shape_count(const rayo_shape *shape) {
  int64_t n = 1;
  for (int32_t i = 0; i < shape->rank; i++) n *= shape->dims[i];
  return n;
}

bool rayo_shape_equal(const rayo_shape *a, const rayo_shape *b) {
  if (a->rank != b->rank) return false;
  for (int32_t i = 0; i < a->rank; i++) {
    if (a->dims[i] != b->dims[i]) return false;
  }
  return true;
}

size_t rayo_tensor_bytes(const rayo_tensor *t) {
  return (size_t)rayo_shape_count(&t->shape) * rayo_dtype_size(t->dtype);
}

int64_t rayo_shape_inner(const rayo_shape *shape) {
  return shape->rank == 0 ? 1 : shape->dims[shape->rank - 1];
}

int64_t rayo_shape_outer(const rayo_shape *shape) {
  int64_t n = 1;
  for (int32_t i = 0; i + 1 < shape->rank; i++) n *= shape->dims[i];
  return n;
}
