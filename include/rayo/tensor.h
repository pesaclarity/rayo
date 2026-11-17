/* rayo/tensor.h: a view of a tensor's memory. */
#ifndef RAYO_TENSOR_H
#define RAYO_TENSOR_H

#include "rayo/types.h"

/* A tensor the caller reads or writes in place. The memory belongs to the
 * session or model it came from and stays valid until that object is
 * destroyed; `shape` is the current one (after rayo_session_bind). */
typedef struct rayo_tensor_view {
  const char *name; /* owned by the model */
  rayo_dtype dtype;
  rayo_shape shape;
  void *data;
  size_t bytes; /* of `data` at the current shape */
} rayo_tensor_view;

#endif
