#include <string.h>

#include "ops/ops.h"

static const rayo_op *const all[] = {&rayo_op_abs,           &rayo_op_add,
                                     &rayo_op_attention,     &rayo_op_cache_write,
                                     &rayo_op_compare,       &rayo_op_concat,
                                     &rayo_op_conv1d,        &rayo_op_dense,
                                     &rayo_op_div,           &rayo_op_exp,
                                     &rayo_op_gather,        &rayo_op_gelu_erf,
                                     &rayo_op_gelu_tanh,     &rayo_op_last_row,
                                     &rayo_op_layer_norm,    &rayo_op_log,
                                     &rayo_op_masked_reduce, &rayo_op_matmul,
                                     &rayo_op_maximum,       &rayo_op_minimum,
                                     &rayo_op_mul,           &rayo_op_neg,
                                     &rayo_op_range,         &rayo_op_reduce,
                                     &rayo_op_relu,          &rayo_op_relu6,
                                     &rayo_op_reshape,       &rayo_op_rsqrt,
                                     &rayo_op_sigmoid,       &rayo_op_slice,
                                     &rayo_op_softmax,       &rayo_op_sqrt,
                                     &rayo_op_square,        &rayo_op_squared_difference,
                                     &rayo_op_sub,           &rayo_op_tanh,
                                     &rayo_op_transpose};

const char *rayo_op_builtin(int index) {
  if (index < 0 || (size_t)index >= sizeof all / sizeof all[0]) return NULL;
  return all[index]->name;
}

const rayo_op *rayo_op_find(const char *name) {
  size_t i;
  for (i = 0; i < sizeof all / sizeof all[0]; i++) {
    if (strcmp(all[i]->name, name) == 0) return all[i];
  }
  return NULL;
}
