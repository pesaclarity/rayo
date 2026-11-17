#include <string.h>

#include "ops/ops.h"

static const rayo_op *const all[] = {
    &rayo_op_add,      &rayo_op_attention,  &rayo_op_cache_write,   &rayo_op_compare,
    &rayo_op_concat,   &rayo_op_conv1d,     &rayo_op_dense,         &rayo_op_gather,
    &rayo_op_last_row, &rayo_op_layer_norm, &rayo_op_masked_reduce, &rayo_op_range,
    &rayo_op_softmax};

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
