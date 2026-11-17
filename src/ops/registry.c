#include <string.h>

#include "ops/ops.h"

static const rayo_op *const all[] = {
    &rayo_op_add,  &rayo_op_concat,     &rayo_op_conv1d,      &rayo_op_dense,   &rayo_op_gather,
    &rayo_op_mask, &rayo_op_masked_max, &rayo_op_masked_mean, &rayo_op_softmax,
};

const rayo_op *rayo_op_find(const char *name) {
  for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
    if (strcmp(all[i]->name, name) == 0) return all[i];
  }
  return NULL;
}
