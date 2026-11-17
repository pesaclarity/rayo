/* concat: two float32 tensors side by side along the innermost dimension
 * ([..., a] and [..., b] to [..., a + b]). */
#include <string.h>

#include "core/message.h"
#include "ops/ops.h"

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  (void)arena;
  if (spec->inputs[0].weight || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: concat takes activations", spec->output);
  }
  *params = NULL;
  return RAYO_OK;
}

static rayo_status infer(const void *params, const rayo_tensor *in, rayo_tensor *out,
                         size_t *scratch, rayo_message *msg) {
  (void)params;
  const rayo_shape *a = &in[0].shape, *b = &in[1].shape;
  bool ok = in[0].dtype == RAYO_DTYPE_F32 && in[1].dtype == RAYO_DTYPE_F32 && a->rank == b->rank &&
            a->rank >= 1;
  for (int32_t i = 0; ok && i + 1 < a->rank; i++) ok = a->dims[i] == b->dims[i];
  if (!ok) return rayo_fail(msg, RAYO_ERR_SHAPE, "concat: float32 tensors alike but innermost");
  out->dtype = RAYO_DTYPE_F32;
  out->shape = *a;
  out->shape.dims[a->rank - 1] += b->dims[b->rank - 1];
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  (void)params;
  (void)exec;
  const size_t rows = (size_t)rayo_shape_outer(&out->shape);
  const size_t na = (size_t)rayo_shape_inner(&in[0].shape);
  const size_t nb = (size_t)rayo_shape_inner(&in[1].shape);
  const float *a = in[0].data, *b = in[1].data;
  float *y = out->data;
  for (size_t r = 0; r < rows; r++) {
    memcpy(y + r * (na + nb), a + r * na, na * sizeof(float));
    memcpy(y + r * (na + nb) + na, b + r * nb, nb * sizeof(float));
  }
  return RAYO_OK;
}

const rayo_op rayo_op_concat = {"concat", 2, 2, prepare, infer, run};
