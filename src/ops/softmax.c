/* softmax over the innermost dimension, float32. */
#include <math.h>

#include "core/message.h"
#include "ops/ops.h"

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  (void)arena;
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: softmax takes an activation", spec->output);
  }
  *params = NULL;
  return RAYO_OK;
}

static rayo_status infer(const void *params, const rayo_tensor *in, rayo_tensor *out,
                         size_t *scratch, rayo_message *msg) {
  (void)params;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank < 1) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "softmax: a float32 tensor");
  }
  *out = in[0];
  out->data = NULL;
  out->weight = NULL;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  (void)params;
  (void)exec;
  const size_t rows = (size_t)rayo_shape_outer(&out->shape);
  const size_t n = (size_t)rayo_shape_inner(&out->shape);
  for (size_t r = 0; r < rows; r++) {
    const float *x = (const float *)in[0].data + r * n;
    float *y = (float *)out->data + r * n;
    float top = x[0];
    for (size_t i = 1; i < n; i++) top = x[i] > top ? x[i] : top;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
      y[i] = expf(x[i] - top);
      sum += y[i];
    }
    const float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) y[i] *= inv;
  }
  return RAYO_OK;
}

const rayo_op rayo_op_softmax = {"softmax", 1, 1, prepare, infer, run};
