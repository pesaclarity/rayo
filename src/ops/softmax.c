/* softmax over the innermost dimension, float32. */
#include "core/mathx.h"
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

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank < 1) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "softmax: a float32 tensor");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static void softmax_row(const float *x, float *y, size_t n) {
  float top = x[0], inv;
  double sum = 0.0;
  size_t i;
  for (i = 1; i < n; i++) top = x[i] > top ? x[i] : top;
  for (i = 0; i < n; i++) {
    y[i] = rayo_expf(x[i] - top);
    sum += y[i];
  }
  inv = (float)(1.0 / sum);
  for (i = 0; i < n; i++) y[i] *= inv;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const size_t rows = rayo_shape_outer(&out->shape);
  const size_t n = rayo_shape_inner(&out->shape);
  size_t r;
  (void)params;
  (void)exec;
  for (r = 0; r < rows; r++) {
    softmax_row((const float *)in[0].data + r * n, (float *)out->data + r * n, n);
  }
  return RAYO_OK;
}

const rayo_op rayo_op_softmax = {"softmax", 1, 1, prepare, infer, run, RAYO_FALSE};
