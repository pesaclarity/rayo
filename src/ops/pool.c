/* masked_max and masked_mean: x [T, C] and a mask [T] (1 for a real
 * position) to [C].
 *
 * masked_max is max over t of (x * m - (1 - m) * fill), as Keras writes it
 * (fill 1e4 by default), so a question with no real token gives -fill.
 * masked_mean is sum over t of x * m, divided by max(sum of m, 1). */
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct pool_params {
  float fill;
} pool_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  if (spec->inputs[0].weight || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: takes activations", spec->output);
  }
  pool_params *p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  return rayo_attr_float(spec, "fill", 1e4f, &p->fill, msg);
}

static rayo_status infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)symbols;
  (void)params;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank != 2 || in[1].dtype != RAYO_DTYPE_F32 ||
      rayo_shape_count(&in[1].shape) != in[0].shape.dims[0]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "masked pooling: float32 [T, C] and a [T] mask");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 1;
  out->shape.dims[0] = in[0].shape.dims[1];
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status max_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                           const rayo_exec *exec) {
  (void)exec;
  const pool_params *p = params;
  const float *x = in[0].data, *m = in[1].data;
  const size_t t = (size_t)in[0].shape.dims[0], c = (size_t)in[0].shape.dims[1];
  float *y = out->data;
  for (size_t j = 0; j < c; j++) y[j] = -3.4e38f;
  for (size_t i = 0; i < t; i++) {
    const float mi = m[i], off = (1.0f - mi) * p->fill;
    for (size_t j = 0; j < c; j++) {
      const float v = x[i * c + j] * mi - off;
      y[j] = v > y[j] ? v : y[j];
    }
  }
  return RAYO_OK;
}

static rayo_status mean_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                            const rayo_exec *exec) {
  (void)params;
  (void)exec;
  const float *x = in[0].data, *m = in[1].data;
  const size_t t = (size_t)in[0].shape.dims[0], c = (size_t)in[0].shape.dims[1];
  float *y = out->data;
  float n = 0.0f;
  for (size_t j = 0; j < c; j++) y[j] = 0.0f;
  for (size_t i = 0; i < t; i++) {
    n += m[i];
    for (size_t j = 0; j < c; j++) y[j] += x[i * c + j] * m[i];
  }
  const float inv = 1.0f / (n > 1.0f ? n : 1.0f);
  for (size_t j = 0; j < c; j++) y[j] *= inv;
  return RAYO_OK;
}

const rayo_op rayo_op_masked_max = {"masked_max", 2, 2, prepare, infer, max_run, false};
const rayo_op rayo_op_masked_mean = {"masked_mean", 2, 2, prepare, infer, mean_run, false};
