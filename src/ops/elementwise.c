/* add: a + b, same shapes, float32. */
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

static rayo_status no_params(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                             rayo_message *msg) {
  int i;
  (void)arena;
  for (i = 0; i < spec->input_count; i++) {
    if (spec->inputs[i].weight) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: takes activations, not weights",
                       spec->output);
    }
  }
  *params = NULL;
  return RAYO_OK;
}

static rayo_status add_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                             rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32 ||
      !rayo_shape_equal(&in[0].shape, &in[1].shape)) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "add: two float32 tensors of one shape");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status add_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                           const rayo_exec *exec) {
  const float *RAYO_RESTRICT a = in[0].data;
  const float *RAYO_RESTRICT b = in[1].data;
  float *RAYO_RESTRICT y = out->data;
  const size_t n = rayo_shape_count(&out->shape);
  size_t i;
  (void)params;
  (void)exec;
  for (i = 0; i < n; i++) y[i] = a[i] + b[i];
  return RAYO_OK;
}

const rayo_op rayo_op_add = {"add", 2, 2, no_params, add_infer, add_run, RAYO_FALSE};
