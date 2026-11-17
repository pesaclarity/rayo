/* dense: y = act(x · kernelᵀ + bias) over the innermost dimension. An int8
 * kernel runs as TFLite's hybrid kernel: each row of x quantised to int8 on
 * the fly, the product in int32. */
#include <string.h>

#include "core/message.h"
#include "kernels/kernels.h"
#include "ops/linear.h"
#include "ops/ops.h"
#include "ops/scratch.h"

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *kernel = spec->inputs[1].weight;
  const rayo_weight *bias = spec->input_count > 2 ? spec->inputs[2].weight : NULL;
  if (!kernel || kernel->shape.rank != 2 || (spec->input_count > 2 && !bias) ||
      spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: dense takes an input, a [out, in] kernel and a bias", spec->output);
  }
  rayo_linear *lin = rayo_arena_alloc(arena, sizeof *lin);
  if (!lin) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  rayo_status s = rayo_attr_activation(spec, &lin->act, msg);
  if (s != RAYO_OK) return s;
  const char *quant;
  s = rayo_attr_string(spec, "input_quantization", "rows", &quant, msg);
  if (s != RAYO_OK) return s;
  if (kernel->dtype == RAYO_DTYPE_I8 && strcmp(quant, "rows") != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: input_quantization %s", spec->output,
                     quant);
  }
  s = rayo_linear_pack(spec, kernel, kernel->shape.dims[0], kernel->shape.dims[1], bias, arena, lin,
                       msg);
  *params = lin;
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)symbols;
  const rayo_linear *lin = params;
  const rayo_tensor *x = &in[0];
  if (x->dtype != RAYO_DTYPE_F32 || x->shape.rank < 1 || rayo_shape_inner(&x->shape) != lin->k) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "dense: input of %lld, kernel of %lld",
                     (long long)rayo_shape_inner(&x->shape), (long long)lin->k);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = x->shape;
  out->shape.dims[out->shape.rank - 1] = lin->out;
  const size_t rows = (size_t)rayo_shape_outer(&x->shape);
  *scratch = 0;
  if (lin->dtype == RAYO_DTYPE_I8) {
    rayo_scratch_add(scratch, rows * (size_t)lin->k_padded);              /* q */
    rayo_scratch_add(scratch, rows * sizeof(float));                      /* scale */
    rayo_scratch_add(scratch, rows * sizeof(int32_t));                    /* zero */
    rayo_scratch_add(scratch, rows * (size_t)lin->out * sizeof(int32_t)); /* acc */
  }
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const rayo_linear *lin = params;
  const float *x = in[0].data;
  float *y = out->data;
  const size_t rows = (size_t)rayo_shape_outer(&in[0].shape);
  const size_t k = (size_t)lin->k, n = (size_t)lin->out;
  const rayo_kernels *kn = exec->kernels;
  if (lin->dtype == RAYO_DTYPE_F32) {
    kn->gemm_f32(x, k, rows, lin->wf, k, n, k, y, n);
    rayo_linear_finish_f32(lin, y, rows);
    return RAYO_OK;
  }
  const size_t kp = (size_t)lin->k_padded;
  unsigned char *at = exec->scratch;
  int8_t *q = rayo_scratch_take(&at, rows * kp);
  float *scale = rayo_scratch_take(&at, rows * sizeof(float));
  int32_t *zero = rayo_scratch_take(&at, rows * sizeof(int32_t));
  int32_t *acc = rayo_scratch_take(&at, rows * n * sizeof(int32_t));
  for (size_t r = 0; r < rows; r++) {
    rayo_quantize_rows(x, k, r, rows, r + 1, q, kp, &scale[r], &zero[r]);
  }
  if (kn->a_offset) rayo_offset_rows(q, kp, rows, k, zero, kn->a_offset);
  kn->gemm_i8(q, kp, rows, lin->w8, kp, n, kp, acc, n);
  rayo_linear_finish_i8(lin, acc, rows, scale, zero, y);
  return RAYO_OK;
}

const rayo_op rayo_op_dense = {"dense", 2, 3, prepare, infer, run, false};
