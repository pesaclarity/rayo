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
  rayo_linear *lin;
  const char *quant;
  rayo_status s;
  if (!kernel || kernel->shape.rank != 2 || (spec->input_count > 2 && !bias) ||
      spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: dense takes an input, a [out, in] kernel and a bias", spec->output);
  }
  lin = rayo_arena_alloc(arena, sizeof *lin);
  if (!lin) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s = rayo_attr_activation(spec, &lin->act, msg);
  if (s != RAYO_OK) return s;
  s = rayo_attr_string(spec, "input_quantization", "rows", &quant, msg);
  if (s != RAYO_OK) return s;
  if (kernel->dtype == RAYO_DTYPE_I8 && strcmp(quant, "rows") != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: input_quantization %s", spec->output,
                     quant);
  }
  s = rayo_linear_pack(spec, kernel, (size_t)kernel->shape.dims[0], (size_t)kernel->shape.dims[1],
                       bias, arena, lin, msg);
  *params = lin;
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const rayo_linear *lin = params;
  const rayo_tensor *x = &in[0];
  size_t rows;
  (void)symbols;
  if (x->dtype != RAYO_DTYPE_F32 || x->shape.rank < 1 || rayo_shape_inner(&x->shape) != lin->k) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "dense: input of %lu, kernel of %lu",
                     (unsigned long)rayo_shape_inner(&x->shape), (unsigned long)lin->k);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = x->shape;
  out->shape.dims[out->shape.rank - 1] = (rayo_i32)lin->out;
  rows = rayo_shape_outer(&x->shape);
  *scratch = 0;
  if (lin->dtype == RAYO_DTYPE_I8) {
    rayo_scratch_add(scratch, rows * lin->k_padded);               /* q */
    rayo_scratch_add(scratch, rows * sizeof(float));               /* scale */
    rayo_scratch_add(scratch, rows * sizeof(rayo_i32));            /* zero */
    rayo_scratch_add(scratch, rows * lin->out * sizeof(rayo_i32)); /* acc */
  }
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const rayo_linear *lin = params;
  const float *x = in[0].data;
  float *y = out->data;
  const size_t rows = rayo_shape_outer(&in[0].shape);
  const size_t k = lin->k, n = lin->out, kp = lin->k_padded;
  const rayo_kernels *kn = exec->kernels;
  unsigned char *at = exec->scratch;
  rayo_i8 *q;
  float *scale;
  rayo_i32 *zero, *acc;
  size_t r;
  if (lin->dtype == RAYO_DTYPE_F32) {
    kn->gemm_f32(x, k, rows, lin->wf, k, n, k, y, n);
    rayo_linear_finish_f32(lin, y, rows);
    return RAYO_OK;
  }
  q = rayo_scratch_take(&at, rows * kp);
  scale = rayo_scratch_take(&at, rows * sizeof(float));
  zero = rayo_scratch_take(&at, rows * sizeof(rayo_i32));
  acc = rayo_scratch_take(&at, rows * n * sizeof(rayo_i32));
  for (r = 0; r < rows; r++) {
    rayo_quantize_rows(x, k, r, rows, r + 1, q, kp, &scale[r], &zero[r]);
  }
  if (kn->a_offset) rayo_offset_rows(q, kp, rows, k, zero, kn->a_offset);
  kn->gemm_i8(q, kp, rows, lin->w8, kp, n, kp, acc, n);
  rayo_linear_finish_i8(lin, acc, rows, scale, zero, y);
  return RAYO_OK;
}

const rayo_op rayo_op_dense = {"dense", 2, 3, prepare, infer, run, RAYO_FALSE};
