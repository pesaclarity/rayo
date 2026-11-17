/* conv1d: a same-padded 1-D convolution over positions. x is [T, C]; the
 * kernel is TFLite's [out, 1, size, C] (or [out, size, C]); y is [T, out].
 * Tap j of position t reads position t + (j - (size - 1) / 2) * dilation,
 * and zeros outside the sequence.
 *
 * An int8 kernel runs as TFLite does: the input quantised to int8 on the
 * fly with one scale for the whole of it, except that a dilated
 * convolution (SPACE_TO_BATCH_ND in TFLite) quantises each of its
 * `dilation` phases (positions t with t % dilation == p) on its own; then
 * im2col and one int8 product. */
#include <string.h>

#include "core/message.h"
#include "kernels/kernels.h"
#include "ops/linear.h"
#include "ops/ops.h"
#include "ops/scratch.h"

typedef struct conv_params {
  rayo_linear lin;
  size_t size;
  size_t dilation;
  size_t channels;
} conv_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *kernel = spec->inputs[1].weight;
  const rayo_weight *bias = spec->input_count > 2 ? spec->inputs[2].weight : NULL;
  conv_params *p;
  rayo_i32 r, size, dilation;
  const char *quant;
  rayo_status s;
  if (!kernel || (spec->input_count > 2 && !bias) || spec->inputs[0].weight ||
      (kernel->shape.rank != 4 && kernel->shape.rank != 3) ||
      (kernel->shape.rank == 4 && kernel->shape.dims[1] != 1)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: conv1d takes an input, an [out, 1, size, in] kernel and a bias",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  r = kernel->shape.rank;
  s = rayo_attr_int(spec, "size", 0, 1, (rayo_i32)RAYO_MAX_DIM, &size, msg);
  if (s == RAYO_OK) {
    s = rayo_attr_int(spec, "dilation", 1, 1, (rayo_i32)RAYO_MAX_DIM, &dilation, msg);
  }
  if (s == RAYO_OK) s = rayo_attr_activation(spec, &p->lin.act, msg);
  if (s != RAYO_OK) return s;
  /* The reach, (size - 1) * dilation, stays below 2^30, so positions and
   * offsets fit a 32-bit long. */
  if (size > 1 && dilation > (RAYO_I32_MAX / 2) / (size - 1)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: size %ld at dilation %ld reaches too far",
                     spec->output, (long)size, (long)dilation);
  }
  if (size != kernel->shape.dims[r - 2] || (size - 1) * dilation % 2 != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: size %ld does not fit the kernel or centre",
                     spec->output, (long)size);
  }
  s = rayo_attr_string(spec, "input_quantization", "tensor_phases", &quant, msg);
  if (s != RAYO_OK) return s;
  if (kernel->dtype == RAYO_DTYPE_I8 && strcmp(quant, "tensor_phases") != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: input_quantization %s", spec->output,
                     quant);
  }
  p->size = (size_t)size;
  p->dilation = (size_t)dilation;
  p->channels = (size_t)kernel->shape.dims[r - 1];
  s = rayo_linear_pack(spec, kernel, (size_t)kernel->shape.dims[0], p->size * p->channels, bias,
                       arena, &p->lin, msg);
  *params = p;
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const conv_params *p = params;
  const rayo_tensor *x = &in[0];
  size_t t, c, kp, n;
  (void)symbols;
  if (x->dtype != RAYO_DTYPE_F32 || x->shape.rank != 2 || (size_t)x->shape.dims[1] != p->channels) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "conv1d: input must be float32 [T, %lu]",
                     (unsigned long)p->channels);
  }
  t = (size_t)x->shape.dims[0];
  c = p->channels;
  kp = p->lin.k_padded;
  n = p->lin.out;
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 2;
  out->shape.dims[0] = x->shape.dims[0];
  out->shape.dims[1] = (rayo_i32)p->lin.out;
  *scratch = 0;
  if (p->lin.dtype == RAYO_DTYPE_I8) {
    rayo_scratch_add(scratch, t * c);                    /* q */
    rayo_scratch_add(scratch, t * kp);                   /* cols */
    rayo_scratch_add(scratch, t * sizeof(float));        /* scale */
    rayo_scratch_add(scratch, t * sizeof(rayo_i32));     /* zero */
    rayo_scratch_add(scratch, t * n * sizeof(rayo_i32)); /* acc */
  } else {
    rayo_scratch_add(scratch, t * kp * sizeof(float)); /* cols */
  }
  return RAYO_OK;
}

/* The source position of tap j of position i, or -1 outside the sequence. */
static long source(size_t i, size_t j, size_t d, size_t half, size_t t) {
  const long src = (long)i + (long)(j * d) - (long)half;
  return src < 0 || src >= (long)t ? -1 : src;
}

static rayo_status run_i8(const conv_params *p, const float *x, size_t t, float *y,
                          const rayo_exec *exec) {
  const size_t c = p->channels, size = p->size, d = p->dilation;
  const size_t kp = p->lin.k_padded, n = p->lin.out;
  const size_t half = (size - 1) / 2 * d;
  unsigned char *at = exec->scratch;
  rayo_i8 *q = rayo_scratch_take(&at, t * c);
  rayo_i8 *cols = rayo_scratch_take(&at, t * kp);
  float *scale = rayo_scratch_take(&at, t * sizeof(float));
  rayo_i32 *zero = rayo_scratch_take(&at, t * sizeof(rayo_i32));
  rayo_i32 *acc = rayo_scratch_take(&at, t * n * sizeof(rayo_i32));
  size_t ph, r, i, j;
  for (ph = 0; ph < d && ph < t; ph++) {
    float s;
    rayo_i32 z;
    rayo_quantize_rows(x, c, ph, d, t, q, c, &s, &z);
    for (r = ph; r < t; r += d) {
      scale[r] = s;
      zero[r] = z;
    }
  }
  if (exec->kernels->a_offset) rayo_offset_rows(q, c, t, c, zero, exec->kernels->a_offset);
  for (i = 0; i < t; i++) {
    rayo_i8 *row = cols + i * kp;
    for (j = 0; j < size; j++) {
      const long src = source(i, j, d, half, t);
      if (src < 0) {
        /* Zero in floats is the zero point in int8. */
        memset(row + j * c, (int)zero[i], c);
      } else {
        memcpy(row + j * c, q + (size_t)src * c, c);
      }
    }
  }
  exec->kernels->gemm_i8(cols, kp, t, p->lin.w8, kp, n, kp, acc, n);
  rayo_linear_finish_i8(&p->lin, acc, t, scale, zero, y);
  return RAYO_OK;
}

static rayo_status run_f32(const conv_params *p, const float *x, size_t t, float *y,
                           const rayo_exec *exec) {
  const size_t c = p->channels, size = p->size, d = p->dilation;
  const size_t k = p->lin.k, n = p->lin.out;
  const size_t half = (size - 1) / 2 * d;
  unsigned char *at = exec->scratch;
  float *cols = rayo_scratch_take(&at, t * k * sizeof(float));
  size_t i, j;
  for (i = 0; i < t; i++) {
    for (j = 0; j < size; j++) {
      const long src = source(i, j, d, half, t);
      float *to = cols + i * k + j * c;
      if (src < 0) {
        memset(to, 0, c * sizeof(float));
      } else {
        memcpy(to, x + (size_t)src * c, c * sizeof(float));
      }
    }
  }
  exec->kernels->gemm_f32(cols, k, t, p->lin.wf, k, n, k, y, n);
  rayo_linear_finish_f32(&p->lin, y, t);
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const conv_params *p = params;
  const size_t t = (size_t)in[0].shape.dims[0];
  if (p->lin.dtype == RAYO_DTYPE_I8) return run_i8(p, in[0].data, t, out->data, exec);
  return run_f32(p, in[0].data, t, out->data, exec);
}

const rayo_op rayo_op_conv1d = {"conv1d", 2, 3, prepare, infer, run, RAYO_FALSE};
