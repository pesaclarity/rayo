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

#define RAYO_CONV_MAX_DILATION 64

typedef struct conv_params {
  rayo_linear lin;
  int64_t size;
  int64_t dilation;
  int64_t channels;
} conv_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *kernel = spec->inputs[1].weight;
  const rayo_weight *bias = spec->input_count > 2 ? spec->inputs[2].weight : NULL;
  if (!kernel || (spec->input_count > 2 && !bias) || spec->inputs[0].weight ||
      (kernel->shape.rank != 4 && kernel->shape.rank != 3) ||
      (kernel->shape.rank == 4 && kernel->shape.dims[1] != 1)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: conv1d takes an input, an [out, 1, size, in] kernel and a bias",
                     spec->output);
  }
  conv_params *p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  const int32_t r = kernel->shape.rank;
  int64_t size;
  rayo_status s = rayo_attr_int(spec, "size", 0, 1, 64, &size, msg);
  if (s == RAYO_OK)
    s = rayo_attr_int(spec, "dilation", 1, 1, RAYO_CONV_MAX_DILATION, &p->dilation, msg);
  if (s == RAYO_OK) s = rayo_attr_activation(spec, &p->lin.act, msg);
  if (s != RAYO_OK) return s;
  if (size != kernel->shape.dims[r - 2] || (size - 1) * p->dilation % 2 != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: size %lld does not fit the kernel or centre",
                     spec->output, (long long)size);
  }
  const char *quant;
  s = rayo_attr_string(spec, "input_quantization", "tensor_phases", &quant, msg);
  if (s != RAYO_OK) return s;
  if (kernel->dtype == RAYO_DTYPE_I8 && strcmp(quant, "tensor_phases") != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: input_quantization %s", spec->output,
                     quant);
  }
  p->size = size;
  p->channels = kernel->shape.dims[r - 1];
  s = rayo_linear_pack(spec, kernel, kernel->shape.dims[0], size * p->channels, bias, arena,
                       &p->lin, msg);
  *params = p;
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)symbols;
  const conv_params *p = params;
  const rayo_tensor *x = &in[0];
  if (x->dtype != RAYO_DTYPE_F32 || x->shape.rank != 2 || x->shape.dims[1] != p->channels) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "conv1d: input must be float32 [T, %lld]",
                     (long long)p->channels);
  }
  const size_t t = (size_t)x->shape.dims[0], c = (size_t)p->channels;
  const size_t kp = (size_t)p->lin.k_padded, n = (size_t)p->lin.out;
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 2;
  out->shape.dims[0] = x->shape.dims[0];
  out->shape.dims[1] = p->lin.out;
  *scratch = 0;
  if (p->lin.dtype == RAYO_DTYPE_I8) {
    rayo_scratch_add(scratch, t * c);                   /* q */
    rayo_scratch_add(scratch, t * kp);                  /* cols */
    rayo_scratch_add(scratch, t * sizeof(float));       /* scale */
    rayo_scratch_add(scratch, t * sizeof(int32_t));     /* zero */
    rayo_scratch_add(scratch, t * n * sizeof(int32_t)); /* acc */
  } else {
    rayo_scratch_add(scratch, t * kp * sizeof(float)); /* cols */
  }
  return RAYO_OK;
}

static rayo_status run_i8(const conv_params *p, const float *x, size_t t, float *y,
                          const rayo_exec *exec) {
  unsigned char *at = exec->scratch;
  const size_t c = (size_t)p->channels, size = (size_t)p->size, d = (size_t)p->dilation;
  const size_t kp = (size_t)p->lin.k_padded, n = (size_t)p->lin.out;
  const ptrdiff_t half = (ptrdiff_t)((size - 1) / 2 * d);
  int8_t *q = rayo_scratch_take(&at, t * c);
  int8_t *cols = rayo_scratch_take(&at, t * kp);
  float *scale = rayo_scratch_take(&at, t * sizeof(float));
  int32_t *zero = rayo_scratch_take(&at, t * sizeof(int32_t));
  int32_t *acc = rayo_scratch_take(&at, t * n * sizeof(int32_t));
  for (size_t ph = 0; ph < d && ph < t; ph++) {
    float s;
    int32_t z;
    rayo_quantize_rows(x, c, ph, d, t, q, c, &s, &z);
    for (size_t r = ph; r < t; r += d) {
      scale[r] = s;
      zero[r] = z;
    }
  }
  if (exec->kernels->a_offset) rayo_offset_rows(q, c, t, c, zero, exec->kernels->a_offset);
  for (size_t i = 0; i < t; i++) {
    int8_t *row = cols + i * kp;
    for (size_t j = 0; j < size; j++) {
      const ptrdiff_t src = (ptrdiff_t)i + (ptrdiff_t)(j * d) - half;
      if (src < 0 || src >= (ptrdiff_t)t) {
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
  unsigned char *at = exec->scratch;
  const size_t c = (size_t)p->channels, size = (size_t)p->size, d = (size_t)p->dilation;
  const size_t k = (size_t)p->lin.k, n = (size_t)p->lin.out;
  const ptrdiff_t half = (ptrdiff_t)((size - 1) / 2 * d);
  float *cols = rayo_scratch_take(&at, t * k * sizeof(float));
  for (size_t i = 0; i < t; i++) {
    for (size_t j = 0; j < size; j++) {
      const ptrdiff_t src = (ptrdiff_t)i + (ptrdiff_t)(j * d) - half;
      float *to = cols + i * k + j * c;
      if (src < 0 || src >= (ptrdiff_t)t) {
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

const rayo_op rayo_op_conv1d = {"conv1d", 2, 3, prepare, infer, run, false};
