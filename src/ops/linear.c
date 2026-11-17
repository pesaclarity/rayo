#include "ops/linear.h"

#include <math.h>
#include <string.h>

#include "core/message.h"
#include "kernels/kernels.h"

static rayo_status pack_i8(const rayo_op_spec *spec, const rayo_weight *kernel, rayo_arena *arena,
                           rayo_linear *lin, rayo_message *msg) {
  if (kernel->zero_point != 0 || (kernel->axis != -1 && kernel->axis != 0)) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "node %s: an int8 kernel needs zero point 0 and scales by output",
                     spec->output);
  }
  const size_t out = (size_t)lin->out, k = (size_t)lin->k;
  const size_t kp = (k + RAYO_K_ALIGN - 1) / RAYO_K_ALIGN * RAYO_K_ALIGN;
  int32_t *rowsum = rayo_arena_array(arena, out, sizeof *rowsum);
  float *wscale = rayo_arena_array(arena, out, sizeof *wscale);
  if (!rowsum || !wscale) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  const int8_t *src = kernel->data;
  /* Rows already a multiple of RAYO_K_ALIGN long are used where they are in
   * the model's copy of the file; others are copied with zeros after them. */
  const int8_t *w8 = src;
  if (kp != k) {
    int8_t *packed = rayo_arena_alloc_aligned(arena, out * kp);
    if (!packed) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    for (size_t o = 0; o < out; o++) memcpy(packed + o * kp, src + o * k, k);
    w8 = packed;
  }
  for (size_t o = 0; o < out; o++) {
    int32_t sum = 0;
    for (size_t p = 0; p < k; p++) sum += src[o * k + p];
    rowsum[o] = sum;
    wscale[o] = kernel->scales[kernel->axis == 0 ? o : 0];
  }
  lin->k_padded = (int64_t)kp;
  lin->w8 = w8;
  lin->rowsum = rowsum;
  lin->wscale = wscale;
  return RAYO_OK;
}

rayo_status rayo_linear_pack(const rayo_op_spec *spec, const rayo_weight *kernel, int64_t out,
                             int64_t k, const rayo_weight *bias, rayo_arena *arena,
                             rayo_linear *lin, rayo_message *msg) {
  if (kernel->count != out * k || out < 1 || k < 1) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: the kernel's size", spec->output);
  }
  if ((int64_t)k * 127 * 128 > INT32_MAX) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: inner size %lld is too large",
                     spec->output, (long long)k);
  }
  lin->dtype = kernel->dtype;
  lin->out = out;
  lin->k = k;
  lin->k_padded = k;
  float *b = rayo_arena_array(arena, (size_t)out, sizeof *b);
  if (!b) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (bias) {
    if (bias->dtype != RAYO_DTYPE_F32 || bias->count != out) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: the bias must be float32 [%lld]",
                       spec->output, (long long)out);
    }
    memcpy(b, bias->data, (size_t)out * sizeof *b);
  }
  lin->bias = b;
  if (kernel->dtype == RAYO_DTYPE_I8) return pack_i8(spec, kernel, arena, lin, msg);
  lin->wf = kernel->data;
  return RAYO_OK;
}

static inline float activate(rayo_activation act, float v) {
  switch (act) {
    case RAYO_ACT_RELU:
      return v < 0.0f ? 0.0f : v;
    case RAYO_ACT_GELU_ERF:
      return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    case RAYO_ACT_NONE:
      break;
  }
  return v;
}

void rayo_linear_finish_i8(const rayo_linear *lin, const int32_t *restrict acc, size_t rows,
                           const float *restrict scale, const int32_t *restrict zero,
                           float *restrict y) {
  const size_t out = (size_t)lin->out;
  const int32_t *restrict rowsum = lin->rowsum;
  const float *restrict wscale = lin->wscale, *restrict bias = lin->bias;
  const rayo_activation act = lin->act;
  for (size_t r = 0; r < rows; r++) {
    const int32_t *a = acc + r * out;
    float *yr = y + r * out;
    const float s = scale[r];
    const int32_t z = zero[r];
    if (act == RAYO_ACT_RELU) {
      for (size_t o = 0; o < out; o++) {
        const float v = bias[o] + (float)(a[o] - z * rowsum[o]) * (s * wscale[o]);
        yr[o] = v < 0.0f ? 0.0f : v;
      }
    } else {
      for (size_t o = 0; o < out; o++) {
        yr[o] = activate(act, bias[o] + (float)(a[o] - z * rowsum[o]) * (s * wscale[o]));
      }
    }
  }
}

void rayo_linear_finish_f32(const rayo_linear *lin, float *restrict y, size_t rows) {
  const size_t out = (size_t)lin->out;
  for (size_t r = 0; r < rows; r++) {
    float *yr = y + r * out;
    for (size_t o = 0; o < out; o++) yr[o] = activate(lin->act, yr[o] + lin->bias[o]);
  }
}
