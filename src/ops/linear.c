#include "ops/linear.h"

#include <string.h>

#include "core/mathx.h"
#include "core/message.h"
#include "kernels/kernels.h"

static rayo_status pack_i8(const rayo_op_spec *spec, const rayo_weight *kernel, rayo_arena *arena,
                           rayo_linear *lin, rayo_message *msg) {
  const size_t out = lin->out, k = lin->k;
  const size_t kp = (k + RAYO_K_ALIGN - 1) / RAYO_K_ALIGN * RAYO_K_ALIGN;
  const rayo_i8 *src = kernel->data;
  const rayo_i8 *w8 = src;
  rayo_i32 *rowsum;
  float *wscale;
  size_t o, p;
  if (kernel->zero_point != 0 || (kernel->axis != -1 && kernel->axis != 0)) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "node %s: an int8 kernel needs zero point 0 and scales by output",
                     spec->output);
  }
  rowsum = rayo_arena_array(arena, out, sizeof *rowsum);
  wscale = rayo_arena_array(arena, out, sizeof *wscale);
  if (!rowsum || !wscale) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  /* Rows already a multiple of RAYO_K_ALIGN long are used where they are in
   * the model's copy of the file; others are copied with zeros after them. */
  if (kp != k) {
    rayo_i8 *packed = rayo_arena_alloc_aligned(arena, out * kp);
    if (!packed) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    for (o = 0; o < out; o++) memcpy(packed + o * kp, src + o * k, k);
    w8 = packed;
  }
  for (o = 0; o < out; o++) {
    rayo_i32 sum = 0;
    for (p = 0; p < k; p++) sum += src[o * k + p];
    rowsum[o] = sum;
    wscale[o] = kernel->scales[kernel->axis == 0 ? o : 0];
  }
  lin->k_padded = kp;
  lin->w8 = w8;
  lin->rowsum = rowsum;
  lin->wscale = wscale;
  return RAYO_OK;
}

rayo_status rayo_linear_pack(const rayo_op_spec *spec, const rayo_weight *kernel, size_t out,
                             size_t k, const rayo_weight *bias, rayo_arena *arena, rayo_linear *lin,
                             rayo_message *msg) {
  float *b;
  if (out < 1 || k < 1 || kernel->count != out * k) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: the kernel's size", spec->output);
  }
  /* The int32 sums must not overflow: the products reach k * 128 * 128, and
   * the zero point (up to 255 with uint8 kernels) times a row's sum
   * k * 255 * 128. */
  if (k > (size_t)(RAYO_I32_MAX / (256 * 128))) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: inner size %lu is too large",
                     spec->output, (unsigned long)k);
  }
  lin->dtype = kernel->dtype;
  lin->out = out;
  lin->k = k;
  lin->k_padded = k;
  b = rayo_arena_array(arena, out, sizeof *b);
  if (!b) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (bias) {
    if (bias->dtype != RAYO_DTYPE_F32 || bias->count != out) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: the bias must be float32 [%lu]", spec->output,
                       (unsigned long)out);
    }
    memcpy(b, bias->data, out * sizeof *b);
  }
  lin->bias = b;
  if (kernel->dtype == RAYO_DTYPE_I8) return pack_i8(spec, kernel, arena, lin, msg);
  if (kernel->dtype == RAYO_DTYPE_I4 || kernel->dtype == RAYO_DTYPE_I3) {
    if (k % RAYO_Q4_GROUP != 0 || kernel->group_size != RAYO_Q4_GROUP) {
      return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                       "node %s: a 4-bit or 3-bit kernel needs groups of %d", spec->output,
                       RAYO_Q4_GROUP);
    }
    lin->w4 = kernel->data;
    lin->gscale = kernel->group_scales;
    lin->gscale_half = kernel->scales_half;
    lin->group_bytes = kernel->dtype == RAYO_DTYPE_I4 ? RAYO_Q4_GROUP_BYTES : RAYO_Q3_GROUP_BYTES;
    return RAYO_OK;
  }
  lin->wf = kernel->data;
  return RAYO_OK;
}

static float activate(rayo_activation act, float v) {
  switch (act) {
    case RAYO_ACT_RELU:
      return v < 0.0f ? 0.0f : v;
    case RAYO_ACT_GELU_ERF:
      return 0.5f * v * (1.0f + rayo_erff(v * 0.70710678118654752f));
    case RAYO_ACT_SILU:
      return rayo_silu(v);
    case RAYO_ACT_NONE:
      break;
  }
  return v;
}

void rayo_linear_finish_i8(const rayo_linear *lin, const rayo_i32 *RAYO_RESTRICT acc, size_t rows,
                           const float *RAYO_RESTRICT scale, const rayo_i32 *RAYO_RESTRICT zero,
                           float *RAYO_RESTRICT y) {
  const size_t out = lin->out;
  const rayo_i32 *RAYO_RESTRICT rowsum = lin->rowsum;
  const float *RAYO_RESTRICT wscale = lin->wscale;
  const float *RAYO_RESTRICT bias = lin->bias;
  const rayo_activation act = lin->act;
  size_t r, o;
  for (r = 0; r < rows; r++) {
    const rayo_i32 *a = acc + r * out;
    float *yr = y + r * out;
    const float s = scale[r];
    const rayo_i32 z = zero[r];
    if (act == RAYO_ACT_RELU) {
      for (o = 0; o < out; o++) {
        const float v = bias[o] + (float)(a[o] - z * rowsum[o]) * (s * wscale[o]);
        yr[o] = v < 0.0f ? 0.0f : v;
      }
    } else {
      for (o = 0; o < out; o++) {
        yr[o] = activate(act, bias[o] + (float)(a[o] - z * rowsum[o]) * (s * wscale[o]));
      }
    }
  }
}

void rayo_linear_finish_f32(const rayo_linear *lin, float *RAYO_RESTRICT y, size_t rows) {
  const size_t out = lin->out;
  size_t r, o;
  for (r = 0; r < rows; r++) {
    float *yr = y + r * out;
    for (o = 0; o < out; o++) yr[o] = activate(lin->act, yr[o] + lin->bias[o]);
  }
}
