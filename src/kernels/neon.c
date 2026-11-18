/* int8 products with NEON (ARMv8.0 baseline on arm64, NEON on armv7):
 * widening multiplies into int16 (vmull_s8, at most 128 * 128), then
 * pairwise adds into int32 (vpadalq_s16). One row of `a` by four of `w` per
 * step. Intrinsics are a compiler extension, confined to this file. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_NEON)

#include <arm_neon.h>

#include "kernels/half.h"
#include "kernels/neon_unpack.h"

static rayo_i32 hsum(int32x4_t v) {
#if defined(__aarch64__)
  return (rayo_i32)vaddvq_s32(v);
#else
  int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
  return (rayo_i32)vget_lane_s32(vpadd_s32(s, s), 0);
#endif
}

static int32x4_t mac16(int32x4_t acc, int8x16_t x, int8x16_t y) {
  acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
  return vpadalq_s16(acc, vmull_s8(vget_high_s8(x), vget_high_s8(y)));
}

static void block1x4(const rayo_i8 *a0, const rayo_i8 *w, size_t ldw, size_t k, rayo_i32 *r) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  size_t p;
  for (p = 0; p < k; p += 16) {
    int8x16_t x = vld1q_s8(a0 + p);
    c0 = mac16(c0, x, vld1q_s8(w0 + p));
    c1 = mac16(c1, x, vld1q_s8(w1 + p));
    c2 = mac16(c2, x, vld1q_s8(w2 + p));
    c3 = mac16(c3, x, vld1q_s8(w3 + p));
  }
  r[0] = hsum(c0);
  r[1] = hsum(c1);
  r[2] = hsum(c2);
  r[3] = hsum(c3);
}

void rayo_gemm_i8_neon(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                       const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                       rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j;
  for (i = 0; i < m; i++) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) block1x4(a0, w + j * ldw, ldw, k, c + i * ldc + j);
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

/* Lane l: the sum of x[4l .. 4l + 3] * y[4l .. 4l + 3], exact. */
static int32x4_t dot4(int8x16_t x, int8x16_t y) {
  const int32x4_t lo = vpaddlq_s16(vmull_s8(vget_low_s8(x), vget_low_s8(y)));
  const int32x4_t hi = vpaddlq_s16(vmull_s8(vget_high_s8(x), vget_high_s8(y)));
#if defined(__aarch64__)
  return vpaddq_s32(lo, hi);
#else
  return vcombine_s32(vpadd_s32(vget_low_s32(lo), vget_high_s32(lo)),
                      vpadd_s32(vget_low_s32(hi), vget_high_s32(hi)));
#endif
}

/* Group-wise weights (kernels.h): a group unpacked to its low and high
 * halves of 16 int8 (4 bits: the low and the high four bits, minus 8;
 * 3 bits: the two-bit planes with the third bits from their bytes, minus
 * 4), products in lanes of four, scaled in float and added into eight
 * lanes, as the generic kernel does (the unpacking: neon_unpack.h). */
static float dot_q(const rayo_i8 *a, const float *as, const rayo_u8 *w, const void *ws,
                   rayo_bool half, size_t first, size_t groups, size_t bytes) {
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    int8x16_t wl, wh;
    float d;
    if (bytes == RAYO_Q4_GROUP_BYTES) {
      neon_unpack_q4(w + g * bytes, &wl, &wh);
    } else {
      neon_unpack_q3(w + g * bytes, &wl, &wh);
    }
    d = as[g] * rayo_group_scale(ws, half, first + g);
    acc0 = vaddq_f32(acc0, vmulq_n_f32(vcvtq_f32_s32(dot4(vld1q_s8(a + g * 32), wl)), d));
    acc1 = vaddq_f32(acc1, vmulq_n_f32(vcvtq_f32_s32(dot4(vld1q_s8(a + g * 32 + 16), wh)), d));
  }
  vst1q_f32(lanes, acc0);
  vst1q_f32(lanes + 4, acc1);
  return rayo_q4_lanes_sum(lanes);
}

static void gemm_q(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale, size_t m,
                   const rayo_u8 *RAYO_RESTRICT w, const void *wscale, rayo_bool half, size_t n,
                   size_t k, float *RAYO_RESTRICT c, size_t ldc, size_t bytes) {
  const size_t groups = k / RAYO_Q4_GROUP;
  size_t i, j;
  for (j = 0; j < n; j++) {
    const rayo_u8 *wj = w + j * groups * bytes;
    for (i = 0; i < m; i++) {
      c[i * ldc + j] =
          dot_q(a + i * k, ascale + i * groups, wj, wscale, half, j * groups, groups, bytes);
    }
  }
}

void rayo_gemm_q4_neon(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale, size_t m,
                       const rayo_u8 *RAYO_RESTRICT w, const void *wscale, rayo_bool half, size_t n,
                       size_t k, float *RAYO_RESTRICT c, size_t ldc) {
  gemm_q(a, ascale, m, w, wscale, half, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
}

void rayo_gemm_q3_neon(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale, size_t m,
                       const rayo_u8 *RAYO_RESTRICT w, const void *wscale, rayo_bool half, size_t n,
                       size_t k, float *RAYO_RESTRICT c, size_t ldc) {
  gemm_q(a, ascale, m, w, wscale, half, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
}
#else
typedef int rayo_neon_unused;
#endif
