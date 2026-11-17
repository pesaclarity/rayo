/* int8 products with NEON (ARMv8.0 baseline on arm64, NEON on armv7):
 * widening multiplies into int16 (vmull_s8, at most 128 * 128), then
 * pairwise adds into int32 (vpadalq_s16). One row of `a` by four of `w` per
 * step. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_NEON)

#include <arm_neon.h>

static inline int32_t hsum(int32x4_t v) {
#if defined(__aarch64__)
  return vaddvq_s32(v);
#else
  int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
  return vget_lane_s32(vpadd_s32(s, s), 0);
#endif
}

static inline int32x4_t mac16(int32x4_t acc, int8x16_t x, int8x16_t y) {
  acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
  return vpadalq_s16(acc, vmull_s8(vget_high_s8(x), vget_high_s8(y)));
}

void rayo_gemm_i8_neon(const int8_t *restrict a, size_t lda, size_t m, const int8_t *restrict w,
                       size_t ldw, size_t n, size_t k, int32_t *restrict c, size_t ldc) {
  for (size_t i = 0; i < m; i++) {
    const int8_t *a0 = a + i * lda;
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const int8_t *w0 = w + j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
      int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
      for (size_t p = 0; p < k; p += 16) {
        int8x16_t x = vld1q_s8(a0 + p);
        c0 = mac16(c0, x, vld1q_s8(w0 + p));
        c1 = mac16(c1, x, vld1q_s8(w1 + p));
        c2 = mac16(c2, x, vld1q_s8(w2 + p));
        c3 = mac16(c3, x, vld1q_s8(w3 + p));
      }
      int32_t *r = c + i * ldc + j;
      r[0] = hsum(c0);
      r[1] = hsum(c1);
      r[2] = hsum(c2);
      r[3] = hsum(c3);
    }
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

#endif
