/* int8 products with the ARMv8.2 dot product (SDOT: four int8 by int8
 * products added into each int32 lane). Compiled for it with a target
 * attribute and chosen at run time, so the library still runs on ARMv8.0
 * phones. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_NEON) && defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))

#include <arm_neon.h>

#if defined(__clang__)
#define RAYO_DOTPROD __attribute__((target("dotprod")))
#else
#define RAYO_DOTPROD __attribute__((target("+dotprod")))
#endif

RAYO_DOTPROD void rayo_gemm_i8_neon_dotprod(const int8_t *restrict a, size_t lda, size_t m,
                                            const int8_t *restrict w, size_t ldw, size_t n,
                                            size_t k, int32_t *restrict c, size_t ldc) {
  size_t i = 0;
  for (; i + 2 <= m; i += 2) {
    const int8_t *a0 = a + i * lda, *a1 = a0 + lda;
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const int8_t *w0 = w + j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
      int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
      int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
      for (size_t p = 0; p < k; p += 16) {
        const int8x16_t x0 = vld1q_s8(a0 + p), x1 = vld1q_s8(a1 + p);
        const int8x16_t y0 = vld1q_s8(w0 + p), y1 = vld1q_s8(w1 + p);
        const int8x16_t y2 = vld1q_s8(w2 + p), y3 = vld1q_s8(w3 + p);
        c00 = vdotq_s32(c00, x0, y0);
        c01 = vdotq_s32(c01, x0, y1);
        c02 = vdotq_s32(c02, x0, y2);
        c03 = vdotq_s32(c03, x0, y3);
        c10 = vdotq_s32(c10, x1, y0);
        c11 = vdotq_s32(c11, x1, y1);
        c12 = vdotq_s32(c12, x1, y2);
        c13 = vdotq_s32(c13, x1, y3);
      }
      int32_t *r0 = c + i * ldc + j, *r1 = r0 + ldc;
      r0[0] = vaddvq_s32(c00);
      r0[1] = vaddvq_s32(c01);
      r0[2] = vaddvq_s32(c02);
      r0[3] = vaddvq_s32(c03);
      r1[0] = vaddvq_s32(c10);
      r1[1] = vaddvq_s32(c11);
      r1[2] = vaddvq_s32(c12);
      r1[3] = vaddvq_s32(c13);
    }
    if (j < n) rayo_gemm_i8_generic(a0, lda, 2, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
  if (i < m) rayo_gemm_i8_generic(a + i * lda, lda, m - i, w, ldw, n, k, c + i * ldc, ldc);
}

#endif
