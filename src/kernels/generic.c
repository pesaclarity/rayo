#include "kernels/kernels.h"

void rayo_gemm_i8_generic(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                          const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                          rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p;
  for (i = 0; i < m; i++) {
    const rayo_i8 *ai = a + i * lda;
    for (j = 0; j < n; j++) {
      const rayo_i8 *wj = w + j * ldw;
      rayo_i32 acc = 0;
      for (p = 0; p < k; p++) acc += (rayo_i32)ai[p] * (rayo_i32)wj[p];
      c[i * ldc + j] = acc;
    }
  }
}

/* Eight partial sums per output, added in a fixed order at the end: the
 * compiler vectorises it, and every build rounds the same way. */
void rayo_gemm_f32_generic(const float *RAYO_RESTRICT a, size_t lda, size_t m,
                           const float *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                           float *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p, l;
  for (i = 0; i < m; i++) {
    const float *ai = a + i * lda;
    for (j = 0; j < n; j++) {
      const float *wj = w + j * ldw;
      float lanes[8];
      float acc;
      for (l = 0; l < 8; l++) lanes[l] = 0.0f;
      for (p = 0; p + 8 <= k; p += 8) {
        for (l = 0; l < 8; l++) lanes[l] += ai[p + l] * wj[p + l];
      }
      acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
            ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
      for (; p < k; p++) acc += ai[p] * wj[p];
      c[i * ldc + j] = acc;
    }
  }
}
