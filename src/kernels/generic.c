#include "kernels/kernels.h"

void rayo_gemm_i8_generic(const int8_t *restrict a, size_t lda, size_t m, const int8_t *restrict w,
                          size_t ldw, size_t n, size_t k, int32_t *restrict c, size_t ldc) {
  for (size_t i = 0; i < m; i++) {
    const int8_t *ai = a + i * lda;
    for (size_t j = 0; j < n; j++) {
      const int8_t *wj = w + j * ldw;
      int32_t acc = 0;
      for (size_t p = 0; p < k; p++) acc += (int32_t)ai[p] * (int32_t)wj[p];
      c[i * ldc + j] = acc;
    }
  }
}

/* Eight partial sums per output, added in a fixed order at the end: the
 * compiler vectorises it, and every build rounds the same way. */
void rayo_gemm_f32_generic(const float *restrict a, size_t lda, size_t m, const float *restrict w,
                           size_t ldw, size_t n, size_t k, float *restrict c, size_t ldc) {
  for (size_t i = 0; i < m; i++) {
    const float *ai = a + i * lda;
    for (size_t j = 0; j < n; j++) {
      const float *wj = w + j * ldw;
      float lanes[8] = {0};
      size_t p = 0;
      for (; p + 8 <= k; p += 8) {
        for (size_t l = 0; l < 8; l++) lanes[l] += ai[p + l] * wj[p + l];
      }
      float acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                  ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
      for (; p < k; p++) acc += ai[p] * wj[p];
      c[i * ldc + j] = acc;
    }
  }
}
