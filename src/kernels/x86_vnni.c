/* int8 products with AVX-VNNI (vpdpbusd: four uint8 by int8 products added
 * into each int32 lane, exact, no saturation). Compiled for AVX-VNNI with a
 * target attribute and chosen at run time by rayo_kernels_best, so the
 * library still runs on CPUs without it. `a` holds int8 + 128 as uint8. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_X86) && (defined(__GNUC__) || defined(__clang__))

#include <immintrin.h>

#define RAYO_VNNI __attribute__((target("avx2,avxvnni")))

RAYO_VNNI static inline int32_t hsum(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

RAYO_VNNI static inline __m256i load(const int8_t *p) {
  return _mm256_loadu_si256((const __m256i *)(const void *)p);
}

RAYO_VNNI void rayo_gemm_u8i8_avxvnni(const int8_t *restrict a, size_t lda, size_t m,
                                      const int8_t *restrict w, size_t ldw, size_t n, size_t k,
                                      int32_t *restrict c, size_t ldc) {
  size_t i = 0;
  for (; i + 2 <= m; i += 2) {
    const int8_t *a0 = a + i * lda, *a1 = a0 + lda;
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const int8_t *w0 = w + j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
      __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
      __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
      for (size_t p = 0; p < k; p += 32) {
        const __m256i x0 = load(a0 + p), x1 = load(a1 + p);
        const __m256i y0 = load(w0 + p), y1 = load(w1 + p), y2 = load(w2 + p), y3 = load(w3 + p);
        c00 = _mm256_dpbusd_avx_epi32(c00, x0, y0);
        c01 = _mm256_dpbusd_avx_epi32(c01, x0, y1);
        c02 = _mm256_dpbusd_avx_epi32(c02, x0, y2);
        c03 = _mm256_dpbusd_avx_epi32(c03, x0, y3);
        c10 = _mm256_dpbusd_avx_epi32(c10, x1, y0);
        c11 = _mm256_dpbusd_avx_epi32(c11, x1, y1);
        c12 = _mm256_dpbusd_avx_epi32(c12, x1, y2);
        c13 = _mm256_dpbusd_avx_epi32(c13, x1, y3);
      }
      int32_t *r0 = c + i * ldc + j, *r1 = r0 + ldc;
      r0[0] = hsum(c00);
      r0[1] = hsum(c01);
      r0[2] = hsum(c02);
      r0[3] = hsum(c03);
      r1[0] = hsum(c10);
      r1[1] = hsum(c11);
      r1[2] = hsum(c12);
      r1[3] = hsum(c13);
    }
    for (; j < n; j++) {
      for (size_t r = 0; r < 2; r++) {
        const uint8_t *ar = (const uint8_t *)(const void *)(a + (i + r) * lda);
        const int8_t *wj = w + j * ldw;
        int32_t acc = 0;
        for (size_t p = 0; p < k; p++) acc += (int32_t)ar[p] * (int32_t)wj[p];
        c[(i + r) * ldc + j] = acc;
      }
    }
  }
  for (; i < m; i++) {
    const uint8_t *ar = (const uint8_t *)(const void *)(a + i * lda);
    for (size_t j = 0; j < n; j++) {
      const int8_t *wj = w + j * ldw;
      int32_t acc = 0;
      for (size_t p = 0; p < k; p++) acc += (int32_t)ar[p] * (int32_t)wj[p];
      c[i * ldc + j] = acc;
    }
  }
}

#endif
