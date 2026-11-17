/* int8 products with SSE4.1 or AVX2: sign-extend to int16, then multiply
 * and add pairs into int32 (pmaddwd), which cannot overflow, unlike
 * pmaddubsw. Two rows of `a` by four of `w` per step with AVX2, one by four
 * with SSE4.1, so each load is used several times. Intrinsics are a
 * compiler extension: this file is built only when the target has them. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_X86_SIMD)

#include <immintrin.h>

#if defined(__AVX2__)

static __m256i load16(const rayo_i8 *p) {
  return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)p));
}

static rayo_i32 hsum(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return (rayo_i32)_mm_cvtsi128_si32(s);
}

/* Rows a0, a1 by w rows j .. j+3. */
static void block2x4(const rayo_i8 *a0, const rayo_i8 *a1, const rayo_i8 *w, size_t ldw, size_t k,
                     rayo_i32 *r0, rayo_i32 *r1) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
  __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  size_t p;
  for (p = 0; p < k; p += 16) {
    __m256i x0 = load16(a0 + p), x1 = load16(a1 + p);
    __m256i y0 = load16(w0 + p), y1 = load16(w1 + p), y2 = load16(w2 + p), y3 = load16(w3 + p);
    c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x0, y0));
    c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x0, y1));
    c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(x0, y2));
    c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(x0, y3));
    c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x1, y0));
    c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x1, y1));
    c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(x1, y2));
    c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(x1, y3));
  }
  r0[0] = hsum(c00);
  r0[1] = hsum(c01);
  r0[2] = hsum(c02);
  r0[3] = hsum(c03);
  r1[0] = hsum(c10);
  r1[1] = hsum(c11);
  r1[2] = hsum(c12);
  r1[3] = hsum(c13);
}

void rayo_gemm_i8_x86(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                      const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                      rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i = 0, j;
  for (; i + 2 <= m; i += 2) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) {
      block2x4(a0, a0 + lda, w + j * ldw, ldw, k, c + i * ldc + j, c + (i + 1) * ldc + j);
    }
    if (j < n) rayo_gemm_i8_generic(a0, lda, 2, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
  if (i < m) rayo_gemm_i8_generic(a + i * lda, lda, m - i, w, ldw, n, k, c + i * ldc, ldc);
}

#else /* SSE4.1 */

static __m128i load8(const rayo_i8 *p) {
  return _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(const void *)p));
}

static rayo_i32 hsum(__m128i s) {
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return (rayo_i32)_mm_cvtsi128_si32(s);
}

static void block1x4(const rayo_i8 *a0, const rayo_i8 *w, size_t ldw, size_t k, rayo_i32 *r) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  __m128i c0 = _mm_setzero_si128(), c1 = c0, c2 = c0, c3 = c0;
  size_t p;
  for (p = 0; p < k; p += 8) {
    __m128i x = load8(a0 + p);
    c0 = _mm_add_epi32(c0, _mm_madd_epi16(x, load8(w0 + p)));
    c1 = _mm_add_epi32(c1, _mm_madd_epi16(x, load8(w1 + p)));
    c2 = _mm_add_epi32(c2, _mm_madd_epi16(x, load8(w2 + p)));
    c3 = _mm_add_epi32(c3, _mm_madd_epi16(x, load8(w3 + p)));
  }
  r[0] = hsum(c0);
  r[1] = hsum(c1);
  r[2] = hsum(c2);
  r[3] = hsum(c3);
}

void rayo_gemm_i8_x86(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                      const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                      rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j;
  for (i = 0; i < m; i++) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) block1x4(a0, w + j * ldw, ldw, k, c + i * ldc + j);
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

#endif

#else
/* ISO C needs a declaration in every translation unit. */
typedef int rayo_x86_unused;
#endif
