/* int8 products with SSE4.1 and AVX2: sign-extend to int16, then multiply
 * and add pairs into int32 (pmaddwd), which cannot overflow, unlike
 * pmaddubsw. Two rows of `a` by four of `w` per step with AVX2, one by four
 * with SSE4.1, so each load is used several times. And 4-bit weights by
 * int8 activations (kernels.h). Each set is compiled for its instructions
 * with a target attribute (or the build's own flags), so a library built
 * for an older x86 still runs AVX2 where the CPU has it (select.c asks
 * cpuid). Intrinsics, target attributes and cpuid are compiler extensions,
 * confined to the x86 files. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_X86_SIMD)

#include <immintrin.h>

#if defined(__AVX2__) || !(defined(__GNUC__) || defined(__clang__))
#define RAYO_AVX2
#else
#define RAYO_AVX2 __attribute__((target("avx2,fma")))
#endif
#if defined(__SSE4_1__) || !(defined(__GNUC__) || defined(__clang__))
#define RAYO_SSE41
#else
#define RAYO_SSE41 __attribute__((target("sse4.1")))
#endif

#if defined(RAYO_KERNELS_X86_AVX2)

RAYO_AVX2 static __m256i load16_avx2(const rayo_i8 *p) {
  return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)p));
}

RAYO_AVX2 static rayo_i32 hsum_avx2(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return (rayo_i32)_mm_cvtsi128_si32(s);
}

/* Rows a0, a1 by w rows j .. j+3. */
RAYO_AVX2 static void block2x4_avx2(const rayo_i8 *a0, const rayo_i8 *a1, const rayo_i8 *w,
                                    size_t ldw, size_t k, rayo_i32 *r0, rayo_i32 *r1) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
  __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  size_t p;
  for (p = 0; p < k; p += 16) {
    __m256i x0 = load16_avx2(a0 + p), x1 = load16_avx2(a1 + p);
    __m256i y0 = load16_avx2(w0 + p), y1 = load16_avx2(w1 + p), y2 = load16_avx2(w2 + p),
            y3 = load16_avx2(w3 + p);
    c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x0, y0));
    c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x0, y1));
    c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(x0, y2));
    c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(x0, y3));
    c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x1, y0));
    c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x1, y1));
    c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(x1, y2));
    c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(x1, y3));
  }
  r0[0] = hsum_avx2(c00);
  r0[1] = hsum_avx2(c01);
  r0[2] = hsum_avx2(c02);
  r0[3] = hsum_avx2(c03);
  r1[0] = hsum_avx2(c10);
  r1[1] = hsum_avx2(c11);
  r1[2] = hsum_avx2(c12);
  r1[3] = hsum_avx2(c13);
}

RAYO_AVX2 void rayo_gemm_i8_avx2(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                                 const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                                 rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i = 0, j;
  for (; i + 2 <= m; i += 2) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) {
      block2x4_avx2(a0, a0 + lda, w + j * ldw, ldw, k, c + i * ldc + j, c + (i + 1) * ldc + j);
    }
    if (j < n) rayo_gemm_i8_generic(a0, lda, 2, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
  if (i < m) {
    /* The last row of an odd count (a decoder's one token): the two-row
     * block on it twice, the second copy dropped. */
    const rayo_i8 *a0 = a + i * lda;
    rayo_i32 spare[4];
    for (j = 0; j + 4 <= n; j += 4)
      block2x4_avx2(a0, a0, w + j * ldw, ldw, k, c + i * ldc + j, spare);
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

/* 4-bit weights: a group's 16 bytes unpacked to 32 int8 (low four bits
 * then high four, minus 8), multiplied by |a| with a's sign moved onto the
 * weight (pmaddubsw, at most 2 * 127 * 8 per pair, no saturation), pairs
 * added into int32 lanes of four products (pmaddwd), then scaled in float
 * and added into eight lanes, as the generic kernel does. */
RAYO_AVX2 static float dot_q4_avx2(const rayo_i8 *a, const float *as, const rayo_u8 *w,
                                   const float *ws, size_t groups) {
  const __m128i low = _mm_set1_epi8(15);
  const __m256i eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
  __m256 acc = _mm256_setzero_ps();
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(w + g * 16));
    const __m128i lo = _mm_and_si128(b, low), hi = _mm_and_si128(_mm_srli_epi16(b, 4), low);
    const __m256i wv =
        _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), eight);
    const __m256i av = _mm256_loadu_si256((const __m256i *)(const void *)(a + g * 32));
    const __m256i p = _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_sign_epi8(av, av), _mm256_sign_epi8(wv, av)), ones);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(as[g] * ws[g])));
  }
  _mm256_storeu_ps(lanes, acc);
  return rayo_q4_lanes_sum(lanes);
}

/* Two rows of activations by one of weights, each weight group unpacked
 * once: the prompt's product. Each row's sums are exactly dot_q4_avx2's. */
RAYO_AVX2 static void dot2_q4_avx2(const rayo_i8 *a0, const rayo_i8 *a1, const float *s0,
                                   const float *s1, const rayo_u8 *w, const float *ws,
                                   size_t groups, float *r0, float *r1) {
  const __m128i low = _mm_set1_epi8(15);
  const __m256i eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(w + g * 16));
    const __m128i lo = _mm_and_si128(b, low), hi = _mm_and_si128(_mm_srli_epi16(b, 4), low);
    const __m256i wv =
        _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), eight);
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(a0 + g * 32));
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(a1 + g * 32));
    const __m256i p0 = _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_sign_epi8(x0, x0), _mm256_sign_epi8(wv, x0)), ones);
    const __m256i p1 = _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_sign_epi8(x1, x1), _mm256_sign_epi8(wv, x1)), ones);
    acc0 =
        _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0[g] * ws[g])));
    acc1 =
        _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(s1[g] * ws[g])));
  }
  _mm256_storeu_ps(lanes, acc0);
  *r0 = rayo_q4_lanes_sum(lanes);
  _mm256_storeu_ps(lanes, acc1);
  *r1 = rayo_q4_lanes_sum(lanes);
}

RAYO_AVX2 void rayo_gemm_q4_avx2(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                                 size_t m, const rayo_u8 *RAYO_RESTRICT w,
                                 const float *RAYO_RESTRICT wscale, size_t n, size_t k,
                                 float *RAYO_RESTRICT c, size_t ldc) {
  const size_t groups = k / RAYO_Q4_GROUP;
  size_t i, j;
  for (j = 0; j < n; j++) {
    const rayo_u8 *wj = w + j * (k / 2);
    const float *sj = wscale + j * groups;
    for (i = 0; i + 2 <= m; i += 2) {
      dot2_q4_avx2(a + i * k, a + (i + 1) * k, ascale + i * groups, ascale + (i + 1) * groups, wj,
                   sj, groups, &c[i * ldc + j], &c[(i + 1) * ldc + j]);
    }
    if (i < m) c[i * ldc + j] = dot_q4_avx2(a + i * k, ascale + i * groups, wj, sj, groups);
  }
}
#endif

/* SSE4.1 */

RAYO_SSE41 static __m128i load8_sse41(const rayo_i8 *p) {
  return _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(const void *)p));
}

RAYO_SSE41 static rayo_i32 hsum_sse41(__m128i s) {
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return (rayo_i32)_mm_cvtsi128_si32(s);
}

RAYO_SSE41 static void block1x4_sse41(const rayo_i8 *a0, const rayo_i8 *w, size_t ldw, size_t k,
                                      rayo_i32 *r) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  __m128i c0 = _mm_setzero_si128(), c1 = c0, c2 = c0, c3 = c0;
  size_t p;
  for (p = 0; p < k; p += 8) {
    __m128i x = load8_sse41(a0 + p);
    c0 = _mm_add_epi32(c0, _mm_madd_epi16(x, load8_sse41(w0 + p)));
    c1 = _mm_add_epi32(c1, _mm_madd_epi16(x, load8_sse41(w1 + p)));
    c2 = _mm_add_epi32(c2, _mm_madd_epi16(x, load8_sse41(w2 + p)));
    c3 = _mm_add_epi32(c3, _mm_madd_epi16(x, load8_sse41(w3 + p)));
  }
  r[0] = hsum_sse41(c0);
  r[1] = hsum_sse41(c1);
  r[2] = hsum_sse41(c2);
  r[3] = hsum_sse41(c3);
}

RAYO_SSE41 void rayo_gemm_i8_sse41(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                                   const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                                   rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j;
  for (i = 0; i < m; i++) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) block1x4_sse41(a0, w + j * ldw, ldw, k, c + i * ldc + j);
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

/* 4-bit weights, as the AVX2 kernel, in two halves of 16. */
RAYO_SSE41 static float dot_q4_sse41(const rayo_i8 *a, const float *as, const rayo_u8 *w,
                                     const float *ws, size_t groups) {
  const __m128i low = _mm_set1_epi8(15), eight = _mm_set1_epi8(8), ones = _mm_set1_epi16(1);
  __m128 acc0 = _mm_setzero_ps(), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(w + g * 16));
    const __m128i wl = _mm_sub_epi8(_mm_and_si128(b, low), eight);
    const __m128i wh = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), low), eight);
    const __m128i al = _mm_loadu_si128((const __m128i *)(const void *)(a + g * 32));
    const __m128i ah = _mm_loadu_si128((const __m128i *)(const void *)(a + g * 32 + 16));
    const __m128i pl =
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(al, al), _mm_sign_epi8(wl, al)), ones);
    const __m128i ph =
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(ah, ah), _mm_sign_epi8(wh, ah)), ones);
    const __m128 d = _mm_set1_ps(as[g] * ws[g]);
    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(pl), d));
    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(ph), d));
  }
  _mm_storeu_ps(lanes, acc0);
  _mm_storeu_ps(lanes + 4, acc1);
  return rayo_q4_lanes_sum(lanes);
}

RAYO_SSE41 void rayo_gemm_q4_sse41(const rayo_i8 *RAYO_RESTRICT a,
                                   const float *RAYO_RESTRICT ascale, size_t m,
                                   const rayo_u8 *RAYO_RESTRICT w,
                                   const float *RAYO_RESTRICT wscale, size_t n, size_t k,
                                   float *RAYO_RESTRICT c, size_t ldc) {
  const size_t groups = k / RAYO_Q4_GROUP;
  size_t i, j;
  for (j = 0; j < n; j++) {
    const rayo_u8 *wj = w + j * (k / 2);
    const float *sj = wscale + j * groups;
    for (i = 0; i < m; i++)
      c[i * ldc + j] = dot_q4_sse41(a + i * k, ascale + i * groups, wj, sj, groups);
  }
}

#else
/* ISO C needs a declaration in every translation unit. */
typedef int rayo_x86_unused;
#endif
