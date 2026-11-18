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

#include "kernels/half.h"

#if defined(__AVX2__) || !(defined(__GNUC__) || defined(__clang__))
#define RAYO_AVX2
#else
#define RAYO_AVX2 __attribute__((target("avx2,fma")))
#endif
/* Inlined into each kernel, so each has its own copy for its group size
 * and scale type, with no test in the inner loop. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_INLINE __inline__ __attribute__((always_inline))
#else
#define RAYO_INLINE
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

/* Group-wise weights (kernels.h): a group unpacked to 32 int8 (4 bits:
 * the low four bits then the high four, minus 8; 3 bits: the two-bit
 * planes, each shifted down, with the third bits spread from their word,
 * minus 4), multiplied by |a| with a's sign moved onto the weight
 * (pmaddubsw, at most 2 * 127 * 8 per pair, no saturation), pairs added
 * into int32 lanes of four products (pmaddwd), then scaled in float and
 * added into eight lanes, as the generic kernel does. */
/* float16 scales to float32 eight at a time (exactly as rayo_half_to_float:
 * normal values move their exponent, subnormal ones are their mantissa
 * times 2^-24), the rest one by one. */
RAYO_AVX2 static RAYO_INLINE void half_row_avx2(const rayo_u16 *h, size_t n, float *out) {
  const __m256i low15 = _mm256_set1_epi32(0x7fff), top = _mm256_set1_epi32(0x8000);
  const __m256i bias = _mm256_set1_epi32(0x38000000), inf = _mm256_set1_epi32(0x7f800000);
  const __m256i big = _mm256_set1_epi32(0x7bff), small = _mm256_set1_epi32(0x0400);
  const __m256 tiny = _mm256_set1_ps(5.9604644775390625e-8f); /* 2^-24 */
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i x =
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(const void *)(h + i)));
    const __m256i em = _mm256_and_si256(x, low15);
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(x, top), 16);
    const __m256i norm = _mm256_add_epi32(_mm256_slli_epi32(em, 13), bias);
    const __m256i bits = _mm256_blendv_epi8(norm, inf, _mm256_cmpgt_epi32(em, big));
    const __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(em), tiny);
    const __m256 f = _mm256_blendv_ps(_mm256_castsi256_ps(bits), sub,
                                      _mm256_castsi256_ps(_mm256_cmpgt_epi32(small, em)));
    _mm256_storeu_ps(out + i, _mm256_or_ps(f, _mm256_castsi256_ps(sign)));
  }
  for (; i < n; i++) out[i] = rayo_half_to_float(h[i]);
}

RAYO_AVX2 static RAYO_INLINE __m256i unpack_q4_avx2(const rayo_u8 *g) {
  const __m128i low = _mm_set1_epi8(15);
  const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)g);
  const __m128i lo = _mm_and_si128(b, low), hi = _mm_and_si128(_mm_srli_epi16(b, 4), low);
  return _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1),
                         _mm256_set1_epi8(8));
}

RAYO_AVX2 static RAYO_INLINE __m256i unpack_q3_avx2(const rayo_u8 *g) {
  const __m256i bit = _mm256_set_epi8(-128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1,
                                      -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
  const __m256i spread = _mm256_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
                                         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const rayo_u32 word =
      (rayo_u32)g[8] | (rayo_u32)g[9] << 8 | (rayo_u32)g[10] << 16 | (rayo_u32)g[11] << 24;
  __m256i low = _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(const void *)g));
  __m256i high = _mm256_shuffle_epi8(_mm256_set1_epi32((int)word), spread);
  low =
      _mm256_and_si256(_mm256_srlv_epi64(low, _mm256_set_epi64x(6, 4, 2, 0)), _mm256_set1_epi8(3));
  high = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high, bit), bit), _mm256_set1_epi8(4));
  return _mm256_sub_epi8(_mm256_or_si256(low, high), _mm256_set1_epi8(4));
}

RAYO_AVX2 static RAYO_INLINE __m256 scaled_avx2(__m256 acc, __m256i av, __m256i wv, float d) {
  const __m256i p =
      _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(av, av), _mm256_sign_epi8(wv, av)),
                        _mm256_set1_epi16(1));
  return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d)));
}

RAYO_AVX2 static RAYO_INLINE float dot_q_avx2(const rayo_i8 *a, const float *as, const rayo_u8 *w,
                                              const void *ws, rayo_bool half, size_t first,
                                              size_t groups, size_t bytes) {
  __m256 acc = _mm256_setzero_ps();
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const __m256i wv = bytes == RAYO_Q4_GROUP_BYTES ? unpack_q4_avx2(w + g * bytes)
                                                    : unpack_q3_avx2(w + g * bytes);
    const __m256i av = _mm256_loadu_si256((const __m256i *)(const void *)(a + g * 32));
    acc = scaled_avx2(acc, av, wv, as[g] * rayo_group_scale(ws, half, first + g));
  }
  _mm256_storeu_ps(lanes, acc);
  return rayo_q4_lanes_sum(lanes);
}

/* Two rows of activations by one of weights, each weight group unpacked
 * once: the prompt's product. Each row's sums are exactly dot_q_avx2's. */
RAYO_AVX2 static RAYO_INLINE void dot2_q_avx2(const rayo_i8 *a0, const rayo_i8 *a1, const float *s0,
                                              const float *s1, const rayo_u8 *w, const void *ws,
                                              rayo_bool half, size_t first, size_t groups,
                                              size_t bytes, float *r0, float *r1) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const __m256i wv = bytes == RAYO_Q4_GROUP_BYTES ? unpack_q4_avx2(w + g * bytes)
                                                    : unpack_q3_avx2(w + g * bytes);
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(a0 + g * 32));
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(a1 + g * 32));
    const float ss = rayo_group_scale(ws, half, first + g);
    acc0 = scaled_avx2(acc0, x0, wv, s0[g] * ss);
    acc1 = scaled_avx2(acc1, x1, wv, s1[g] * ss);
  }
  _mm256_storeu_ps(lanes, acc0);
  *r0 = rayo_q4_lanes_sum(lanes);
  _mm256_storeu_ps(lanes, acc1);
  *r1 = rayo_q4_lanes_sum(lanes);
}

/* One row of activations by four of weights (`stride` bytes apart, their
 * scales in st[4 * g + r]): a decoder's product, four independent sums so
 * the additions overlap. Each output's sums are exactly dot_q_avx2's. */
RAYO_AVX2 static RAYO_INLINE void dot4_q_avx2(const rayo_i8 *a, const float *as, const rayo_u8 *w,
                                              size_t stride, const float *st, size_t groups,
                                              size_t bytes, float *c) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    const rayo_u8 *wg = w + g * bytes;
    const __m256i av = _mm256_loadu_si256((const __m256i *)(const void *)(a + g * 32));
    const float s = as[g];
#define RAYO_UNPACK(p) (bytes == RAYO_Q4_GROUP_BYTES ? unpack_q4_avx2(p) : unpack_q3_avx2(p))
    acc0 = scaled_avx2(acc0, av, RAYO_UNPACK(wg), s * st[4 * g]);
    acc1 = scaled_avx2(acc1, av, RAYO_UNPACK(wg + stride), s * st[4 * g + 1]);
    acc2 = scaled_avx2(acc2, av, RAYO_UNPACK(wg + 2 * stride), s * st[4 * g + 2]);
    acc3 = scaled_avx2(acc3, av, RAYO_UNPACK(wg + 3 * stride), s * st[4 * g + 3]);
#undef RAYO_UNPACK
  }
  _mm256_storeu_ps(lanes, acc0);
  c[0] = rayo_q4_lanes_sum(lanes);
  _mm256_storeu_ps(lanes, acc1);
  c[1] = rayo_q4_lanes_sum(lanes);
  _mm256_storeu_ps(lanes, acc2);
  c[2] = rayo_q4_lanes_sum(lanes);
  _mm256_storeu_ps(lanes, acc3);
  c[3] = rayo_q4_lanes_sum(lanes);
}

RAYO_AVX2 static RAYO_INLINE void gemm_q_avx2(const rayo_i8 *RAYO_RESTRICT a,
                                              const float *RAYO_RESTRICT ascale, size_t m,
                                              const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                              rayo_bool half, size_t n, size_t k,
                                              float *RAYO_RESTRICT c, size_t ldc, size_t bytes) {
  const size_t groups = k / RAYO_Q4_GROUP;
  float row[RAYO_SCALE_ROW], st[4 * RAYO_SCALE_ROW];
  size_t i, j = 0, r, g;
  if (m == 1 && groups <= RAYO_SCALE_ROW) {
    /* A decoder's single row: four rows of weights at a time. */
    for (; j + 4 <= n; j += 4) {
      for (r = 0; r < 4; r++) {
        const float *sr = (const float *)wscale + (j + r) * groups;
        if (half) {
          half_row_avx2((const rayo_u16 *)wscale + (j + r) * groups, groups, row);
          sr = row;
        }
        for (g = 0; g < groups; g++) st[4 * g + r] = sr[g];
      }
      dot4_q_avx2(a, ascale, w + j * groups * bytes, groups * bytes, st, groups, bytes, c + j);
    }
  }
  for (; j < n; j++) {
    const rayo_u8 *wj = w + j * groups * bytes;
    /* A row's float16 scales converted once, for all its groups and rows. */
    const rayo_bool conv = half && groups <= RAYO_SCALE_ROW;
    const void *ws = conv ? (const void *)row : wscale;
    const rayo_bool h = conv ? RAYO_FALSE : half;
    const size_t first = conv ? 0 : j * groups;
    if (conv) half_row_avx2((const rayo_u16 *)wscale + j * groups, groups, row);
    for (i = 0; i + 2 <= m; i += 2) {
      dot2_q_avx2(a + i * k, a + (i + 1) * k, ascale + i * groups, ascale + (i + 1) * groups, wj,
                  ws, h, first, groups, bytes, &c[i * ldc + j], &c[(i + 1) * ldc + j]);
    }
    if (i < m) {
      c[i * ldc + j] = dot_q_avx2(a + i * k, ascale + i * groups, wj, ws, h, first, groups, bytes);
    }
  }
}

RAYO_AVX2 void rayo_gemm_q4_avx2(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                                 size_t m, const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                 rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c,
                                 size_t ldc) {
  if (half) {
    gemm_q_avx2(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  } else {
    gemm_q_avx2(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  }
}

RAYO_AVX2 void rayo_gemm_q3_avx2(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                                 size_t m, const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                 rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c,
                                 size_t ldc) {
  if (half) {
    gemm_q_avx2(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
  } else {
    gemm_q_avx2(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
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

/* Group-wise weights, as the AVX2 kernel, in two halves of 16. */
/* float16 scales to float32 four at a time, as half_row_avx2. */
RAYO_SSE41 static RAYO_INLINE void half_row_sse41(const rayo_u16 *h, size_t n, float *out) {
  const __m128i low15 = _mm_set1_epi32(0x7fff), top = _mm_set1_epi32(0x8000);
  const __m128i bias = _mm_set1_epi32(0x38000000), inf = _mm_set1_epi32(0x7f800000);
  const __m128i big = _mm_set1_epi32(0x7bff), small = _mm_set1_epi32(0x0400);
  const __m128 tiny = _mm_set1_ps(5.9604644775390625e-8f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m128i x = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(h + i)));
    const __m128i em = _mm_and_si128(x, low15);
    const __m128i sign = _mm_slli_epi32(_mm_and_si128(x, top), 16);
    const __m128i norm = _mm_add_epi32(_mm_slli_epi32(em, 13), bias);
    const __m128i bits = _mm_blendv_epi8(norm, inf, _mm_cmpgt_epi32(em, big));
    const __m128 sub = _mm_mul_ps(_mm_cvtepi32_ps(em), tiny);
    const __m128 f =
        _mm_blendv_ps(_mm_castsi128_ps(bits), sub, _mm_castsi128_ps(_mm_cmpgt_epi32(small, em)));
    _mm_storeu_ps(out + i, _mm_or_ps(f, _mm_castsi128_ps(sign)));
  }
  for (; i < n; i++) out[i] = rayo_half_to_float(h[i]);
}

RAYO_SSE41 static RAYO_INLINE void unpack_q4_sse41(const rayo_u8 *g, __m128i *wl, __m128i *wh) {
  const __m128i low = _mm_set1_epi8(15), eight = _mm_set1_epi8(8);
  const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)g);
  *wl = _mm_sub_epi8(_mm_and_si128(b, low), eight);
  *wh = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), low), eight);
}

RAYO_SSE41 static RAYO_INLINE void unpack_q3_sse41(const rayo_u8 *g, __m128i *wl, __m128i *wh) {
  const __m128i bit = _mm_set_epi8(-128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
  const __m128i three = _mm_set1_epi8(3), four = _mm_set1_epi8(4);
  const rayo_u32 word =
      (rayo_u32)g[8] | (rayo_u32)g[9] << 8 | (rayo_u32)g[10] << 16 | (rayo_u32)g[11] << 24;
  const __m128i b = _mm_loadl_epi64((const __m128i *)(const void *)g);
  const __m128i h = _mm_set1_epi32((int)word);
  __m128i hl = _mm_shuffle_epi8(h, _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
  __m128i hh = _mm_shuffle_epi8(h, _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2));
  hl = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hl, bit), bit), four);
  hh = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hh, bit), bit), four);
  *wl = _mm_and_si128(_mm_unpacklo_epi64(b, _mm_srli_epi64(b, 2)), three);
  *wh = _mm_and_si128(_mm_unpacklo_epi64(_mm_srli_epi64(b, 4), _mm_srli_epi64(b, 6)), three);
  *wl = _mm_sub_epi8(_mm_or_si128(*wl, hl), four);
  *wh = _mm_sub_epi8(_mm_or_si128(*wh, hh), four);
}

RAYO_SSE41 static RAYO_INLINE float dot_q_sse41(const rayo_i8 *a, const float *as, const rayo_u8 *w,
                                                const void *ws, rayo_bool half, size_t first,
                                                size_t groups, size_t bytes) {
  const __m128i ones = _mm_set1_epi16(1);
  __m128 acc0 = _mm_setzero_ps(), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    __m128i wl, wh;
    const __m128i al = _mm_loadu_si128((const __m128i *)(const void *)(a + g * 32));
    const __m128i ah = _mm_loadu_si128((const __m128i *)(const void *)(a + g * 32 + 16));
    __m128i pl, ph;
    __m128 d;
    if (bytes == RAYO_Q4_GROUP_BYTES) {
      unpack_q4_sse41(w + g * bytes, &wl, &wh);
    } else {
      unpack_q3_sse41(w + g * bytes, &wl, &wh);
    }
    pl = _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(al, al), _mm_sign_epi8(wl, al)), ones);
    ph = _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(ah, ah), _mm_sign_epi8(wh, ah)), ones);
    d = _mm_set1_ps(as[g] * rayo_group_scale(ws, half, first + g));
    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(pl), d));
    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(ph), d));
  }
  _mm_storeu_ps(lanes, acc0);
  _mm_storeu_ps(lanes + 4, acc1);
  return rayo_q4_lanes_sum(lanes);
}

RAYO_SSE41 static RAYO_INLINE void gemm_q_sse41(const rayo_i8 *RAYO_RESTRICT a,
                                                const float *RAYO_RESTRICT ascale, size_t m,
                                                const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                                rayo_bool half, size_t n, size_t k,
                                                float *RAYO_RESTRICT c, size_t ldc, size_t bytes) {
  const size_t groups = k / RAYO_Q4_GROUP;
  float row[RAYO_SCALE_ROW];
  size_t i, j;
  for (j = 0; j < n; j++) {
    const rayo_u8 *wj = w + j * groups * bytes;
    /* A row's float16 scales converted once, for all its groups and rows. */
    const rayo_bool conv = half && groups <= RAYO_SCALE_ROW;
    const void *ws = conv ? (const void *)row : wscale;
    const rayo_bool h = conv ? RAYO_FALSE : half;
    const size_t first = conv ? 0 : j * groups;
    if (conv) half_row_sse41((const rayo_u16 *)wscale + j * groups, groups, row);
    for (i = 0; i < m; i++) {
      c[i * ldc + j] = dot_q_sse41(a + i * k, ascale + i * groups, wj, ws, h, first, groups, bytes);
    }
  }
}

RAYO_SSE41 void rayo_gemm_q4_sse41(const rayo_i8 *RAYO_RESTRICT a,
                                   const float *RAYO_RESTRICT ascale, size_t m,
                                   const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                   rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c,
                                   size_t ldc) {
  if (half) {
    gemm_q_sse41(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  } else {
    gemm_q_sse41(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  }
}

RAYO_SSE41 void rayo_gemm_q3_sse41(const rayo_i8 *RAYO_RESTRICT a,
                                   const float *RAYO_RESTRICT ascale, size_t m,
                                   const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                   rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c,
                                   size_t ldc) {
  if (half) {
    gemm_q_sse41(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
  } else {
    gemm_q_sse41(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
  }
}

#else
/* ISO C needs a declaration in every translation unit. */
typedef int rayo_x86_unused;
#endif
