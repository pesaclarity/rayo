/* int8 products with AVX-VNNI (vpdpbusd: four uint8 by int8 products added
 * into each int32 lane, exact, no saturation), and the CPU check that
 * chooses them at run time. Compiled for AVX-VNNI with a target attribute,
 * so the library still runs on CPUs without it; `a` holds int8 + 128 as
 * uint8. Intrinsics, target attributes, cpuid and xgetbv are compiler
 * extensions, confined to this file. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_X86_VNNI)

#include <cpuid.h>
#include <immintrin.h>

#define RAYO_VNNI __attribute__((target("avx2,avxvnni")))

RAYO_VNNI static rayo_i32 hsum(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return (rayo_i32)_mm_cvtsi128_si32(s);
}

RAYO_VNNI static __m256i load(const rayo_i8 *p) {
  return _mm256_loadu_si256((const __m256i *)(const void *)p);
}

RAYO_VNNI static void block2x4(const rayo_i8 *a0, const rayo_i8 *a1, const rayo_i8 *w, size_t ldw,
                               size_t k, rayo_i32 *r0, rayo_i32 *r1) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
  __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  size_t p;
  for (p = 0; p < k; p += 32) {
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
  r0[0] = hsum(c00);
  r0[1] = hsum(c01);
  r0[2] = hsum(c02);
  r0[3] = hsum(c03);
  r1[0] = hsum(c10);
  r1[1] = hsum(c11);
  r1[2] = hsum(c12);
  r1[3] = hsum(c13);
}

/* One output, `a` read as uint8. */
static rayo_i32 dot_u8i8(const rayo_i8 *a, const rayo_i8 *w, size_t k) {
  const rayo_u8 *u = (const rayo_u8 *)(const void *)a;
  rayo_i32 acc = 0;
  size_t p;
  for (p = 0; p < k; p++) acc += (rayo_i32)u[p] * (rayo_i32)w[p];
  return acc;
}

RAYO_VNNI void rayo_gemm_u8i8_avxvnni(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                                      const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n,
                                      size_t k, rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i = 0, j;
  for (; i + 2 <= m; i += 2) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) {
      block2x4(a0, a0 + lda, w + j * ldw, ldw, k, c + i * ldc + j, c + (i + 1) * ldc + j);
    }
    for (; j < n; j++) {
      c[i * ldc + j] = dot_u8i8(a0, w + j * ldw, k);
      c[(i + 1) * ldc + j] = dot_u8i8(a0 + lda, w + j * ldw, k);
    }
  }
  for (; i < m; i++) {
    for (j = 0; j < n; j++) c[i * ldc + j] = dot_u8i8(a + i * lda, w + j * ldw, k);
  }
}

/* AVX-VNNI: CPUID leaf 7 subleaf 1, EAX bit 4; with AVX2 (leaf 7, EBX bit 5)
 * and the OS saving the YMM registers (XCR0 bits 1 and 2). */
rayo_bool rayo_cpu_has_avxvnni(void) {
  unsigned a, b, c, d, lo, hi;
  if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) return RAYO_FALSE; /* OSXSAVE */
  __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  (void)hi;
  if ((lo & 6u) != 6u) return RAYO_FALSE;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(b & (1u << 5))) return RAYO_FALSE;
  if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return RAYO_FALSE;
  return (a & (1u << 4)) != 0;
}

#else
typedef int rayo_x86_vnni_unused;
#endif
