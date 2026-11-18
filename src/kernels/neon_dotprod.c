/* int8 products with the ARMv8.2 dot product (SDOT: four int8 by int8
 * products added into each int32 lane), and the CPU check that chooses them
 * at run time. Compiled for it with a target attribute, so the library still
 * runs on ARMv8.0 phones. Intrinsics, target attributes, getauxval and
 * sysctlbyname are extensions, confined to this file. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_NEON_DOTPROD)

#include <arm_neon.h>

#include "kernels/half.h"
#include "kernels/neon_unpack.h"
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__clang__)
#define RAYO_DOTPROD __attribute__((target("dotprod")))
#else
#define RAYO_DOTPROD __attribute__((target("+dotprod")))
#endif

RAYO_DOTPROD static void block2x4(const rayo_i8 *a0, const rayo_i8 *a1, const rayo_i8 *w,
                                  size_t ldw, size_t k, rayo_i32 *r0, rayo_i32 *r1) {
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
  int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  size_t p;
  for (p = 0; p < k; p += 16) {
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
  r0[0] = (rayo_i32)vaddvq_s32(c00);
  r0[1] = (rayo_i32)vaddvq_s32(c01);
  r0[2] = (rayo_i32)vaddvq_s32(c02);
  r0[3] = (rayo_i32)vaddvq_s32(c03);
  r1[0] = (rayo_i32)vaddvq_s32(c10);
  r1[1] = (rayo_i32)vaddvq_s32(c11);
  r1[2] = (rayo_i32)vaddvq_s32(c12);
  r1[3] = (rayo_i32)vaddvq_s32(c13);
}

RAYO_DOTPROD void rayo_gemm_i8_neon_dotprod(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                                            const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n,
                                            size_t k, rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i = 0, j;
  for (; i + 2 <= m; i += 2) {
    const rayo_i8 *a0 = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) {
      block2x4(a0, a0 + lda, w + j * ldw, ldw, k, c + i * ldc + j, c + (i + 1) * ldc + j);
    }
    if (j < n) rayo_gemm_i8_generic(a0, lda, 2, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
  if (i < m) {
    /* The last row of an odd count (a decoder's one token): the two-row
     * block on it twice, the second copy dropped. */
    const rayo_i8 *a0 = a + i * lda;
    rayo_i32 spare[4];
    for (j = 0; j + 4 <= n; j += 4) block2x4(a0, a0, w + j * ldw, ldw, k, c + i * ldc + j, spare);
    if (j < n) rayo_gemm_i8_generic(a0, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}

/* Group-wise weights as neon.c's kernel (and its unpacking), with SDOT
 * for the lanes of four. */
RAYO_DOTPROD static RAYO_NEON_INLINE float dot_q(const rayo_i8 *a, const float *as,
                                                 const rayo_u8 *w, const void *ws, rayo_bool half,
                                                 size_t first, size_t groups, size_t bytes) {
  const int32x4_t zero = vdupq_n_s32(0);
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0;
  float lanes[8];
  size_t g;
  for (g = 0; g < groups; g++) {
    int8x16_t wl, wh;
    int32x4_t pl, ph;
    float d;
    if (bytes == RAYO_Q4_GROUP_BYTES) {
      neon_unpack_q4(w + g * bytes, &wl, &wh);
    } else {
      neon_unpack_q3(w + g * bytes, &wl, &wh);
    }
    d = as[g] * rayo_group_scale(ws, half, first + g);
    pl = vdotq_s32(zero, vld1q_s8(a + g * 32), wl);
    ph = vdotq_s32(zero, vld1q_s8(a + g * 32 + 16), wh);
    acc0 = vaddq_f32(acc0, vmulq_n_f32(vcvtq_f32_s32(pl), d));
    acc1 = vaddq_f32(acc1, vmulq_n_f32(vcvtq_f32_s32(ph), d));
  }
  vst1q_f32(lanes, acc0);
  vst1q_f32(lanes + 4, acc1);
  return rayo_q4_lanes_sum(lanes);
}

RAYO_DOTPROD static RAYO_NEON_INLINE void gemm_q(const rayo_i8 *RAYO_RESTRICT a,
                                                 const float *RAYO_RESTRICT ascale, size_t m,
                                                 const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                                 rayo_bool half, size_t n, size_t k,
                                                 float *RAYO_RESTRICT c, size_t ldc, size_t bytes) {
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

RAYO_DOTPROD void rayo_gemm_q4_neon_dotprod(const rayo_i8 *RAYO_RESTRICT a,
                                            const float *RAYO_RESTRICT ascale, size_t m,
                                            const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                            rayo_bool half, size_t n, size_t k,
                                            float *RAYO_RESTRICT c, size_t ldc) {
  if (half) {
    gemm_q(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  } else {
    gemm_q(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
  }
}

RAYO_DOTPROD void rayo_gemm_q3_neon_dotprod(const rayo_i8 *RAYO_RESTRICT a,
                                            const float *RAYO_RESTRICT ascale, size_t m,
                                            const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                                            rayo_bool half, size_t n, size_t k,
                                            float *RAYO_RESTRICT c, size_t ldc) {
  if (half) {
    gemm_q(a, ascale, m, w, wscale, RAYO_TRUE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
  } else {
    gemm_q(a, ascale, m, w, wscale, RAYO_FALSE, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
  }
}

rayo_bool rayo_cpu_has_dotprod(void) {
#if defined(__linux__)
  return (getauxval(AT_HWCAP) & (1ul << 20)) != 0; /* HWCAP_ASIMDDP */
#elif defined(__APPLE__)
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, NULL, 0) == 0 && v != 0;
#else
  return RAYO_FALSE;
#endif
}

#else
typedef int rayo_neon_dotprod_unused;
#endif
