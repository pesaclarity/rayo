/* int8 products with the ARMv8.6 int8 matrix multiply (SMMLA: a 2x8 by
 * 8x2 block of int8 products added into a 2x2 block of int32, exact), and
 * the CPU check that chooses it at run time. Four rows of `a` by four of
 * `w` a step: the rows paired eight bytes at a time (zip), eight SMMLA for
 * every 16 bytes of the eight rows, twice the products of an SDOT a step.
 * The rows left over, and a decoder's single row, go to the SDOT kernel.
 * Compiled for it with a target attribute, so the library still runs on
 * older phones. Intrinsics, target attributes, getauxval and sysctlbyname
 * are extensions, confined to this file. */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_NEON_I8MM)

#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__clang__)
#define RAYO_I8MM __attribute__((target("dotprod,i8mm")))
#else
#define RAYO_I8MM __attribute__((target("+dotprod+i8mm")))
#endif

/* Rows x and y, eight bytes of each: the low halves, then the high. */
RAYO_I8MM static int8x16_t pair_lo(int8x16_t x, int8x16_t y) {
  return vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x), vreinterpretq_s64_s8(y)));
}

RAYO_I8MM static int8x16_t pair_hi(int8x16_t x, int8x16_t y) {
  return vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x), vreinterpretq_s64_s8(y)));
}

/* A 2x2 block (lanes r0w0, r0w1, r1w0, r1w1) to c. */
static void store2x2(int32x4_t v, rayo_i32 *c0, rayo_i32 *c1) {
  c0[0] = (rayo_i32)vgetq_lane_s32(v, 0);
  c0[1] = (rayo_i32)vgetq_lane_s32(v, 1);
  c1[0] = (rayo_i32)vgetq_lane_s32(v, 2);
  c1[1] = (rayo_i32)vgetq_lane_s32(v, 3);
}

RAYO_I8MM static void block4x4(const rayo_i8 *a, size_t lda, const rayo_i8 *w, size_t ldw, size_t k,
                               rayo_i32 *c, size_t ldc) {
  const rayo_i8 *a0 = a, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
  const rayo_i8 *w0 = w, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c10 = c00, c11 = c00;
  size_t p;
  for (p = 0; p < k; p += 16) {
    const int8x16_t x0 = vld1q_s8(a0 + p), x1 = vld1q_s8(a1 + p);
    const int8x16_t x2 = vld1q_s8(a2 + p), x3 = vld1q_s8(a3 + p);
    const int8x16_t y0 = vld1q_s8(w0 + p), y1 = vld1q_s8(w1 + p);
    const int8x16_t y2 = vld1q_s8(w2 + p), y3 = vld1q_s8(w3 + p);
    const int8x16_t a01l = pair_lo(x0, x1), a01h = pair_hi(x0, x1);
    const int8x16_t a23l = pair_lo(x2, x3), a23h = pair_hi(x2, x3);
    const int8x16_t w01l = pair_lo(y0, y1), w01h = pair_hi(y0, y1);
    const int8x16_t w23l = pair_lo(y2, y3), w23h = pair_hi(y2, y3);
    c00 = vmmlaq_s32(c00, a01l, w01l);
    c01 = vmmlaq_s32(c01, a01l, w23l);
    c10 = vmmlaq_s32(c10, a23l, w01l);
    c11 = vmmlaq_s32(c11, a23l, w23l);
    c00 = vmmlaq_s32(c00, a01h, w01h);
    c01 = vmmlaq_s32(c01, a01h, w23h);
    c10 = vmmlaq_s32(c10, a23h, w01h);
    c11 = vmmlaq_s32(c11, a23h, w23h);
  }
  store2x2(c00, c, c + ldc);
  store2x2(c01, c + 2, c + ldc + 2);
  store2x2(c10, c + 2 * ldc, c + 3 * ldc);
  store2x2(c11, c + 2 * ldc + 2, c + 3 * ldc + 2);
}

RAYO_I8MM void rayo_gemm_i8_neon_i8mm(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                                      const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n,
                                      size_t k, rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i = 0, j;
  for (; i + 4 <= m; i += 4) {
    const rayo_i8 *ai = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) block4x4(ai, lda, w + j * ldw, ldw, k, c + i * ldc + j, ldc);
    if (j < n)
      rayo_gemm_i8_neon_dotprod(ai, lda, 4, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
  if (i < m) rayo_gemm_i8_neon_dotprod(a + i * lda, lda, m - i, w, ldw, n, k, c + i * ldc, ldc);
}

rayo_bool rayo_cpu_has_i8mm(void) {
#if defined(__linux__)
  /* HWCAP2_I8MM, with the dot product (HWCAP_ASIMDDP) its fallback uses. */
  return (getauxval(AT_HWCAP2) & (1ul << 13)) != 0 && rayo_cpu_has_dotprod();
#elif defined(__APPLE__)
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &n, NULL, 0) == 0 && v != 0 &&
         rayo_cpu_has_dotprod();
#else
  return RAYO_FALSE;
#endif
}

#else
typedef int rayo_neon_i8mm_unused;
#endif
