/* Unpacking a group of 4-bit or 3-bit weights (kernels.h) into two halves
 * of 16 int8 with NEON: 4 bits, the low and the high four bits minus 8;
 * 3 bits, the two-bit planes with the third bits from their bytes, minus 4.
 * Static, for neon.c and neon_dotprod.c to inline. Include after
 * arm_neon.h. */
#ifndef RAYO_KERNELS_NEON_UNPACK_H
#define RAYO_KERNELS_NEON_UNPACK_H

#include "rayo/types.h"

/* Inlined into each kernel, so each has its own copy for its group size
 * and scale type, with no test in the inner loop. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_NEON_INLINE __inline__ __attribute__((always_inline))
#else
#define RAYO_NEON_INLINE
#endif

static RAYO_NEON_INLINE void neon_unpack_q4(const rayo_u8 *g, int8x16_t *wl, int8x16_t *wh) {
  const uint8x16_t b = vld1q_u8(g);
  const int8x16_t eight = vdupq_n_s8(8);
  *wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(15))), eight);
  *wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);
}

/* The third bits of the 8 values of byte `b`, as 4 or 0. */
static RAYO_NEON_INLINE uint8x8_t third_bits(rayo_u8 b) {
  static const rayo_u8 bits[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return vand_u8(vtst_u8(vdup_n_u8(b), vld1_u8(bits)), vdup_n_u8(4));
}

static RAYO_NEON_INLINE void neon_unpack_q3(const rayo_u8 *g, int8x16_t *wl, int8x16_t *wh) {
  const uint8x8_t b = vld1_u8(g), three = vdup_n_u8(3);
  const int8x16_t four = vdupq_n_s8(4);
  const uint8x16_t lo = vcombine_u8(vorr_u8(vand_u8(b, three), third_bits(g[8])),
                                    vorr_u8(vand_u8(vshr_n_u8(b, 2), three), third_bits(g[9])));
  const uint8x16_t hi = vcombine_u8(vorr_u8(vand_u8(vshr_n_u8(b, 4), three), third_bits(g[10])),
                                    vorr_u8(vshr_n_u8(b, 6), third_bits(g[11])));
  *wl = vsubq_s8(vreinterpretq_s8_u8(lo), four);
  *wh = vsubq_s8(vreinterpretq_s8_u8(hi), four);
}

/* float16 scales to float32 four at a time, exactly as
 * rayo_half_to_float (kernels/half.h, included first). */
static RAYO_NEON_INLINE void neon_half_row(const rayo_u16 *h, size_t n, float *out) {
  const uint32x4_t low15 = vdupq_n_u32(0x7fff), top = vdupq_n_u32(0x8000);
  const uint32x4_t bias = vdupq_n_u32(0x38000000), inf = vdupq_n_u32(0x7f800000);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint32x4_t x = vmovl_u16(vld1_u16(h + i));
    const uint32x4_t em = vandq_u32(x, low15);
    const uint32x4_t sign = vshlq_n_u32(vandq_u32(x, top), 16);
    const uint32x4_t norm = vaddq_u32(vshlq_n_u32(em, 13), bias);
    const uint32x4_t bits = vbslq_u32(vcgtq_u32(em, vdupq_n_u32(0x7bff)), inf, norm);
    const float32x4_t sub = vmulq_n_f32(vcvtq_f32_u32(em), 5.9604644775390625e-8f);
    const float32x4_t f =
        vbslq_f32(vcltq_u32(em, vdupq_n_u32(0x0400)), sub, vreinterpretq_f32_u32(bits));
    vst1q_f32(out + i, vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(f), sign)));
  }
  for (; i < n; i++) out[i] = rayo_half_to_float(h[i]);
}

#endif
