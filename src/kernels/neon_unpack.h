/* Unpacking a group of 3-bit weights (kernels.h) into two halves of 16
 * int8 with NEON, the two-bit planes with the third bits from their bytes,
 * and float16 scales to float32. Static, for neon.c and neon_dotprod.c to
 * inline (the 4-bit groups: neon_q.h). Include after arm_neon.h. */
#ifndef RAYO_KERNELS_NEON_UNPACK_H
#define RAYO_KERNELS_NEON_UNPACK_H

#include "rayo/types.h"

/* Inlined into each kernel where a call would cost; kept out of line
 * where a copy for each group size would only make the library larger. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_NEON_INLINE __inline__ __attribute__((always_inline))
#define RAYO_NEON_NOINLINE __attribute__((noinline))
#else
#define RAYO_NEON_INLINE
#define RAYO_NEON_NOINLINE
#endif

/* The third bits of the 8 values of byte `b`, as 4 or 0. */
static RAYO_NEON_INLINE uint8x8_t third_bits(rayo_u8 b) {
  static const rayo_u8 bits[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return vand_u8(vtst_u8(vdup_n_u8(b), vld1_u8(bits)), vdup_n_u8(4));
}

/* A group of 3-bit weights as stored, 0 to 7. */
static RAYO_NEON_INLINE void neon_unpack_q3_stored(const rayo_u8 *g, int8x16_t *wl, int8x16_t *wh) {
  const uint8x8_t b = vld1_u8(g), three = vdup_n_u8(3);
  const uint8x16_t lo = vcombine_u8(vorr_u8(vand_u8(b, three), third_bits(g[8])),
                                    vorr_u8(vand_u8(vshr_n_u8(b, 2), three), third_bits(g[9])));
  const uint8x16_t hi = vcombine_u8(vorr_u8(vand_u8(vshr_n_u8(b, 4), three), third_bits(g[10])),
                                    vorr_u8(vshr_n_u8(b, 6), third_bits(g[11])));
  *wl = vreinterpretq_s8_u8(lo);
  *wh = vreinterpretq_s8_u8(hi);
}

/* float16 scales to float32 four at a time, exactly as rayo_half_to_float
 * (kernels/half.h, included first), out of line (it runs once a row of
 * weights): on arm64 with the conversion instruction (exact for every
 * float16; infinities and NaNs, which rayo-convert never writes, made
 * infinities as the generic kernel makes them), on armv7 with integer
 * operations. */
static RAYO_NEON_NOINLINE void neon_half_row(const rayo_u16 *h, size_t n, float *out) {
  const uint32x4_t top = vdupq_n_u32(0x8000), inf = vdupq_n_u32(0x7f800000);
#if !defined(__aarch64__)
  const uint32x4_t low15 = vdupq_n_u32(0x7fff), bias = vdupq_n_u32(0x38000000);
#endif
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint16x4_t x16 = vld1_u16(h + i);
    const uint32x4_t x = vmovl_u16(x16);
    const uint32x4_t sign = vshlq_n_u32(vandq_u32(x, top), 16);
#if defined(__aarch64__)
    const uint32x4_t big = vcgtq_u32(vandq_u32(x, vdupq_n_u32(0x7fff)), vdupq_n_u32(0x7bff));
    const float32x4_t f = vcvt_f32_f16(vreinterpret_f16_u16(x16));
    vst1q_f32(out + i, vbslq_f32(big, vreinterpretq_f32_u32(vorrq_u32(sign, inf)), f));
#else
    const uint32x4_t em = vandq_u32(x, low15);
    const uint32x4_t norm = vaddq_u32(vshlq_n_u32(em, 13), bias);
    const uint32x4_t bits = vbslq_u32(vcgtq_u32(em, vdupq_n_u32(0x7bff)), inf, norm);
    const float32x4_t sub = vmulq_n_f32(vcvtq_f32_u32(em), 5.9604644775390625e-8f);
    const float32x4_t f =
        vbslq_f32(vcltq_u32(em, vdupq_n_u32(0x0400)), sub, vreinterpretq_f32_u32(bits));
    vst1q_f32(out + i, vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(f), sign)));
#endif
  }
  for (; i < n; i++) out[i] = rayo_half_to_float(h[i]);
}

#endif
