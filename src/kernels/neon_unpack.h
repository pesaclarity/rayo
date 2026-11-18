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

#endif
