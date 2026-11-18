/* IEEE float16 to float32, exactly, for the kernels' group scales: a
 * static function, so each kernel file has its own copy to inline. Every
 * float16 is a float32: normal values move their exponent, subnormal ones
 * are their mantissa times 2^-24, and infinities and NaNs (which
 * rayo-convert never writes) become infinities. */
#ifndef RAYO_KERNELS_HALF_H
#define RAYO_KERNELS_HALF_H

#include <string.h>

#include "rayo/types.h"

/* Always inlined where the compiler can: a SIMD kernel that called them
 * would switch instruction encodings (SSE and AVX) at every group, which
 * costs more than the work. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_HALF_INLINE __inline__ __attribute__((always_inline))
#else
#define RAYO_HALF_INLINE
#endif

static RAYO_HALF_INLINE float rayo_half_to_float(rayo_u16 h) {
  const rayo_u32 sign = (rayo_u32)(h >> 15) << 31;
  const rayo_u32 exp = (rayo_u32)(h >> 10) & 31u, man = (rayo_u32)h & 1023u;
  rayo_u32 bits;
  float f;
  if (exp == 0) {
    f = (float)man * 5.9604644775390625e-8f; /* 2^-24 */
    return sign ? -f : f;
  }
  bits = exp == 31 ? (sign | 0x7f800000u) : (sign | ((exp + 112u) << 23) | (man << 13));
  memcpy(&f, &bits, sizeof f);
  return f;
}

/* Group scale `g` of `scales`: float16 when `half`, else float32. */
static RAYO_HALF_INLINE float rayo_group_scale(const void *scales, rayo_bool half, size_t g) {
  return half ? rayo_half_to_float(((const rayo_u16 *)scales)[g]) : ((const float *)scales)[g];
}

/* The most group scales of one weight row that the kernels convert up
 * front, into a stack buffer (k up to 8192); longer rows convert each
 * group's as it comes. */
#define RAYO_SCALE_ROW 256

/* n (at most RAYO_SCALE_ROW) float16 scales to float32, the same values
 * as rayo_half_to_float: the normal ones and infinities in a loop of
 * integer operations the compiler vectorises, then the rare subnormal
 * ones (and zeros) one by one. */
static RAYO_HALF_INLINE void rayo_half_row(const rayo_u16 *h, size_t n, float *out) {
  size_t i;
  for (i = 0; i < n; i++) {
    const rayo_u32 x = h[i], em = x & 0x7fffu, sign = (x & 0x8000u) << 16;
    const rayo_u32 bits = sign | (em >= 0x7c00u ? 0x7f800000u : (em << 13) + 0x38000000u);
    memcpy(&out[i], &bits, sizeof bits);
  }
  for (i = 0; i < n; i++) {
    if ((h[i] & 0x7c00u) == 0) out[i] = rayo_half_to_float(h[i]);
  }
}

#endif
