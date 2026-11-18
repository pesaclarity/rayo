/* IEEE float16 to float32, exactly, for the kernels' group scales: a
 * static function, so each kernel file has its own copy to inline. Every
 * float16 is a float32: normal values move their exponent, subnormal ones
 * are their mantissa times 2^-24, and infinities and NaNs (which
 * rayo-convert never writes) become infinities. */
#ifndef RAYO_KERNELS_HALF_H
#define RAYO_KERNELS_HALF_H

#include <string.h>

#include "rayo/types.h"

static float rayo_half_to_float(rayo_u16 h) {
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
static float rayo_group_scale(const void *scales, rayo_bool half, size_t g) {
  return half ? rayo_half_to_float(((const rayo_u16 *)scales)[g]) : ((const float *)scales)[g];
}

#endif
