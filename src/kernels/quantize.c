#include <math.h>

#include "kernels/kernels.h"

/* std::round (halves away from zero), exactly: a float plus or minus 0.5 is
 * exact in double, and truncation does the rest. */
static rayo_i32 round_half_away(float v) {
  const double d = (double)v;
  return (rayo_i32)(d >= 0.0 ? d + 0.5 : d - 0.5);
}

/* The same for a double that is not near the int32 limits. */
static rayo_i32 round_double(double d) {
  return (rayo_i32)(d >= 0.0 ? floor(d + 0.5) : ceil(d - 0.5));
}

static void range_of(const float *RAYO_RESTRICT x, size_t width, size_t first, size_t step,
                     size_t rows, float *lo_out, float *hi_out) {
  /* Eight independent lanes, so the compiler can use SIMD min and max; min
   * and max do not depend on the order. */
  float lo8[8], hi8[8], lo = 0.0f, hi = 0.0f;
  size_t r, i, l;
  for (l = 0; l < 8; l++) lo8[l] = hi8[l] = 0.0f;
  for (r = first; r < rows; r += step) {
    const float *row = x + r * width;
    for (i = 0; i + 8 <= width; i += 8) {
      for (l = 0; l < 8; l++) {
        lo8[l] = row[i + l] < lo8[l] ? row[i + l] : lo8[l];
        hi8[l] = row[i + l] > hi8[l] ? row[i + l] : hi8[l];
      }
    }
    for (; i < width; i++) {
      lo8[0] = row[i] < lo8[0] ? row[i] : lo8[0];
      hi8[0] = row[i] > hi8[0] ? row[i] : hi8[0];
    }
  }
  for (l = 0; l < 8; l++) {
    lo = lo8[l] < lo ? lo8[l] : lo;
    hi = hi8[l] > hi ? hi8[l] : hi;
  }
  *lo_out = lo;
  *hi_out = hi;
}

void rayo_quantize_rows(const float *RAYO_RESTRICT x, size_t width, size_t first, size_t step,
                        size_t rows, rayo_i8 *RAYO_RESTRICT q, size_t ldq, float *scale,
                        rayo_i32 *zero_point) {
  float lo, hi, sf, inv;
  double s, zp_from_min, zp_from_max, err_min, err_max, zp_d;
  rayo_i32 zp;
  size_t r, i;
  range_of(x, width, first, step, rows, &lo, &hi);
  /* A NaN or infinity (only a damaged model makes one) quantises to zeros,
   * as no value is: converting it to an integer would be undefined. */
  if (lo == hi || !(hi - lo <= 3.0e38f)) {
    for (r = first; r < rows; r += step) {
      for (i = 0; i < width; i++) q[r * ldq + i] = 0;
    }
    *scale = 1.0f;
    *zero_point = 0;
    return;
  }
  /* As TFLite: the scale and zero point in double, then the scale as float. */
  s = ((double)hi - (double)lo) / 255.0;
  zp_from_min = -128.0 - lo / s;
  zp_from_max = 127.0 - hi / s;
  err_min = 128.0 + fabs(lo / s);
  err_max = 127.0 + fabs(hi / s);
  zp_d = err_min < err_max ? zp_from_min : zp_from_max;
  if (zp_d <= -128.0) {
    zp = -128;
  } else if (zp_d >= 127.0) {
    zp = 127;
  } else {
    zp = round_double(zp_d);
  }
  sf = (float)s;
  inv = 1.0f / sf;
  for (r = first; r < rows; r += step) {
    const float *row = x + r * width;
    rayo_i8 *out = q + r * ldq;
    for (i = 0; i < width; i++) {
      float v = (float)zp + row[i] * inv;
      rayo_i32 iv;
      /* Written so that a NaN clamps too. */
      v = v >= -128.5f ? v : -128.5f;
      v = v <= 127.5f ? v : 127.5f;
      iv = round_half_away(v);
      iv = iv < -128 ? -128 : (iv > 127 ? 127 : iv);
      out[i] = (rayo_i8)iv;
    }
  }
  *scale = sf;
  *zero_point = zp;
}

void rayo_offset_rows(rayo_i8 *RAYO_RESTRICT q, size_t ldq, size_t rows, size_t width,
                      rayo_i32 *RAYO_RESTRICT zero, rayo_i32 offset) {
  rayo_u8 *u = (rayo_u8 *)(void *)q;
  size_t r, i;
  for (r = 0; r < rows; r++) {
    for (i = 0; i < width; i++) u[r * ldq + i] ^= 0x80u;
    zero[r] += offset;
  }
}
