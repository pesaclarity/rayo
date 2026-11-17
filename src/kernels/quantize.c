#include <math.h>

#include "kernels/kernels.h"

/* std::round (halves away from zero), exactly, without a libm call: a float
 * plus or minus 0.5 is exact in double, and truncation does the rest. */
static inline int32_t round_half_away(float v) {
  const double d = (double)v;
  return (int32_t)(d >= 0.0 ? d + 0.5 : d - 0.5);
}

void rayo_quantize_rows(const float *restrict x, size_t width, size_t first, size_t step,
                        size_t rows, int8_t *restrict q, size_t ldq, float *scale,
                        int32_t *zero_point) {
  /* Eight independent lanes, so the compiler can use SIMD min and max;
   * min and max do not depend on the order. */
  float lo8[8] = {0}, hi8[8] = {0};
  for (size_t r = first; r < rows; r += step) {
    const float *row = x + r * width;
    size_t i = 0;
    for (; i + 8 <= width; i += 8) {
      for (size_t l = 0; l < 8; l++) {
        lo8[l] = row[i + l] < lo8[l] ? row[i + l] : lo8[l];
        hi8[l] = row[i + l] > hi8[l] ? row[i + l] : hi8[l];
      }
    }
    for (; i < width; i++) {
      lo8[0] = row[i] < lo8[0] ? row[i] : lo8[0];
      hi8[0] = row[i] > hi8[0] ? row[i] : hi8[0];
    }
  }
  float lo = 0.0f, hi = 0.0f;
  for (size_t l = 0; l < 8; l++) {
    lo = lo8[l] < lo ? lo8[l] : lo;
    hi = hi8[l] > hi ? hi8[l] : hi;
  }
  if (lo == hi) {
    for (size_t r = first; r < rows; r += step) {
      for (size_t i = 0; i < width; i++) q[r * ldq + i] = 0;
    }
    *scale = 1.0f;
    *zero_point = 0;
    return;
  }
  /* As TFLite: the scale and zero point in double, then the scale as float. */
  const double s = ((double)hi - (double)lo) / 255.0;
  const double zp_from_min = -128.0 - lo / s;
  const double zp_from_max = 127.0 - hi / s;
  const double err_min = 128.0 + fabs(lo / s);
  const double err_max = 127.0 + fabs(hi / s);
  const double zp_d = err_min < err_max ? zp_from_min : zp_from_max;
  int32_t zp;
  if (zp_d <= -128.0) {
    zp = -128;
  } else if (zp_d >= 127.0) {
    zp = 127;
  } else {
    zp = (int32_t)lround(zp_d);
  }
  const float sf = (float)s;
  const float inv = 1.0f / sf;
  for (size_t r = first; r < rows; r += step) {
    const float *row = x + r * width;
    int8_t *out = q + r * ldq;
    for (size_t i = 0; i < width; i++) {
      float v = (float)zp + row[i] * inv;
      v = v < -128.5f ? -128.5f : (v > 127.5f ? 127.5f : v);
      int32_t iv = round_half_away(v);
      iv = iv < -128 ? -128 : (iv > 127 ? 127 : iv);
      out[i] = (int8_t)iv;
    }
  }
  *scale = sf;
  *zero_point = zp;
}

void rayo_offset_rows(int8_t *restrict q, size_t ldq, size_t rows, size_t width,
                      int32_t *restrict zero, int32_t offset) {
  uint8_t *u = (uint8_t *)(void *)q;
  for (size_t r = 0; r < rows; r++) {
    for (size_t i = 0; i < width; i++) u[r * ldq + i] ^= 0x80u;
    zero[r] += offset;
  }
}
