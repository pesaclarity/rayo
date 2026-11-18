#include "kernels/half.h"
#include "kernels/kernels.h"

void rayo_gemm_i8_generic(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                          const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                          rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p;
  for (i = 0; i < m; i++) {
    const rayo_i8 *ai = a + i * lda;
    for (j = 0; j < n; j++) {
      const rayo_i8 *wj = w + j * ldw;
      rayo_i32 acc = 0;
      for (p = 0; p < k; p++) acc += (rayo_i32)ai[p] * (rayo_i32)wj[p];
      c[i * ldc + j] = acc;
    }
  }
}

/* Eight partial sums per output, added in a fixed order at the end: the
 * compiler vectorises it, and every build rounds the same way. */
void rayo_gemm_f32_generic(const float *RAYO_RESTRICT a, size_t lda, size_t m,
                           const float *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                           float *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p, l;
  for (i = 0; i < m; i++) {
    const float *ai = a + i * lda;
    for (j = 0; j < n; j++) {
      const float *wj = w + j * ldw;
      float lanes[8];
      float acc;
      for (l = 0; l < 8; l++) lanes[l] = 0.0f;
      for (p = 0; p + 8 <= k; p += 8) {
        for (l = 0; l < 8; l++) lanes[l] += ai[p + l] * wj[p + l];
      }
      acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
            ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
      for (; p < k; p++) acc += ai[p] * wj[p];
      c[i * ldc + j] = acc;
    }
  }
}

float rayo_q4_lanes_sum(const float *lanes) {
  return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
         ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
}

/* Value e (0 to 31) of a group of 4-bit or 3-bit weights, -8 to 7 or -4 to
 * 3. */
static rayo_i32 q4_value(const rayo_u8 *g, size_t e) {
  return e < 16 ? (rayo_i32)(g[e] & 15u) - 8 : (rayo_i32)(g[e - 16] >> 4) - 8;
}

static rayo_i32 q3_value(const rayo_u8 *g, size_t e) {
  const rayo_u32 low = ((rayo_u32)g[e % 8] >> (2 * (e / 8))) & 3u;
  const rayo_u32 high = ((rayo_u32)g[8 + e / 8] >> (e % 8)) & 1u;
  return (rayo_i32)(low | high << 2) - 4;
}

/* One output: row `a` of the activations by row `w` of the weights, groups
 * of `bytes` bytes. */
static float dot_q(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT as,
                   const rayo_u8 *RAYO_RESTRICT w, const void *ws, rayo_bool half, size_t first,
                   size_t groups, size_t bytes) {
  float lanes[8];
  size_t g, l, e;
  for (l = 0; l < 8; l++) lanes[l] = 0.0f;
  for (g = 0; g < groups; g++) {
    const rayo_i8 *ag = a + g * RAYO_Q4_GROUP;
    const rayo_u8 *wg = w + g * bytes;
    const float d = as[g] * rayo_group_scale(ws, half, first + g);
    for (l = 0; l < 8; l++) {
      rayo_i32 sum = 0;
      float t;
      for (e = 4 * l; e < 4 * l + 4; e++) {
        const rayo_i32 v = bytes == RAYO_Q4_GROUP_BYTES ? q4_value(wg, e) : q3_value(wg, e);
        sum += (rayo_i32)ag[e] * v;
      }
      /* Two statements, so no compiler fuses them into one rounding. */
      t = (float)sum * d;
      lanes[l] += t;
    }
  }
  return rayo_q4_lanes_sum(lanes);
}

static void gemm_q(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale, size_t m,
                   const rayo_u8 *RAYO_RESTRICT w, const void *wscale, rayo_bool half, size_t n,
                   size_t k, float *RAYO_RESTRICT c, size_t ldc, size_t bytes) {
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

void rayo_gemm_q4_generic(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                          size_t m, const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                          rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c, size_t ldc) {
  gemm_q(a, ascale, m, w, wscale, half, n, k, c, ldc, RAYO_Q4_GROUP_BYTES);
}

void rayo_gemm_q3_generic(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                          size_t m, const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                          rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c, size_t ldc) {
  gemm_q(a, ascale, m, w, wscale, half, n, k, c, ldc, RAYO_Q3_GROUP_BYTES);
}
