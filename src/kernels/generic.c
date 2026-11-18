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

/* One output: row `a` of the activations by row `w` of the weights. */
static float dot_q4(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT as,
                    const rayo_u8 *RAYO_RESTRICT w, const float *RAYO_RESTRICT ws, size_t groups) {
  float lanes[8];
  size_t g, l, e;
  for (l = 0; l < 8; l++) lanes[l] = 0.0f;
  for (g = 0; g < groups; g++) {
    const rayo_i8 *ag = a + g * RAYO_Q4_GROUP;
    const rayo_u8 *wg = w + g * (RAYO_Q4_GROUP / 2);
    const float d = as[g] * ws[g];
    for (l = 0; l < 8; l++) {
      rayo_i32 sum = 0;
      float t;
      for (e = 4 * l; e < 4 * l + 4; e++) {
        const rayo_i32 v = e < 16 ? (rayo_i32)(wg[e] & 15u) - 8 : (rayo_i32)(wg[e - 16] >> 4) - 8;
        sum += (rayo_i32)ag[e] * v;
      }
      /* Two statements, so no compiler fuses them into one rounding. */
      t = (float)sum * d;
      lanes[l] += t;
    }
  }
  return rayo_q4_lanes_sum(lanes);
}

void rayo_gemm_q4_generic(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                          size_t m, const rayo_u8 *RAYO_RESTRICT w,
                          const float *RAYO_RESTRICT wscale, size_t n, size_t k,
                          float *RAYO_RESTRICT c, size_t ldc) {
  const size_t groups = k / RAYO_Q4_GROUP;
  size_t i, j;
  for (j = 0; j < n; j++) {
    const rayo_u8 *wj = w + j * (k / 2);
    const float *sj = wscale + j * groups;
    for (i = 0; i < m; i++) c[i * ldc + j] = dot_q4(a + i * k, ascale + i * groups, wj, sj, groups);
  }
}
