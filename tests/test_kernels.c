/* kernels: quantisation as TFLite does it, and every SIMD product against
 * the generic one on random data. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/kernels.h"
#include "test.h"

static uint32_t state = 12345u;

static uint32_t next(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static float uniform(void) {
  return (float)(next() % 20001u) / 10000.0f - 1.0f;
}

void test_quantize(void) {
  float x[6] = {-1.0f, 0.5f, 2.0f, 0.0f, 1.0f, -0.25f};
  int8_t q[6];
  float scale;
  int32_t zero;
  rayo_quantize_rows(x, 6, 0, 1, 1, q, 6, &scale, &zero);
  /* Range [-1, 2]: scale 3/255; the zero point from the minimum is -43. */
  EXPECT(fabsf(scale - 3.0f / 255.0f) < 1e-9f);
  EXPECT(zero == -43);
  for (int i = 0; i < 6; i++)
    EXPECT(fabsf((float)(q[i] - zero) * scale - x[i]) <= scale / 2 + 1e-7f);
  /* All zero: zeros, scale 1. */
  float z[3] = {0};
  rayo_quantize_rows(z, 3, 0, 1, 1, q, 3, &scale, &zero);
  EXPECT(q[0] == 0 && q[2] == 0 && scale == 1.0f && zero == 0);
  /* Halves round away from zero, like std::round: 0.5 / scale exactly. */
  float h[2] = {0.0f, 127.0f};
  rayo_quantize_rows(h, 2, 0, 1, 1, q, 2, &scale, &zero);
  EXPECT(zero == -128 && q[1] == 127);
  /* A group of every other row, quantised together. */
  float rows[8] = {1, 1, 9, 9, 3, 3, 9, 9};
  rayo_quantize_rows(rows, 2, 0, 2, 4, q, 2, &scale, &zero);
  EXPECT(fabsf(scale - 3.0f / 255.0f) < 1e-9f);
}

static void check_i8(rayo_gemm_i8_fn *fn, int32_t a_offset, const char *name) {
  const size_t m = 7, n = 9, k = 96;
  int8_t *a = malloc(m * k), *au = malloc(m * k), *w = malloc(n * k);
  int32_t *want = malloc(m * n * 4), *got = malloc(m * n * 4);
  int32_t rowsum[9] = {0};
  for (size_t i = 0; i < m * k; i++) a[i] = (int8_t)((int)(next() % 256u) - 128);
  for (size_t i = 0; i < n * k; i++) w[i] = (int8_t)((int)(next() % 255u) - 127);
  for (size_t j = 0; j < n; j++) {
    for (size_t p = 0; p < k; p++) rowsum[j] += w[j * k + p];
  }
  for (size_t i = 0; i < m * k; i++) au[i] = (int8_t)((uint8_t)a[i] ^ 0x80u);
  rayo_gemm_i8_generic(a, k, m, w, k, n, k, want, n);
  fn(a_offset ? au : a, k, m, w, k, n, k, got, n);
  int same = 1;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) same &= got[i * n + j] - a_offset * rowsum[j] == want[i * n + j];
  }
  if (!same) fprintf(stderr, "kernel %s differs from generic\n", name);
  EXPECT(same);
  free(a);
  free(au);
  free(w);
  free(want);
  free(got);
}

void test_kernels(void) {
  rayo_kernels best;
  rayo_kernels_best(&best);
  printf("  kernels here: %s\n", best.name);
  check_i8(best.gemm_i8, best.a_offset, best.name);
#if defined(RAYO_KERNELS_X86) && (defined(__AVX2__) || defined(__SSE4_1__))
  check_i8(rayo_gemm_i8_x86, 0, "x86");
#endif
#if defined(RAYO_KERNELS_NEON)
  check_i8(rayo_gemm_i8_neon, 0, "neon");
#endif
  float a[3 * 20], w[4 * 20], c[12];
  for (int i = 0; i < 60; i++) a[i] = uniform();
  for (int i = 0; i < 80; i++) w[i] = uniform();
  rayo_gemm_f32_generic(a, 20, 3, w, 20, 4, 20, c, 4);
  double want = 0;
  for (int p = 0; p < 20; p++) want += (double)a[20 + p] * w[40 + p];
  EXPECT(fabs(c[1 * 4 + 2] - want) < 1e-5);
}
