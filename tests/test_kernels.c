/* kernels: quantisation as TFLite does it, and every SIMD product this
 * build and CPU have against the generic one on random data. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/kernels.h"
#include "test.h"

static rayo_u32 state = 12345u;

static rayo_u32 next(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static float uniform(void) {
  return (float)(next() % 20001u) / 10000.0f - 1.0f;
}

void test_quantize(void) {
  float x[6], z[3], h[2], rows[8];
  rayo_i8 q[8];
  float scale;
  rayo_i32 zero;
  int i;
  x[0] = -1.0f;
  x[1] = 0.5f;
  x[2] = 2.0f;
  x[3] = 0.0f;
  x[4] = 1.0f;
  x[5] = -0.25f;
  rayo_quantize_rows(x, 6, 0, 1, 1, q, 6, &scale, &zero);
  /* Range [-1, 2]: scale 3/255; the zero point from the minimum is -43. */
  EXPECT(fabs(scale - 3.0 / 255.0) < 1e-9);
  EXPECT(zero == -43);
  for (i = 0; i < 6; i++) EXPECT(fabs((double)(q[i] - zero) * scale - x[i]) <= scale / 2 + 1e-7);
  /* All zero: zeros, scale 1. */
  for (i = 0; i < 3; i++) z[i] = 0.0f;
  rayo_quantize_rows(z, 3, 0, 1, 1, q, 3, &scale, &zero);
  EXPECT(q[0] == 0 && q[2] == 0 && scale == 1.0f && zero == 0);
  /* [0, 127]: zero point -128 and the top at 127. */
  h[0] = 0.0f;
  h[1] = 127.0f;
  rayo_quantize_rows(h, 2, 0, 1, 1, q, 2, &scale, &zero);
  EXPECT(zero == -128 && q[1] == 127);
  /* A group of every other row, quantised together. */
  for (i = 0; i < 8; i++) rows[i] = (i / 2) % 2 ? 9.0f : (float)(1 + 2 * (i / 4));
  rayo_quantize_rows(rows, 2, 0, 2, 4, q, 2, &scale, &zero);
  EXPECT(fabs(scale - 3.0 / 255.0) < 1e-9);
}

static void check_i8(rayo_gemm_i8_fn *fn, rayo_i32 a_offset, const char *name) {
  enum { M = 7, N = 9, K = 96 };
  rayo_i8 *a = malloc(M * K), *au = malloc(M * K), *w = malloc(N * K);
  rayo_i32 *want = malloc(M * N * sizeof(rayo_i32)), *got = malloc(M * N * sizeof(rayo_i32));
  rayo_i32 rowsum[N];
  int same = 1;
  size_t i, j, p;
  if (!a || !au || !w || !want || !got) {
    EXPECT(0);
    return;
  }
  for (i = 0; i < M * K; i++) a[i] = (rayo_i8)((int)(next() % 256u) - 128);
  for (i = 0; i < N * K; i++) w[i] = (rayo_i8)((int)(next() % 255u) - 127);
  for (j = 0; j < N; j++) {
    rowsum[j] = 0;
    for (p = 0; p < K; p++) rowsum[j] += w[j * K + p];
  }
  for (i = 0; i < M * K; i++) au[i] = (rayo_i8)(int)((rayo_u8)a[i] ^ 0x80u);
  rayo_gemm_i8_generic(a, K, M, w, K, N, K, want, N);
  fn(a_offset ? au : a, K, M, w, K, N, K, got, N);
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) same &= got[i * N + j] - a_offset * rowsum[j] == want[i * N + j];
  }
  printf("  int8 kernel %s: %s\n", name, same ? "same as generic" : "DIFFERS from generic");
  EXPECT(same);
  free(a);
  free(au);
  free(w);
  free(want);
  free(got);
}

/* A Q4 product against double maths on the dequantised values, and the
 * kernel against the generic one, bit for bit. */
static void check_q4(rayo_gemm_q4_fn *fn, const char *name) {
  enum { M = 3, N = 7, K = 96, G = K / RAYO_Q4_GROUP };
  float x[M * K], ascale[M * G], wscale[N * G], want[M * N], got[M * N];
  rayo_i8 q[M * K];
  rayo_u8 w[N * K / 2];
  int same = 1, close = 1;
  size_t i, j, p;
  for (i = 0; i < M * K; i++) x[i] = uniform();
  x[5] = 0.0f;
  for (i = 0; i < N * K / 2; i++) w[i] = (rayo_u8)(next() % 256u);
  for (i = 0; i < N * G; i++) wscale[i] = 0.01f + 0.1f * (uniform() + 1.0f);
  rayo_quantize_groups(x, M, K, q, ascale);
  rayo_gemm_q4_generic(q, ascale, M, w, wscale, N, K, want, N);
  fn(q, ascale, M, w, wscale, N, K, got, N);
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      double ref = 0.0;
      for (p = 0; p < K; p++) {
        const rayo_u8 b = w[j * K / 2 + p / 32 * 16 + p % 16];
        const int v = (p % 32 < 16 ? b & 15 : b >> 4) - 8;
        ref += (double)q[i * K + p] * ascale[i * G + p / 32] * v * wscale[j * G + p / 32];
      }
      close &= fabs(ref - want[i * N + j]) < 1e-4;
#if defined(__arm__)
      /* armv7 NEON flushes denormals: allow the last bits. */
      same &= fabs(got[i * N + j] - want[i * N + j]) <= 1e-6 * (1.0 + fabs(want[i * N + j]));
#else
      same &= got[i * N + j] == want[i * N + j];
#endif
    }
  }
  /* The quantised activations are within half a step of x. */
  for (i = 0; i < M * K; i++)
    close &= fabs(q[i] * ascale[i / 32] - x[i]) <= ascale[i / 32] * 0.5001;
  EXPECT(ascale[0] > 0.0f);
  printf("  q4 kernel %s: %s\n", name, same ? "same as generic" : "DIFFERS from generic");
  EXPECT(close);
  EXPECT(same);
}

void test_kernels(void) {
  rayo_kernels best, simd;
  float a[3 * 20], w[4 * 20], c[12];
  double want = 0;
  int i;
  rayo_kernels_best(&best);
  rayo_kernels_simd(&simd);
  printf("  kernels: build %s, best here %s\n", simd.name, best.name);
  check_i8(simd.gemm_i8, simd.a_offset, simd.name);
  check_i8(best.gemm_i8, best.a_offset, best.name);
  check_q4(simd.gemm_q4, simd.name);
  check_q4(best.gemm_q4, best.name);
#if defined(RAYO_KERNELS_X86_VNNI)
  if (!rayo_cpu_has_avxvnni()) printf("  avx-vnni: this CPU has none, not run\n");
#endif
#if defined(RAYO_KERNELS_NEON_DOTPROD)
  if (!rayo_cpu_has_dotprod()) printf("  neon-dotprod: this CPU has none, not run\n");
#endif
  for (i = 0; i < 60; i++) a[i] = uniform();
  for (i = 0; i < 80; i++) w[i] = uniform();
  rayo_gemm_f32_generic(a, 20, 3, w, 20, 4, 20, c, 4);
  for (i = 0; i < 20; i++) want += (double)a[20 + i] * w[40 + i];
  EXPECT(fabs(c[1 * 4 + 2] - want) < 1e-5);
}
