/* kernels: quantisation as TFLite does it, and every SIMD product this
 * build and CPU have against the generic one on random data. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/half.h"
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

/* One shape of an int8 product: the kernel against the generic one. */
static int same_i8(rayo_gemm_i8_fn *fn, rayo_i32 a_offset, size_t m, size_t n, size_t k) {
  enum { MMAX = 9, NMAX = 13, KMAX = 160 };
  static rayo_i8 a[MMAX * KMAX], au[MMAX * KMAX], w[NMAX * KMAX];
  static rayo_i32 want[MMAX * NMAX], got[MMAX * NMAX], rowsum[NMAX];
  int same = 1;
  size_t i, j, p;
  for (i = 0; i < m * k; i++) a[i] = (rayo_i8)((int)(next() % 256u) - 128);
  for (i = 0; i < n * k; i++) w[i] = (rayo_i8)((int)(next() % 255u) - 127);
  /* The extremes, where a sum is largest. */
  a[0] = -128;
  w[0] = -127;
  for (j = 0; j < n; j++) {
    rowsum[j] = 0;
    for (p = 0; p < k; p++) rowsum[j] += w[j * k + p];
  }
  for (i = 0; i < m * k; i++) au[i] = (rayo_i8)(int)((rayo_u8)a[i] ^ 0x80u);
  rayo_gemm_i8_generic(a, k, m, w, k, n, k, want, n);
  fn(a_offset ? au : a, k, m, w, k, n, k, got, n);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) same &= got[i * n + j] - a_offset * rowsum[j] == want[i * n + j];
  }
  return same;
}

/* Shapes with the blocks' leftovers (one row, odd rows, columns past a
 * multiple of four and of eight). */
static void check_i8(rayo_gemm_i8_fn *fn, rayo_i32 a_offset, const char *name) {
  static const size_t ms[] = {1, 2, 4, 7, 9}, ns[] = {1, 4, 9, 13}, ks[] = {32, 96, 160};
  int same = 1;
  size_t x, y, z;
  for (x = 0; x < sizeof ms / sizeof ms[0]; x++) {
    for (y = 0; y < sizeof ns / sizeof ns[0]; y++) {
      for (z = 0; z < sizeof ks / sizeof ks[0]; z++)
        same &= same_i8(fn, a_offset, ms[x], ns[y], ks[z]);
    }
  }
  printf("  int8 kernel %s: %s\n", name, same ? "same as generic" : "DIFFERS from generic");
  EXPECT(same);
}

/* Value e of a group of `bits` (4 or 3) weights, as docs/FORMAT.md
 * describes them. */
static int group_value(const rayo_u8 *g, int bits, size_t e) {
  if (bits == 4) return (e < 16 ? g[e] & 15 : g[e - 16] >> 4) - 8;
  return (int)((g[e % 8] >> (2 * (e / 8))) & 3) + 4 * ((g[8 + e / 8] >> (e % 8)) & 1) - 4;
}

/* float16 to double, from the definition. */
static double half_value(rayo_u16 h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const double v = e == 0 ? ldexp((double)m, -24) : ldexp(1024.0 + m, e - 25);
  return (h >> 15) ? -v : v;
}

/* A product of group-wise weights (`bits` 4 or 3, scales float32 or, with
 * `half`, float16) against double maths on the dequantised values, and the
 * kernel against the generic one, bit for bit. */
static int check_q(rayo_gemm_q4_fn *fn, int bits, rayo_bool half, size_t K, size_t M, size_t N) {
  enum { MMAX = 5, NMAX = 9, KMAX = 608, GMAX = KMAX / RAYO_Q4_GROUP };
  const size_t gb = bits == 4 ? RAYO_Q4_GROUP_BYTES : RAYO_Q3_GROUP_BYTES;
  const size_t G = K / RAYO_Q4_GROUP;
  static float x[MMAX * KMAX], ascale[MMAX * GMAX], wscale[NMAX * GMAX], want[MMAX * NMAX],
      got[MMAX * NMAX];
  static rayo_u16 wscale16[NMAX * GMAX];
  static rayo_i8 q[MMAX * KMAX];
  static rayo_u8 w[NMAX * GMAX * RAYO_Q4_GROUP_BYTES];
  const void *ws = half ? (const void *)wscale16 : (const void *)wscale;
  rayo_gemm_q4_fn *generic = bits == 4 ? rayo_gemm_q4_generic : rayo_gemm_q3_generic;
  int same = 1, close = 1;
  size_t i, j, p;
  for (i = 0; i < M * K; i++) x[i] = uniform();
  x[5] = 0.0f;
  for (i = 0; i < sizeof w; i++) w[i] = (rayo_u8)(next() % 256u);
  for (i = 0; i < N * G; i++) {
    wscale[i] = 0.01f + 0.1f * (uniform() + 1.0f);
    /* Normal float16 values of 2^-10 to 2^-1, either sign, and a subnormal. */
    wscale16[i] = (rayo_u16)((next() % 2u) << 15 | (5u + next() % 9u) << 10 | next() % 1024u);
  }
  wscale16[3] = 0x0123; /* subnormal ones, in a block of eight and in the tail */
  wscale16[N * G - 1] = 0x8007;
  wscale16[10 % (N * G)] = 0x0000;
  rayo_quantize_groups(x, M, K, q, ascale);
  generic(q, ascale, M, w, ws, half, N, K, want, N);
  fn(q, ascale, M, w, ws, half, N, K, got, N);
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      double ref = 0.0;
      for (p = 0; p < K; p++) {
        const size_t g = j * G + p / 32;
        const int v = group_value(w + g * gb, bits, p % 32);
        const double sc = half ? half_value(wscale16[g]) : (double)wscale[g];
        ref += (double)q[i * K + p] * ascale[i * G + p / 32] * v * sc;
      }
      close &= fabs(ref - want[i * N + j]) < 1e-4 * (1.0 + fabs(ref));
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
  EXPECT(close);
  return same;
}

/* M and N with the blocks' leftovers: one row (a decoder's), two, an odd
 * count; rows of weights past a multiple of four. */
static void check_q4(rayo_gemm_q4_fn *q4, rayo_gemm_q4_fn *q3, const char *name) {
  static const size_t shapes[][2] = {{3, 7}, {1, 9}, {2, 4}, {5, 8}};
  int bits, half;
  for (bits = 4; bits >= 3; bits--) {
    for (half = 0; half < 2; half++) {
      int same = 1;
      size_t k, s;
      for (k = 96; k <= 608; k += 512) {
        for (s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
          same &=
              check_q(bits == 4 ? q4 : q3, bits, (rayo_bool)half, k, shapes[s][0], shapes[s][1]);
        }
      }
      printf("  q%d kernel %s, %s scales: %s\n", bits, name, half ? "float16" : "float32",
             same ? "same as generic" : "DIFFERS from generic");
      EXPECT(same);
    }
  }
}

/* Every float16 but infinities and NaNs converts exactly. */
static void check_half(void) {
  const rayo_u16 one = 0x3c00;
  const float two = 2.0f;
  rayo_u32 h;
  int exact = 1;
  for (h = 0; h < 65536u; h++) {
    if (((h >> 10) & 31u) == 31u) continue;
    exact &= (double)rayo_half_to_float((rayo_u16)h) == half_value((rayo_u16)h);
  }
  EXPECT(exact);
  EXPECT(rayo_group_scale(&one, RAYO_TRUE, 0) == 1.0f);
  EXPECT(rayo_group_scale(&two, RAYO_FALSE, 0) == 2.0f);
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
  check_half();
  check_q4(simd.gemm_q4, simd.gemm_q3, simd.name);
  check_q4(best.gemm_q4, best.gemm_q3, best.name);
#if defined(RAYO_KERNELS_X86_VNNI)
  /* Each x86 set this CPU runs, whatever the build targets. */
  if (rayo_cpu_has_sse41()) {
    check_i8(rayo_gemm_i8_sse41, 0, "sse4.1 (run time)");
    check_q4(rayo_gemm_q4_sse41, rayo_gemm_q3_sse41, "sse4.1 (run time)");
  }
  if (rayo_cpu_has_avx2()) {
    check_i8(rayo_gemm_i8_avx2, 0, "avx2 (run time)");
    check_q4(rayo_gemm_q4_avx2, rayo_gemm_q3_avx2, "avx2 (run time)");
  }
#endif
#if defined(RAYO_KERNELS_X86_VNNI)
  if (!rayo_cpu_has_avxvnni()) printf("  avx-vnni: this CPU has none, not run\n");
#endif
#if defined(RAYO_KERNELS_NEON_DOTPROD)
  /* Each ARM set this CPU runs, whichever is best. */
  if (rayo_cpu_has_dotprod()) {
    check_i8(rayo_gemm_i8_neon_dotprod, 0, "neon-dotprod (run time)");
    check_q4(rayo_gemm_q4_neon_dotprod, rayo_gemm_q3_neon_dotprod, "neon-dotprod (run time)");
  } else {
    printf("  neon-dotprod: this CPU has none, not run\n");
  }
#endif
#if defined(RAYO_KERNELS_NEON_I8MM)
  if (rayo_cpu_has_i8mm()) {
    check_i8(rayo_gemm_i8_neon_i8mm, 0, "neon-i8mm (run time)");
  } else {
    printf("  neon-i8mm: this CPU has none, not run\n");
  }
#endif
  for (i = 0; i < 60; i++) a[i] = uniform();
  for (i = 0; i < 80; i++) w[i] = uniform();
  rayo_gemm_f32_generic(a, 20, 3, w, 20, 4, 20, c, 4);
  for (i = 0; i < 20; i++) want += (double)a[20 + i] * w[40 + i];
  EXPECT(fabs(c[1 * 4 + 2] - want) < 1e-5);
}
