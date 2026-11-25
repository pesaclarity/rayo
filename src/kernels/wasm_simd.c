/* Spike: int8 GEMM with WebAssembly SIMD128 (built with -msimd128). Integer
 * sums, so the result equals the generic kernel's exactly. */
#include "kernels/kernels.h"

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>

static rayo_i32 hsum(v128_t v) {
  return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1) +
         wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

static v128_t dot16(v128_t a, v128_t w, v128_t acc) {
  acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a),
                                                 wasm_i16x8_extend_low_i8x16(w)));
  return wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a),
                                                  wasm_i16x8_extend_high_i8x16(w)));
}

void rayo_gemm_i8_wasm(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                       const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                       rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p, k16 = k & ~(size_t)15;
  for (i = 0; i < m; i++) {
    const rayo_i8 *ai = a + i * lda;
    for (j = 0; j + 4 <= n; j += 4) {
      const rayo_i8 *w0 = w + j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
      v128_t s0 = wasm_i32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
      rayo_i32 t0, t1, t2, t3;
      for (p = 0; p < k16; p += 16) {
        v128_t av = wasm_v128_load(ai + p);
        s0 = dot16(av, wasm_v128_load(w0 + p), s0);
        s1 = dot16(av, wasm_v128_load(w1 + p), s1);
        s2 = dot16(av, wasm_v128_load(w2 + p), s2);
        s3 = dot16(av, wasm_v128_load(w3 + p), s3);
      }
      t0 = hsum(s0); t1 = hsum(s1); t2 = hsum(s2); t3 = hsum(s3);
      for (; p < k; p++) {
        rayo_i32 x = ai[p];
        t0 += x * w0[p]; t1 += x * w1[p]; t2 += x * w2[p]; t3 += x * w3[p];
      }
      c[i * ldc + j] = t0; c[i * ldc + j + 1] = t1; c[i * ldc + j + 2] = t2; c[i * ldc + j + 3] = t3;
    }
    if (j < n) rayo_gemm_i8_generic(ai, lda, 1, w + j * ldw, ldw, n - j, k, c + i * ldc + j, ldc);
  }
}
#else
typedef int rayo_wasm_simd_unused;
#endif
