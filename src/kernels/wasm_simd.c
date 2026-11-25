/* int8 by int8 with WebAssembly SIMD128, for builds with -msimd128 (wasi-sdk
 * or emscripten). Sixteen products a step, widened to int16 and summed in
 * pairs into int32 (i32x4.dot_i16x8_s), four rows of `w` at a time: integer
 * sums, so the result is the generic kernel's exactly. The 4-bit and float
 * products stay the generic ones (which clang autovectorises with
 * -msimd128). */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_WASM_SIMD)
#include <wasm_simd128.h>

static rayo_i32 hsum(v128_t v) {
  return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1) +
         wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

/* acc + the sixteen products of a and w, in four int32 lanes. */
static v128_t dot16(v128_t a, v128_t w, v128_t acc) {
  acc = wasm_i32x4_add(
      acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a), wasm_i16x8_extend_low_i8x16(w)));
  return wasm_i32x4_add(
      acc, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a), wasm_i16x8_extend_high_i8x16(w)));
}

void rayo_gemm_i8_wasm_simd128(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                               const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                               rayo_i32 *RAYO_RESTRICT c, size_t ldc) {
  size_t i, j, p;
  for (i = 0; i < m; i++) {
    const rayo_i8 *ai = a + i * lda;
    rayo_i32 *ci = c + i * ldc;
    for (j = 0; j + 4 <= n; j += 4) {
      const rayo_i8 *w0 = w + j * ldw;
      const rayo_i8 *w1 = w0 + ldw;
      const rayo_i8 *w2 = w1 + ldw;
      const rayo_i8 *w3 = w2 + ldw;
      v128_t s0 = wasm_i32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
      /* k is a multiple of RAYO_K_ALIGN (32): no tail. */
      for (p = 0; p < k; p += 16) {
        v128_t av = wasm_v128_load(ai + p);
        s0 = dot16(av, wasm_v128_load(w0 + p), s0);
        s1 = dot16(av, wasm_v128_load(w1 + p), s1);
        s2 = dot16(av, wasm_v128_load(w2 + p), s2);
        s3 = dot16(av, wasm_v128_load(w3 + p), s3);
      }
      ci[j] = hsum(s0);
      ci[j + 1] = hsum(s1);
      ci[j + 2] = hsum(s2);
      ci[j + 3] = hsum(s3);
    }
    if (j < n) rayo_gemm_i8_generic(ai, lda, 1, w + j * ldw, ldw, n - j, k, ci + j, ldc);
  }
}
#else
/* ISO C needs a declaration in every file. */
typedef int rayo_wasm_simd_unused;
#endif
