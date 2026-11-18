/* Choosing kernels: what the build targets, plus the dot-product and
 * matrix instructions when this CPU has them (asked each time, never
 * cached). */
#include "kernels/kernels.h"

void rayo_kernels_generic(rayo_kernels *out) {
  out->name = "generic";
  out->a_offset = 0;
  out->gemm_i8 = rayo_gemm_i8_generic;
  out->gemm_f32 = rayo_gemm_f32_generic;
  out->gemm_q4 = rayo_gemm_q4_generic;
  out->gemm_q3 = rayo_gemm_q3_generic;
}

void rayo_kernels_simd(rayo_kernels *out) {
  rayo_kernels_generic(out);
#if defined(__AVX2__) && defined(RAYO_KERNELS_X86_AVX2)
  out->name = "avx2";
  out->gemm_i8 = rayo_gemm_i8_avx2;
  out->gemm_q4 = rayo_gemm_q4_avx2;
  out->gemm_q3 = rayo_gemm_q3_avx2;
#elif defined(__SSE4_1__) && defined(RAYO_KERNELS_X86_SIMD)
  out->name = "sse4.1";
  out->gemm_i8 = rayo_gemm_i8_sse41;
  out->gemm_q4 = rayo_gemm_q4_sse41;
  out->gemm_q3 = rayo_gemm_q3_sse41;
#elif defined(RAYO_KERNELS_NEON)
  out->name = "neon";
  out->gemm_i8 = rayo_gemm_i8_neon;
  out->gemm_q4 = rayo_gemm_q4_neon;
  out->gemm_q3 = rayo_gemm_q3_neon;
#endif
}

void rayo_kernels_best(rayo_kernels *out) {
  rayo_kernels_simd(out);
#if defined(RAYO_KERNELS_X86_VNNI)
  /* What the CPU has beyond what the build targets. */
  if (!out->gemm_q4 || out->gemm_q4 == rayo_gemm_q4_generic) {
    if (rayo_cpu_has_sse41()) {
      out->name = "sse4.1";
      out->gemm_i8 = rayo_gemm_i8_sse41;
      out->gemm_q4 = rayo_gemm_q4_sse41;
      out->gemm_q3 = rayo_gemm_q3_sse41;
    }
  }
  if (out->gemm_q4 != rayo_gemm_q4_avx2 && rayo_cpu_has_avx2()) {
    out->name = "avx2";
    out->gemm_i8 = rayo_gemm_i8_avx2;
    out->gemm_q4 = rayo_gemm_q4_avx2;
    out->gemm_q3 = rayo_gemm_q3_avx2;
  }
  if (rayo_cpu_has_avxvnni()) {
    out->name = "avx-vnni";
    out->a_offset = 128;
    out->gemm_i8 = rayo_gemm_u8i8_avxvnni;
  }
#endif
#if defined(RAYO_KERNELS_NEON_DOTPROD)
  if (rayo_cpu_has_dotprod()) {
    out->name = "neon-dotprod";
    out->gemm_i8 = rayo_gemm_i8_neon_dotprod;
    out->gemm_q4 = rayo_gemm_q4_neon_dotprod;
    out->gemm_q3 = rayo_gemm_q3_neon_dotprod;
  }
#endif
#if defined(RAYO_KERNELS_NEON_I8MM)
  if (rayo_cpu_has_i8mm()) {
    out->name = "neon-i8mm";
    out->gemm_i8 = rayo_gemm_i8_neon_i8mm;
  }
#endif
}
