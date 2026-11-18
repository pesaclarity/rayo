/* Choosing kernels: what the build targets, plus the dot-product
 * instructions when this CPU has them (asked each time, never cached). */
#include "kernels/kernels.h"

void rayo_kernels_generic(rayo_kernels *out) {
  out->name = "generic";
  out->a_offset = 0;
  out->gemm_i8 = rayo_gemm_i8_generic;
  out->gemm_f32 = rayo_gemm_f32_generic;
  out->gemm_q4 = rayo_gemm_q4_generic;
}

void rayo_kernels_simd(rayo_kernels *out) {
  rayo_kernels_generic(out);
#if defined(RAYO_KERNELS_X86_SIMD)
#if defined(__AVX2__)
  out->name = "avx2";
#else
  out->name = "sse4.1";
#endif
  out->gemm_i8 = rayo_gemm_i8_x86;
  out->gemm_q4 = rayo_gemm_q4_x86;
#elif defined(RAYO_KERNELS_NEON)
  out->name = "neon";
  out->gemm_i8 = rayo_gemm_i8_neon;
  out->gemm_q4 = rayo_gemm_q4_neon;
#endif
}

void rayo_kernels_best(rayo_kernels *out) {
  rayo_kernels_simd(out);
#if defined(RAYO_KERNELS_X86_VNNI)
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
  }
#endif
}
