/* Choosing kernels: what the build targets, plus the dot-product
 * instructions when this CPU has them (checked here, never cached). */
#include "kernels/kernels.h"

#if defined(RAYO_KERNELS_X86) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif
#if defined(RAYO_KERNELS_NEON) && defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

void rayo_kernels_generic(rayo_kernels *out) {
  out->name = "generic";
  out->a_offset = 0;
  out->gemm_i8 = rayo_gemm_i8_generic;
  out->gemm_f32 = rayo_gemm_f32_generic;
}

#if defined(RAYO_KERNELS_X86) && (defined(__GNUC__) || defined(__clang__))
/* AVX-VNNI: CPUID leaf 7 subleaf 1, EAX bit 4; with AVX2 (leaf 7, EBX bit 5)
 * and the OS saving the YMM registers (XCR0 bits 1 and 2). */
static int has_avxvnni(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) return 0; /* OSXSAVE */
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  if ((lo & 6u) != 6u) return 0;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(b & (1u << 5))) return 0;
  if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return 0;
  return (a & (1u << 4)) != 0;
}
#endif

void rayo_kernels_best(rayo_kernels *out) {
  rayo_kernels_generic(out);
#if defined(RAYO_KERNELS_X86)
#if defined(__AVX2__)
  out->name = "avx2";
  out->gemm_i8 = rayo_gemm_i8_x86;
#elif defined(__SSE4_1__)
  out->name = "sse4.1";
  out->gemm_i8 = rayo_gemm_i8_x86;
#endif
#if defined(__GNUC__) || defined(__clang__)
  if (has_avxvnni()) {
    out->name = "avx-vnni";
    out->a_offset = 128;
    out->gemm_i8 = rayo_gemm_u8i8_avxvnni;
  }
#endif
#elif defined(RAYO_KERNELS_NEON)
  out->name = "neon";
  out->gemm_i8 = rayo_gemm_i8_neon;
#if defined(__aarch64__) && defined(__linux__)
  if (getauxval(AT_HWCAP) & (1ul << 20)) { /* HWCAP_ASIMDDP */
    out->name = "neon-dotprod";
    out->gemm_i8 = rayo_gemm_i8_neon_dotprod;
  }
#endif
#endif
}
