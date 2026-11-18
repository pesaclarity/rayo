/* The numeric kernels. Each has a portable C version (generic.c) and, where
 * the compiler targets it, SIMD ones (x86.c: SSE4.1 or AVX2; neon.c: NEON on
 * arm64 and armv7; x86_vnni.c, neon_dotprod.c and neon_i8mm.c: dot-product
 * and matrix instructions chosen at run time). The tests run every SIMD kernel against the generic
 * one. Only the SIMD files use compiler extensions (intrinsics, target
 * attributes, CPU checks); everything here is C89. */
#ifndef RAYO_KERNELS_H
#define RAYO_KERNELS_H

#include <stddef.h>

#include "core/compat.h"
#include "rayo/types.h"

/* Inner dimensions of int8 products are padded with zeros to a multiple of
 * this, so no kernel needs a tail loop. */
#define RAYO_K_ALIGN 32

/* c[i * ldc + j] = sum over p < k of a[i * lda + p] * w[j * ldw + p], for
 * i < m and j < n: int8 products accumulated exactly in int32. `k` is a
 * multiple of RAYO_K_ALIGN, and k * 127 * 128 fits in int32. */
typedef void rayo_gemm_i8_fn(const rayo_i8 *RAYO_RESTRICT a, size_t lda, size_t m,
                             const rayo_i8 *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                             rayo_i32 *RAYO_RESTRICT c, size_t ldc);

/* The same in float32: c = a · wᵀ. */
typedef void rayo_gemm_f32_fn(const float *RAYO_RESTRICT a, size_t lda, size_t m,
                              const float *RAYO_RESTRICT w, size_t ldw, size_t n, size_t k,
                              float *RAYO_RESTRICT c, size_t ldc);

/* Group-wise weights of 4 or 3 bits (docs/FORMAT.md, "int4" and "int3"),
 * groups of RAYO_Q4_GROUP values along each row of `w`:
 *
 * - 4 bits: 16 bytes a group, byte j holding value j in its low four bits
 *   and value j + 16 in its high four, each stored plus 8 (0 to 15 for -8
 *   to 7); a row is k / 2 bytes.
 * - 3 bits: 12 bytes a group: bytes 0 to 7 hold the low two bits, byte j
 *   those of values j, j + 8, j + 16 and j + 24 (bits 0-1, 2-3, 4-5, 6-7);
 *   bytes 8 to 11, read as a little-endian 32-bit word, bit i the third bit
 *   of value i; each stored plus 4 (0 to 7 for -4 to 3); a row is k * 3 / 8
 *   bytes.
 *
 * `wscale` has a scale per group, k / RAYO_Q4_GROUP per row: float32, or
 * with `half` set IEEE float16 (rayo_u16), converted exactly. `a` holds
 * the activations quantised to int8 by rayo_quantize_groups (k per row,
 * `ascale` k / RAYO_Q4_GROUP).
 *
 * c[i * ldc + j] is the same in every kernel, bit for bit: per group, the
 * eight sums of four neighbouring products (exact in int32) are converted
 * to float, multiplied by ascale * wscale and added into eight lanes, in
 * group order; the lanes are added at the end as
 * ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7)). `k` is a multiple of
 * RAYO_Q4_GROUP. */
#define RAYO_Q4_GROUP 32

typedef void rayo_gemm_q4_fn(const rayo_i8 *RAYO_RESTRICT a, const float *RAYO_RESTRICT ascale,
                             size_t m, const rayo_u8 *RAYO_RESTRICT w, const void *wscale,
                             rayo_bool half, size_t n, size_t k, float *RAYO_RESTRICT c,
                             size_t ldc);

/* Bytes of a group of 4-bit and of 3-bit weights. */
#define RAYO_Q4_GROUP_BYTES 16
#define RAYO_Q3_GROUP_BYTES 12

rayo_gemm_i8_fn rayo_gemm_i8_generic;
rayo_gemm_f32_fn rayo_gemm_f32_generic;
rayo_gemm_q4_fn rayo_gemm_q4_generic;
rayo_gemm_q4_fn rayo_gemm_q3_generic;

/* The eight lanes added in the order every Q4 kernel uses. */
float rayo_q4_lanes_sum(const float *lanes);

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#define RAYO_KERNELS_X86 1
/* x86.c: SSE4.1 and AVX2, int8 by int8 and 4-bit weights by int8. With
 * gcc or clang both are compiled (with target attributes) and chosen at run
 * time; other compilers get what the build's flags target. */
#if defined(__SSE4_1__) || defined(__GNUC__) || defined(__clang__)
#define RAYO_KERNELS_X86_SIMD 1
rayo_gemm_i8_fn rayo_gemm_i8_sse41;
rayo_gemm_q4_fn rayo_gemm_q4_sse41;
rayo_gemm_q4_fn rayo_gemm_q3_sse41;
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
#define RAYO_KERNELS_X86_AVX2 1
rayo_gemm_i8_fn rayo_gemm_i8_avx2;
rayo_gemm_q4_fn rayo_gemm_q4_avx2;
rayo_gemm_q4_fn rayo_gemm_q3_avx2;
/* The AVX2 decoder product (a single row by four weight rows, x86.c) is
 * left out of Android builds, to keep the x86_64 library within CI's
 * 200 KB budget: Android on x86_64 is emulators and a few Chromebooks,
 * not phones, and without it a decoder's row takes the row by row product,
 * the same floats a little slower. -DRAYO_AVX2_DECODE=1 puts it back, =0
 * leaves it out elsewhere. */
#if !defined(RAYO_AVX2_DECODE)
#if defined(__ANDROID__)
#define RAYO_AVX2_DECODE 0
#else
#define RAYO_AVX2_DECODE 1
#endif
#endif
#endif
#endif
#if defined(__GNUC__) || defined(__clang__)
/* Whether this CPU (and OS) runs SSE4.1, and AVX2 with FMA (x86_vnni.c). */
rayo_bool rayo_cpu_has_sse41(void);
rayo_bool rayo_cpu_has_avx2(void);
#endif
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_KERNELS_X86_VNNI 1
/* x86_vnni.c: AVX-VNNI, chosen at run time; reads `a` as uint8 (int8 + 128). */
rayo_gemm_i8_fn rayo_gemm_u8i8_avxvnni;
/* Whether this CPU (and OS) runs AVX-VNNI. */
rayo_bool rayo_cpu_has_avxvnni(void);
#endif
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define RAYO_KERNELS_NEON 1
/* neon.c: ARMv8.0 / ARMv7 NEON. */
rayo_gemm_i8_fn rayo_gemm_i8_neon;
rayo_gemm_q4_fn rayo_gemm_q4_neon;
rayo_gemm_q4_fn rayo_gemm_q3_neon;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define RAYO_KERNELS_NEON_DOTPROD 1
/* neon_dotprod.c: SDOT (ARMv8.2 dot product), chosen at run time. */
rayo_gemm_i8_fn rayo_gemm_i8_neon_dotprod;
rayo_gemm_q4_fn rayo_gemm_q4_neon_dotprod;
rayo_gemm_q4_fn rayo_gemm_q3_neon_dotprod;
/* Whether this CPU has the dot product (Linux, Android and macOS, else no). */
rayo_bool rayo_cpu_has_dotprod(void);
#define RAYO_KERNELS_NEON_I8MM 1
/* neon_i8mm.c: SMMLA (ARMv8.6 int8 matrix multiply), chosen at run time. */
rayo_gemm_i8_fn rayo_gemm_i8_neon_i8mm;
/* Whether this CPU has the int8 matrix multiply and the dot product. */
rayo_bool rayo_cpu_has_i8mm(void);
#endif
#endif

/* A set of kernels. A session picks one when it is made and keeps it, so
 * there is no global state. */
typedef struct rayo_kernels {
  const char *name;
  /* 0, or 128 when gemm_i8 reads its `a` operand as uint8: the caller then
   * stores int8 values plus 128 (their bits with the top one flipped) and
   * adds 128 to the zero point. */
  rayo_i32 a_offset;
  rayo_gemm_i8_fn *gemm_i8;
  rayo_gemm_f32_fn *gemm_f32;
  rayo_gemm_q4_fn *gemm_q4;
  rayo_gemm_q4_fn *gemm_q3;
} rayo_kernels;

/* The portable C kernels. */
void rayo_kernels_generic(rayo_kernels *out);

/* The build's SIMD kernels, without the run-time dot-product choice. */
void rayo_kernels_simd(rayo_kernels *out);

/* The fastest kernels this build has and this CPU runs. */
void rayo_kernels_best(rayo_kernels *out);

/* Quantises a group of rows of the row-major [*, width] float matrix `x` to
 * int8 with one scale and zero point for the whole group, exactly as
 * TFLite's hybrid kernels quantise a float input on the fly
 * (tensor_utils::AsymmetricQuantizeFloats): the range widened to include 0,
 * rounding halves away from zero. The group is rows first, first + step, ...
 * below `rows`; row r goes to q + r * ldq (zeros after `width` up to `ldq`
 * are left alone). */
void rayo_quantize_rows(const float *RAYO_RESTRICT x, size_t width, size_t first, size_t step,
                        size_t rows, rayo_i8 *RAYO_RESTRICT q, size_t ldq, float *scale,
                        rayo_i32 *zero_point);

/* Quantises each of `rows` rows of `x` ([rows, k], k a multiple of
 * RAYO_Q4_GROUP) to int8, symmetric, one scale per group of RAYO_Q4_GROUP:
 * scale = max |x| / 127, q = x * (127 / max |x|) rounded half away from
 * zero. A group of zeros, or one that is not finite, gives zeros and scale
 * 0. Row r goes to q + r * k and scale + r * k / RAYO_Q4_GROUP. */
void rayo_quantize_groups(const float *RAYO_RESTRICT x, size_t rows, size_t k,
                          rayo_i8 *RAYO_RESTRICT q, float *RAYO_RESTRICT scale);

/* rayo_quantize_groups with groups of `group` (k a multiple of it). */
void rayo_quantize_groups_of(const float *RAYO_RESTRICT x, size_t rows, size_t k, size_t group,
                             rayo_i8 *RAYO_RESTRICT q, float *RAYO_RESTRICT scale);

/* For kernels that read uint8: adds `offset` (128) to the first `width`
 * int8 values of each of `rows` rows of q (stride ldq), in place, by
 * flipping their top bit, and to each row's zero point. */
void rayo_offset_rows(rayo_i8 *RAYO_RESTRICT q, size_t ldq, size_t rows, size_t width,
                      rayo_i32 *RAYO_RESTRICT zero, rayo_i32 offset);

#endif
