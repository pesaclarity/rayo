/* The numeric kernels. Each has a portable C version (generic.c) and, where
 * the compiler targets it, a SIMD one (x86.c: SSE4.1 or AVX2; neon.c: NEON
 * on arm64 and armv7), chosen at compile time by the rayo_* dispatchers.
 * The tests run every SIMD kernel against the generic one. */
#ifndef RAYO_KERNELS_H
#define RAYO_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* Inner dimensions of int8 products are padded with zeros to a multiple of
 * this, so no kernel needs a tail loop. */
#define RAYO_K_ALIGN 32

/* c[i * ldc + j] = sum over p < k of a[i * lda + p] * w[j * ldw + p], for
 * i < m and j < n: int8 products accumulated exactly in int32. `k` is a
 * multiple of RAYO_K_ALIGN, and k * 127 * 128 fits in int32. */
typedef void rayo_gemm_i8_fn(const int8_t *restrict a, size_t lda, size_t m,
                             const int8_t *restrict w, size_t ldw, size_t n, size_t k,
                             int32_t *restrict c, size_t ldc);

/* The same in float32: c = a · wᵀ. */
typedef void rayo_gemm_f32_fn(const float *restrict a, size_t lda, size_t m,
                              const float *restrict w, size_t ldw, size_t n, size_t k,
                              float *restrict c, size_t ldc);

rayo_gemm_i8_fn rayo_gemm_i8_generic;
rayo_gemm_f32_fn rayo_gemm_f32_generic;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#define RAYO_KERNELS_X86 1
/* x86.c: SSE4.1 or AVX2 (whichever the build targets), int8 by int8. */
rayo_gemm_i8_fn rayo_gemm_i8_x86;
/* x86_vnni.c: AVX-VNNI, chosen at run time; reads `a` as uint8 (int8 + 128). */
rayo_gemm_i8_fn rayo_gemm_u8i8_avxvnni;
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define RAYO_KERNELS_NEON 1
/* neon.c: ARMv8.0 / ARMv7 NEON. */
rayo_gemm_i8_fn rayo_gemm_i8_neon;
#if defined(__aarch64__)
/* neon_dotprod.c: SDOT (ARMv8.2 dot product), chosen at run time. */
rayo_gemm_i8_fn rayo_gemm_i8_neon_dotprod;
#endif
#endif

/* A set of kernels. A session picks one when it is made and keeps it, so
 * there is no global state. */
typedef struct rayo_kernels {
  const char *name;
  /* 0, or 128 when gemm_i8 reads its `a` operand as uint8: the caller then
   * stores int8 values plus 128 (their bits with the top one flipped) and
   * adds 128 to the zero point. */
  int32_t a_offset;
  rayo_gemm_i8_fn *gemm_i8;
  rayo_gemm_f32_fn *gemm_f32;
} rayo_kernels;

/* The portable C kernels. */
void rayo_kernels_generic(rayo_kernels *out);

/* The fastest kernels this build has and this CPU runs. */
void rayo_kernels_best(rayo_kernels *out);

/* Quantises a group of rows of the row-major [*, width] float matrix `x` to
 * int8 with one scale and zero point for the whole group, exactly as
 * TFLite's hybrid kernels quantise a float input on the fly
 * (tensor_utils::AsymmetricQuantizeFloats): the range widened to include 0,
 * rounding halves away from zero. The group is rows first, first + step, ...
 * below `rows`; row r goes to q + r * ldq (zeros after `width` up to `ldq`
 * are left alone). */
void rayo_quantize_rows(const float *restrict x, size_t width, size_t first, size_t step,
                        size_t rows, int8_t *restrict q, size_t ldq, float *scale,
                        int32_t *zero_point);

/* For kernels that read uint8: adds `offset` (128) to the first `width`
 * int8 values of each of `rows` rows of q (stride ldq), in place, by
 * flipping their top bit, and to each row's zero point. */
void rayo_offset_rows(int8_t *restrict q, size_t ldq, size_t rows, size_t width,
                      int32_t *restrict zero, int32_t offset);

#endif
