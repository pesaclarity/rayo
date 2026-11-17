/* The shared part of dense and conv1d: a kernel seen as [out][k], packed at
 * load for the int8 or float product, and the step after the product (bias,
 * dequantisation, activation). */
#ifndef RAYO_OPS_LINEAR_H
#define RAYO_OPS_LINEAR_H

#include <stddef.h>
#include <stdint.h>

#include "ops/attrs.h"
#include "ops/op.h"

typedef struct rayo_linear {
  rayo_dtype dtype; /* the kernel's: RAYO_DTYPE_I8 or RAYO_DTYPE_F32 */
  int64_t out;
  int64_t k;
  int64_t k_padded;      /* int8: k rounded up to RAYO_K_ALIGN; float: k */
  const int8_t *w8;      /* int8: [out][k_padded], zeros after k */
  const float *wf;       /* float: [out][k] */
  const int32_t *rowsum; /* int8: the sum of each row */
  const float *wscale;   /* int8: each row's scale */
  const float *bias;     /* [out]; zeros when the node has none */
  rayo_activation act;
} rayo_linear;

/* Packs `kernel` (row-major, `out` rows of `k`) and `bias` (NULL or [out]
 * float32) into `arena`. An int8 kernel must be symmetric (zero point 0)
 * with one scale per row or one in all. */
rayo_status rayo_linear_pack(const rayo_op_spec *spec, const rayo_weight *kernel, int64_t out,
                             int64_t k, const rayo_weight *bias, rayo_arena *arena,
                             rayo_linear *lin, rayo_message *msg);

/* y[r][o] = act(bias[o] + (acc[r][o] - zero[r] * rowsum[o]) * scale[r] *
 * wscale[o]): the int8 product of inputs quantised with `scale` and `zero`
 * per row, back in floats. */
void rayo_linear_finish_i8(const rayo_linear *lin, const int32_t *restrict acc, size_t rows,
                           const float *restrict scale, const int32_t *restrict zero,
                           float *restrict y);

/* y = act(y + bias), in place, for `rows` rows of a float product. */
void rayo_linear_finish_f32(const rayo_linear *lin, float *restrict y, size_t rows);

#endif
