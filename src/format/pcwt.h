/* The plain-weights file (PCWT, pesaclarity-ml docs/WEIGHTS.md): a JSON
 * header and the tensors. Reading checks every size and offset against the
 * file, which is untrusted. */
#ifndef RAYO_FORMAT_PCWT_H
#define RAYO_FORMAT_PCWT_H

#include <stddef.h>

#include "core/arena.h"
#include "core/json.h"
#include "rayo/error.h"
#include "rayo/types.h"

#define RAYO_PCWT_VERSION 1

/* One tensor of the file. An int8 tensor's real values are
 * `scales[c] * (value - zero_point)`, `c` its index along `axis` (-1: one
 * scale for the whole tensor). */
typedef struct rayo_weight {
  const char *name;
  rayo_dtype dtype; /* RAYO_DTYPE_F32 or RAYO_DTYPE_I8 */
  rayo_shape shape;
  size_t count; /* elements; with their bytes below 2^31 */
  const void *data;
  const float *scales; /* int8 only */
  size_t scale_count;
  rayo_i32 zero_point;
  rayo_i32 axis;
} rayo_weight;

typedef struct rayo_pcwt {
  const rayo_json *header;
  const char *model; /* the header's "model" */
  rayo_weight *weights;
  size_t weight_count;
} rayo_pcwt;

/* Reads the `size` bytes at `file`, which must be 4-byte aligned and stay
 * alive as long as `out` is used (the weights point into it). The header
 * and the table live in `arena`. */
rayo_status rayo_pcwt_read(const rayo_u8 *file, size_t size, rayo_arena *arena, rayo_pcwt *out,
                           rayo_message *msg);

/* The weight called `name`, or NULL. */
const rayo_weight *rayo_pcwt_find(const rayo_pcwt *pcwt, const char *name);

#endif
