/* The plain-weights file (PCWT, pesaclarity-ml docs/WEIGHTS.md): a JSON
 * header and the tensors. Reading checks every size and offset against the
 * file, which is untrusted. */
#ifndef RAYO_FORMAT_PCWT_H
#define RAYO_FORMAT_PCWT_H

#include <stddef.h>

#include "core/arena.h"
#include "core/json.h"
#include "core/names.h"
#include "rayo/error.h"
#include "rayo/types.h"

#define RAYO_PCWT_VERSION 1

/* The one group size int4 and int3 weights may have (the kernels'
 * RAYO_Q4_GROUP), and the bytes of a group of each. */
#define RAYO_PCWT_INT4_GROUP 32
#define RAYO_PCWT_INT4_GROUP_BYTES 16
#define RAYO_PCWT_INT3_GROUP_BYTES 12

/* One tensor of the file. An int8 tensor's real values are
 * `scales[c] * (value - zero_point)`, `c` its index along `axis` (-1: one
 * scale for the whole tensor). An int4 tensor holds two values a byte in
 * groups of `group_size` along its innermost dimension (docs/FORMAT.md):
 * its real values are `scales[group] * (value - 8)`; an int3 tensor 12
 * bytes a group, `scales[group] * (value - 4)`. Group scales are float32 or
 * float16 (`scales_half`), at `group_scales`. */
typedef struct rayo_weight {
  const char *name;
  rayo_dtype dtype; /* RAYO_DTYPE_F32, RAYO_DTYPE_I8, RAYO_DTYPE_I4 or RAYO_DTYPE_I3 */
  rayo_shape shape;
  size_t count; /* elements; with their bytes below 2^31 */
  const void *data;
  const float *scales; /* int8; int4 and int3 with float32 scales */
  size_t scale_count;
  rayo_i32 zero_point;
  rayo_i32 axis;
  rayo_i32 group_size;      /* int4 and int3 */
  const void *group_scales; /* int4 and int3: float32, or float16 with scales_half */
  rayo_bool scales_half;    /* int4 and int3 */
} rayo_weight;

typedef struct rayo_pcwt {
  const rayo_json *header;
  const char *model; /* the header's "model" */
  rayo_weight *weights;
  size_t weight_count;
  rayo_names names; /* each weight's index by name */
  /* The tokeniser's table (the header's "tokenizer"), or NULL. */
  const rayo_u8 *tokenizer;
  size_t tokenizer_size;
} rayo_pcwt;

/* Reads the `size` bytes at `file`, which must be 4-byte aligned and stay
 * alive as long as `out` is used (the weights point into it). The header
 * and the table live in `arena`. A header over RAYO_MAX_HEADER_BYTES or
 * RAYO_MAX_HEADER_VALUES, or more than RAYO_MAX_TENSORS tensors, is
 * RAYO_ERR_LIMIT. */
rayo_status rayo_pcwt_read(const rayo_u8 *file, size_t size, rayo_arena *arena, rayo_pcwt *out,
                           rayo_message *msg);

/* The weight called `name`, or NULL. */
const rayo_weight *rayo_pcwt_find(const rayo_pcwt *pcwt, const char *name);

#endif
