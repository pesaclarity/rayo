/* gather: rows of a weight table by int32 id: table [rows, width] and ids
 * [n] to float32 [n, width]. An int8 table is dequantised as it is read. */
#include <string.h>

#include "core/message.h"
#include "kernels/half.h"
#include "ops/ops.h"

typedef struct gather_params {
  const rayo_weight *table;
  rayo_i32 rows;
  size_t width;
} gather_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *t = spec->inputs[0].weight;
  gather_params *p;
  if (!t || t->shape.rank != 2 || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: gather takes a [rows, width] table and ids",
                     spec->output);
  }
  if (t->dtype == RAYO_DTYPE_I8 && t->axis != -1 && t->axis != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "node %s: an int8 table needs one scale, or one per row", spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  p->table = t;
  p->rows = t->shape.dims[0];
  p->width = (size_t)t->shape.dims[1];
  *params = p;
  return RAYO_OK;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const gather_params *p = params;
  (void)symbols;
  if (in[1].dtype != RAYO_DTYPE_I32 || in[1].shape.rank != 1) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "gather: ids must be int32 [n]");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 2;
  out->shape.dims[0] = in[1].shape.dims[0];
  out->shape.dims[1] = (rayo_i32)p->width;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const gather_params *p = params;
  const rayo_i32 *ids = in[1].data;
  const size_t n = (size_t)in[1].shape.dims[0], w = p->width;
  float *y = out->data;
  size_t i, j;
  (void)exec;
  for (i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= p->rows) return RAYO_ERR_SHAPE;
  }
  if (p->table->dtype == RAYO_DTYPE_F32) {
    const float *t = p->table->data;
    for (i = 0; i < n; i++) memcpy(y + i * w, t + (size_t)ids[i] * w, w * sizeof(float));
  } else if (p->table->dtype == RAYO_DTYPE_I4 || p->table->dtype == RAYO_DTYPE_I3) {
    /* Groups of RAYO_PCWT_INT4_GROUP along the row (docs/FORMAT.md). */
    const rayo_weight *tw = p->table;
    const rayo_u8 *t = tw->data;
    const size_t g = (size_t)tw->group_size, groups = w / g;
    const size_t bytes =
        tw->dtype == RAYO_DTYPE_I4 ? RAYO_PCWT_INT4_GROUP_BYTES : RAYO_PCWT_INT3_GROUP_BYTES;
    for (i = 0; i < n; i++) {
      const rayo_u8 *row = t + (size_t)ids[i] * groups * bytes;
      for (j = 0; j < w; j++) {
        const rayo_u8 *gb = row + j / g * bytes;
        const size_t e = j % g;
        const size_t s = (size_t)ids[i] * groups + j / g;
        const float sc = rayo_group_scale(tw->group_scales, tw->scales_half, s);
        int v;
        if (tw->dtype == RAYO_DTYPE_I4) {
          v = (e < 16 ? gb[e] & 15 : gb[e - 16] >> 4) - 8;
        } else {
          v = (int)((((unsigned)gb[e % 8] >> (2 * (e / 8))) & 3u) |
                    ((((unsigned)gb[8 + e / 8] >> (e % 8)) & 1u) << 2)) -
              4;
        }
        y[i * w + j] = (float)v * sc;
      }
    }
  } else {
    const rayo_i8 *t = p->table->data;
    const rayo_i32 z = p->table->zero_point;
    for (i = 0; i < n; i++) {
      const rayo_i8 *row = t + (size_t)ids[i] * w;
      const float s = p->table->scales[p->table->axis == 0 ? (size_t)ids[i] : 0];
      for (j = 0; j < w; j++) y[i * w + j] = (float)(row[j] - z) * s;
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_gather = {"gather", 2, 2, prepare, infer, run, RAYO_FALSE};
