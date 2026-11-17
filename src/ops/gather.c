/* gather: rows of a weight table by int32 id: table [rows, width] and ids
 * [n] to float32 [n, width]. An int8 table is dequantised as it is read. */
#include <string.h>

#include "core/message.h"
#include "ops/ops.h"

typedef struct gather_params {
  const rayo_weight *table;
  int64_t rows;
  int64_t width;
} gather_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *t = spec->inputs[0].weight;
  if (!t || t->shape.rank != 2 || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: gather takes a [rows, width] table and ids",
                     spec->output);
  }
  if (t->dtype == RAYO_DTYPE_I8 && t->axis != -1) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: an int8 table needs one scale",
                     spec->output);
  }
  gather_params *p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  p->table = t;
  p->rows = t->shape.dims[0];
  p->width = t->shape.dims[1];
  *params = p;
  return RAYO_OK;
}

static rayo_status infer(const void *params, const rayo_tensor *in, rayo_tensor *out,
                         size_t *scratch, rayo_message *msg) {
  const gather_params *p = params;
  if (in[1].dtype != RAYO_DTYPE_I32 || in[1].shape.rank != 1) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "gather: ids must be int32 [n]");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 2;
  out->shape.dims[0] = in[1].shape.dims[0];
  out->shape.dims[1] = p->width;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  (void)exec;
  const gather_params *p = params;
  const int32_t *ids = in[1].data;
  const size_t n = (size_t)in[1].shape.dims[0], w = (size_t)p->width;
  float *y = out->data;
  for (size_t i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= p->rows) return RAYO_ERR_SHAPE;
  }
  if (p->table->dtype == RAYO_DTYPE_F32) {
    const float *t = p->table->data;
    for (size_t i = 0; i < n; i++) memcpy(y + i * w, t + (size_t)ids[i] * w, w * sizeof(float));
    return RAYO_OK;
  }
  const int8_t *t = p->table->data;
  const float s = p->table->scales[0];
  const int32_t z = p->table->zero_point;
  for (size_t i = 0; i < n; i++) {
    const int8_t *row = t + (size_t)ids[i] * w;
    for (size_t j = 0; j < w; j++) y[i * w + j] = (float)(row[j] - z) * s;
  }
  return RAYO_OK;
}

const rayo_op rayo_op_gather = {"gather", 2, 2, prepare, infer, run};
