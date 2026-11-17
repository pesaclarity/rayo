/* Ops for running a sequence step by step.
 *
 * range: int32 [size]: start, start + 1, ... (each a whole number or a size
 *   symbol), such as the positions P .. P + T - 1 of the new tokens.
 * cache_write: writes x [T, C] into rows at .. at + T - 1 of a state
 *   [N, C] and gives the state's rows 0 .. at + T - 1, a view: the keys or
 *   values of every position so far.
 * last_row: the last row of x [T, C], as [C]. */
#include <string.h>

#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct size_params {
  rayo_size_attr start;
  rayo_size_attr size;
} size_params;

static rayo_status range_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                 rayo_message *msg) {
  size_params *p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  rayo_status s = rayo_attr_size(spec, "start", 0, &p->start, msg);
  if (s == RAYO_OK) s = rayo_attr_size(spec, "size", 1, &p->size, msg);
  return s;
}

static rayo_status range_infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)in;
  const size_params *p = params;
  const int64_t n = rayo_size_value(p->size, symbols);
  if (n < 1 || rayo_size_value(p->start, symbols) + n > INT32_MAX) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "range: an empty or too large range");
  }
  out->dtype = RAYO_DTYPE_I32;
  out->shape.rank = 1;
  out->shape.dims[0] = n;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status range_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                             const rayo_exec *exec) {
  (void)in;
  const size_params *p = params;
  const int64_t start = rayo_size_value(p->start, exec->symbols);
  int32_t *y = out->data;
  for (int64_t i = 0; i < out->shape.dims[0]; i++) y[i] = (int32_t)(start + i);
  return RAYO_OK;
}

static rayo_status cache_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                 rayo_message *msg) {
  if (spec->inputs[0].shape.rank != 2 || spec->inputs[0].weight || spec->inputs[1].weight ||
      spec->inputs[0].data) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: cache_write takes a [N, C] state and rows",
                     spec->output);
  }
  size_params *p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  return rayo_attr_size(spec, "at", 0, &p->start, msg);
}

static rayo_status cache_infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const size_params *p = params;
  const rayo_shape *st = &in[0].shape, *x = &in[1].shape;
  const int64_t at = rayo_size_value(p->start, symbols);
  if (in[1].dtype != RAYO_DTYPE_F32 || x->rank != 2 || x->dims[1] != st->dims[1] ||
      at + x->dims[0] > st->dims[0]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "cache_write: %lld rows at %lld do not fit %lld",
                     (long long)x->dims[0], (long long)at, (long long)st->dims[0]);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = *st;
  out->shape.dims[0] = at + x->dims[0];
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status cache_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                             const rayo_exec *exec) {
  const size_params *p = params;
  const size_t at = (size_t)rayo_size_value(p->start, exec->symbols);
  const size_t c = (size_t)in[0].shape.dims[1];
  /* out->data is the state's memory: the view the next ops read. */
  memcpy((float *)out->data + at * c, in[1].data, rayo_tensor_bytes(&in[1]));
  return RAYO_OK;
}

static rayo_status last_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                rayo_message *msg) {
  (void)arena;
  *params = NULL;
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: last_row takes an activation", spec->output);
  }
  return RAYO_OK;
}

static rayo_status last_infer(const void *params, const rayo_tensor *in, const int64_t *symbols,
                              rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank != 2) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "last_row: float32 [T, C]");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = 1;
  out->shape.dims[0] = in[0].shape.dims[1];
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status last_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                            const rayo_exec *exec) {
  (void)params;
  (void)exec;
  const size_t t = (size_t)in[0].shape.dims[0], c = (size_t)in[0].shape.dims[1];
  memcpy(out->data, (const float *)in[0].data + (t - 1) * c, c * sizeof(float));
  return RAYO_OK;
}

const rayo_op rayo_op_range = {"range", 0, 0, range_prepare, range_infer, range_run, false};
const rayo_op rayo_op_cache_write = {"cache_write", 2,         2,   cache_prepare,
                                     cache_infer,   cache_run, true};
const rayo_op rayo_op_last_row = {"last_row", 1, 1, last_prepare, last_infer, last_run, false};
