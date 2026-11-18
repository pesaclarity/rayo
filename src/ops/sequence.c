/* Ops for running a sequence step by step.
 *
 * range: int32 [size]: start, start + 1, ... (each a whole number or a size
 *   symbol), such as the positions P .. P + T - 1 of the new tokens.
 * cache_write: writes x [T, C] into rows at .. at + T - 1 of a state
 *   [N, C] and gives the state's rows 0 .. at + T - 1, a view: the keys or
 *   values of every position so far. A session may keep the state in int8
 *   with a scale per row and group (rayo_session_options.cache): each row
 *   is then quantised as it is written, and the view is int8 too.
 * last_row: the last row of x [T, C], as [C]. */
#include <string.h>

#include "core/message.h"
#include "kernels/kernels.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct size_params {
  rayo_size_attr start;
  rayo_size_attr size;
} size_params;

static rayo_status range_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                 rayo_message *msg) {
  size_params *p = rayo_arena_alloc(arena, sizeof *p);
  rayo_status s;
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_size(spec, "start", 0, &p->start, msg);
  if (s == RAYO_OK) s = rayo_attr_size(spec, "size", 1, &p->size, msg);
  return s;
}

static rayo_status range_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const size_params *p = params;
  const rayo_i32 n = rayo_size_value(p->size, symbols);
  (void)in;
  if (n < 1 || rayo_size_value(p->start, symbols) > RAYO_I32_MAX - n) {
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
  const size_params *p = params;
  const rayo_i32 start = rayo_size_value(p->start, exec->symbols);
  rayo_i32 *y = out->data;
  rayo_i32 i;
  (void)in;
  for (i = 0; i < out->shape.dims[0]; i++) y[i] = start + i;
  return RAYO_OK;
}

static rayo_status cache_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                 rayo_message *msg) {
  size_params *p;
  if (spec->inputs[0].shape.rank != 2 || spec->inputs[0].weight || spec->inputs[1].weight ||
      spec->inputs[0].data) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: cache_write takes a [N, C] state and rows",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  return rayo_attr_size(spec, "at", 0, &p->start, msg);
}

static rayo_status cache_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const size_params *p = params;
  const rayo_shape *st = &in[0].shape, *x = &in[1].shape;
  const rayo_i32 at = rayo_size_value(p->start, symbols);
  if (in[1].dtype != RAYO_DTYPE_F32 || x->rank != 2 || x->dims[1] != st->dims[1] ||
      at > st->dims[0] - x->dims[0]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "cache_write: %ld rows at %ld do not fit %ld",
                     (long)x->dims[0], (long)at, (long)st->dims[0]);
  }
  if (in[0].dtype == RAYO_DTYPE_I8 && (in[0].group <= 0 || st->dims[1] % in[0].group != 0)) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "cache_write: an int8 state needs groups of its rows");
  }
  out->dtype = in[0].dtype;
  out->shape = *st;
  out->shape.dims[0] = at + x->dims[0];
  out->scales = in[0].scales;
  out->group = in[0].group;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status cache_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                             const rayo_exec *exec) {
  const size_params *p = params;
  const size_t at = (size_t)rayo_size_value(p->start, exec->symbols);
  const size_t c = (size_t)in[0].shape.dims[1];
  /* out->data is the state's memory: the view the next ops read. */
  if (out->dtype == RAYO_DTYPE_I8) {
    const size_t g = (size_t)out->group;
    rayo_quantize_groups_of(in[1].data, (size_t)in[1].shape.dims[0], c, g,
                            (rayo_i8 *)out->data + at * c, out->scales + at * (c / g));
    return RAYO_OK;
  }
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

static rayo_status last_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
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
  const size_t t = (size_t)in[0].shape.dims[0], c = (size_t)in[0].shape.dims[1];
  (void)params;
  (void)exec;
  memcpy(out->data, (const float *)in[0].data + (t - 1) * c, c * sizeof(float));
  return RAYO_OK;
}

const rayo_op rayo_op_range = {"range", 0, 0, range_prepare, range_infer, range_run, RAYO_FALSE};
const rayo_op rayo_op_cache_write = {"cache_write", 2,         2,        cache_prepare,
                                     cache_infer,   cache_run, RAYO_TRUE};
const rayo_op rayo_op_last_row = {"last_row", 1, 1, last_prepare, last_infer, last_run, RAYO_FALSE};
