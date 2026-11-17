/* masked_reduce: x (float32) reduced over one axis, keeping only the
 * positions a mask keeps. The mask has one value per position of that axis;
 * a position is kept where its value is not 0. The output is x's shape
 * without the axis.
 *
 * Attributes:
 * - "reduction": "max", "sum" or "mean".
 * - "axis": the axis reduced (default 0; negative counts from the end).
 * - "fill": when given, each position the mask drops counts as this value
 *   instead of being left out. Keras models often mask a max this way
 *   (x * m - (1 - m) * 1e4: "fill": -10000).
 * - "empty": the result where nothing counts (default 0).
 *
 * The mean divides the sum of the positions that count by their number. */
#include <string.h>

#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef enum reduction { MAX, SUM, MEAN } reduction;

typedef struct reduce_params {
  reduction how;
  rayo_i32 axis;
  rayo_bool has_fill;
  float fill;
  float empty;
} reduce_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  reduce_params *p;
  const char *how;
  rayo_status s;
  if (spec->inputs[0].weight || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: takes activations", spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s = rayo_attr_string(spec, "reduction", NULL, &how, msg);
  if (s == RAYO_OK) {
    s = rayo_attr_int(spec, "axis", 0, -RAYO_MAX_RANK, RAYO_MAX_RANK - 1, &p->axis, msg);
  }
  if (s == RAYO_OK) s = rayo_attr_float(spec, "fill", 0.0f, &p->fill, msg);
  if (s == RAYO_OK) s = rayo_attr_float(spec, "empty", 0.0f, &p->empty, msg);
  if (s != RAYO_OK) return s;
  p->has_fill = rayo_attr_has(spec, "fill");
  if (how && strcmp(how, "max") == 0) {
    p->how = MAX;
  } else if (how && strcmp(how, "sum") == 0) {
    p->how = SUM;
  } else if (how && strcmp(how, "mean") == 0) {
    p->how = MEAN;
  } else {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: reduction %s", spec->output,
                     how ? how : "(none)");
  }
  *params = p;
  return RAYO_OK;
}

static rayo_i32 axis_of(const reduce_params *p, rayo_i32 rank) {
  return p->axis < 0 ? p->axis + rank : p->axis;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const reduce_params *p = params;
  const rayo_i32 rank = in[0].shape.rank, axis = axis_of(p, rank);
  rayo_i32 d, k = 0;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32 || axis < 0 || axis >= rank ||
      rayo_shape_count(&in[1].shape) != (size_t)in[0].shape.dims[axis]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE,
                     "masked_reduce: float32 x and a float32 mask as long as the axis");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = rank - 1;
  for (d = 0; d < rank; d++) {
    if (d != axis) out->shape.dims[k++] = in[0].shape.dims[d];
  }
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const reduce_params *p = params;
  const rayo_shape *sh = &in[0].shape;
  const rayo_i32 axis = axis_of(p, sh->rank);
  const float *x = in[0].data, *m = in[1].data;
  float *y = out->data;
  size_t outer = 1, inner = 1, len, o, t, j, count = 0;
  rayo_i32 d;
  (void)exec;
  for (d = 0; d < axis; d++) outer *= (size_t)sh->dims[d];
  for (d = axis + 1; d < sh->rank; d++) inner *= (size_t)sh->dims[d];
  len = (size_t)sh->dims[axis];
  for (t = 0; t < len; t++) count += m[t] != 0.0f || p->has_fill;
  for (o = 0; o < outer; o++) {
    float *row = y + o * inner;
    const float *xs = x + o * len * inner;
    for (j = 0; j < inner; j++) row[j] = 0.0f;
    if (count == 0) {
      for (j = 0; j < inner; j++) row[j] = p->empty;
      continue;
    }
    if (p->how == MAX) {
      rayo_bool first = RAYO_TRUE;
      for (t = 0; t < len; t++) {
        const rayo_bool kept = m[t] != 0.0f;
        if (!kept && !p->has_fill) continue;
        for (j = 0; j < inner; j++) {
          const float v = kept ? xs[t * inner + j] : p->fill;
          row[j] = first || v > row[j] ? v : row[j];
        }
        first = RAYO_FALSE;
      }
    } else {
      for (t = 0; t < len; t++) {
        const rayo_bool kept = m[t] != 0.0f;
        if (!kept && !p->has_fill) continue;
        for (j = 0; j < inner; j++) row[j] += kept ? xs[t * inner + j] : p->fill;
      }
      if (p->how == MEAN) {
        const float inv = 1.0f / (float)count;
        for (j = 0; j < inner; j++) row[j] *= inv;
      }
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_masked_reduce = {"masked_reduce", 2, 2, prepare, infer, run, RAYO_FALSE};
