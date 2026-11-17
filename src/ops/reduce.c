/* reduce: sum, mean, max, min or prod of float32 x over one axis, summed in
 * float32 in order, as TFLite's reference kernels do. `keepdims` 1 keeps
 * the axis with size 1. */
#include <string.h>

#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef enum reduce_kind { RED_SUM, RED_MEAN, RED_MAX, RED_MIN, RED_PROD } reduce_kind;

typedef struct reduce_params {
  reduce_kind kind;
  rayo_i32 axis; /* negative counts from the end */
  rayo_i32 keepdims;
} reduce_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  static const char *const names[] = {"sum", "mean", "max", "min", "prod"};
  reduce_params *p;
  const char *how;
  rayo_status s;
  size_t i;
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: reduce takes an activation", spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_string(spec, "reduction", NULL, &how, msg);
  if (s != RAYO_OK) return s;
  for (i = 0; how && i < sizeof names / sizeof names[0]; i++) {
    if (strcmp(how, names[i]) == 0) break;
  }
  if (!how || i == sizeof names / sizeof names[0]) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: reduction sum, mean, max, min or prod",
                     spec->output);
  }
  p->kind = (reduce_kind)i;
  s = rayo_attr_int(spec, "axis", -1, -RAYO_MAX_RANK, RAYO_MAX_RANK - 1, &p->axis, msg);
  if (s == RAYO_OK) s = rayo_attr_int(spec, "keepdims", 0, 0, 1, &p->keepdims, msg);
  return s;
}

static rayo_i32 axis_of(const reduce_params *p, rayo_i32 rank) {
  return p->axis < 0 ? p->axis + rank : p->axis;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const reduce_params *p = params;
  const rayo_i32 rank = in[0].shape.rank, axis = axis_of(p, rank);
  rayo_i32 i;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || axis < 0 || axis >= rank || in[0].shape.dims[axis] < 1 ||
      (rank == 1 && !p->keepdims)) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "reduce: float32 of rank 2 or more (1 with keepdims)");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  if (p->keepdims) {
    out->shape.dims[axis] = 1;
  } else {
    for (i = axis; i + 1 < rank; i++) out->shape.dims[i] = out->shape.dims[i + 1];
    out->shape.rank = rank - 1;
  }
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const reduce_params *p = params;
  const rayo_shape *s = &in[0].shape;
  const rayo_i32 axis = axis_of(p, s->rank);
  const float *x = in[0].data;
  float *y = out->data;
  size_t outer = 1, inner = 1, n = (size_t)s->dims[axis], o, i, j;
  rayo_i32 d;
  (void)exec;
  for (d = 0; d < axis; d++) outer *= (size_t)s->dims[d];
  for (d = axis + 1; d < s->rank; d++) inner *= (size_t)s->dims[d];
  for (o = 0; o < outer; o++) {
    const float *xo = x + o * n * inner;
    float *yo = y + o * inner;
    memcpy(yo, xo, inner * sizeof(float));
    for (j = 1; j < n; j++) {
      const float *row = xo + j * inner;
      for (i = 0; i < inner; i++) {
        switch (p->kind) {
          case RED_SUM:
          case RED_MEAN:
            yo[i] += row[i];
            break;
          case RED_MAX:
            yo[i] = row[i] > yo[i] ? row[i] : yo[i];
            break;
          case RED_MIN:
            yo[i] = row[i] < yo[i] ? row[i] : yo[i];
            break;
          case RED_PROD:
            yo[i] *= row[i];
            break;
        }
      }
    }
    if (p->kind == RED_MEAN) {
      for (i = 0; i < inner; i++) yo[i] /= (float)n;
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_reduce = {"reduce", 1, 1, prepare, infer, run, RAYO_FALSE};
