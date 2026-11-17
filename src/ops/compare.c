/* compare: x (float32 or int32) against a number, elementwise, to 1 where
 * the comparison holds and 0 where it does not, as float32 (default) or
 * int32. Attributes: "comparison", one of greater, greater_equal, less,
 * less_equal, equal and not_equal; "value", the number (default 0);
 * "dtype", the output's. A padding mask over token ids is
 * {"comparison": "greater", "value": 0}. */
#include <string.h>

#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef enum comparison { GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, NOT_EQUAL } comparison;

typedef struct compare_params {
  comparison how;
  double value;
  rayo_dtype dtype;
} compare_params;

static const char *const names[] = {"greater",    "greater_equal", "less",
                                    "less_equal", "equal",         "not_equal"};

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  compare_params *p;
  const char *how, *dtype;
  rayo_status s;
  size_t i;
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: compare takes an activation", spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s = rayo_attr_string(spec, "comparison", NULL, &how, msg);
  if (s == RAYO_OK) s = rayo_attr_double(spec, "value", 0.0, &p->value, msg);
  if (s == RAYO_OK) s = rayo_attr_string(spec, "dtype", "float32", &dtype, msg);
  if (s != RAYO_OK) return s;
  if (!how)
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: compare needs a comparison", spec->output);
  for (i = 0; i < sizeof names / sizeof names[0]; i++) {
    if (strcmp(how, names[i]) == 0) break;
  }
  if (i == sizeof names / sizeof names[0]) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: comparison %s", spec->output, how);
  }
  p->how = (comparison)i;
  if (strcmp(dtype, "float32") == 0) {
    p->dtype = RAYO_DTYPE_F32;
  } else if (strcmp(dtype, "int32") == 0) {
    p->dtype = RAYO_DTYPE_I32;
  } else {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: output dtype %s", spec->output, dtype);
  }
  *params = p;
  return RAYO_OK;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const compare_params *p = params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 && in[0].dtype != RAYO_DTYPE_I32) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "compare: a float32 or int32 input");
  }
  out->dtype = p->dtype;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_bool holds(comparison how, double a, double b) {
  switch (how) {
    case GREATER:
      return a > b;
    case GREATER_EQUAL:
      return a >= b;
    case LESS:
      return a < b;
    case LESS_EQUAL:
      return a <= b;
    case EQUAL:
      return a == b;
    default:
      return a != b;
  }
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const compare_params *p = params;
  const size_t n = rayo_shape_count(&out->shape);
  size_t i;
  (void)exec;
  for (i = 0; i < n; i++) {
    const double a = in[0].dtype == RAYO_DTYPE_F32 ? (double)((const float *)in[0].data)[i]
                                                   : (double)((const rayo_i32 *)in[0].data)[i];
    const rayo_bool yes = holds(p->how, a, p->value);
    if (p->dtype == RAYO_DTYPE_F32) {
      ((float *)out->data)[i] = yes ? 1.0f : 0.0f;
    } else {
      ((rayo_i32 *)out->data)[i] = yes ? 1 : 0;
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_compare = {"compare", 1, 1, prepare, infer, run, RAYO_FALSE};
