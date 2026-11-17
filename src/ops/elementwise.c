/* Elementwise ops, float32.
 *
 * Binary (add, sub, mul, div, maximum, minimum, squared_difference): two
 * tensors broadcast as NumPy does (shapes aligned at the innermost
 * dimension; a dimension of 1, or a missing one, repeats). Either input may
 * be a float32 weight, such as a bias or a scalar.
 *
 * Unary (neg, abs, square, sqrt, rsqrt, exp, log, tanh, sigmoid, relu,
 * relu6, gelu_erf, gelu_tanh): one activation, the same shape out. */
#include <math.h>

#include "core/mathx.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef enum binary_kind {
  BIN_ADD,
  BIN_SUB,
  BIN_MUL,
  BIN_DIV,
  BIN_MAX,
  BIN_MIN,
  BIN_SQDIFF
} binary_kind;

static const binary_kind binary_kinds[] = {BIN_ADD, BIN_SUB, BIN_MUL,   BIN_DIV,
                                           BIN_MAX, BIN_MIN, BIN_SQDIFF};

static rayo_status binary_prepare(const rayo_op_spec *spec, const binary_kind *kind,
                                  const void **params, rayo_message *msg) {
  int i;
  for (i = 0; i < 2; i++) {
    if (spec->inputs[i].weight && spec->inputs[i].weight->dtype != RAYO_DTYPE_F32) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: elementwise weights must be float32",
                       spec->output);
    }
  }
  *params = kind;
  return RAYO_OK;
}

#define BINARY_PREPARE(fn, k)                                                             \
  static rayo_status fn(const rayo_op_spec *spec, rayo_arena *arena, const void **params, \
                        rayo_message *msg) {                                              \
    (void)arena;                                                                          \
    return binary_prepare(spec, &binary_kinds[k], params, msg);                           \
  }

BINARY_PREPARE(add_prepare, BIN_ADD)
BINARY_PREPARE(sub_prepare, BIN_SUB)
BINARY_PREPARE(mul_prepare, BIN_MUL)
BINARY_PREPARE(div_prepare, BIN_DIV)
BINARY_PREPARE(max_prepare, BIN_MAX)
BINARY_PREPARE(min_prepare, BIN_MIN)
BINARY_PREPARE(sqdiff_prepare, BIN_SQDIFF)

/* `s` as rank 4, ones in front. */
static void pad4(const rayo_shape *s, rayo_i32 *d) {
  rayo_i32 i, k = RAYO_MAX_RANK - s->rank;
  for (i = 0; i < RAYO_MAX_RANK; i++) d[i] = i < k ? 1 : s->dims[i - k];
}

static rayo_status binary_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                                rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  rayo_i32 a[RAYO_MAX_RANK], b[RAYO_MAX_RANK], i;
  const rayo_i32 rank = in[0].shape.rank > in[1].shape.rank ? in[0].shape.rank : in[1].shape.rank;
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "elementwise: float32 inputs");
  }
  pad4(&in[0].shape, a);
  pad4(&in[1].shape, b);
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = rank;
  for (i = 0; i < RAYO_MAX_RANK; i++) {
    if (a[i] != b[i] && a[i] != 1 && b[i] != 1) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "elementwise: sizes %ld and %ld do not broadcast",
                       (long)a[i], (long)b[i]);
    }
    if (i >= RAYO_MAX_RANK - rank) {
      out->shape.dims[i - (RAYO_MAX_RANK - rank)] = a[i] == 1 ? b[i] : a[i];
    }
  }
  *scratch = 0;
  return RAYO_OK;
}

/* y[i] = a[i * sa] op b[i * sb] for i < n; sa and sb are 0 or 1. */
static void apply(binary_kind k, const float *a, size_t sa, const float *b, size_t sb, float *y,
                  size_t n) {
  size_t i;
  float d;
  switch (k) {
    case BIN_ADD:
      for (i = 0; i < n; i++) y[i] = a[i * sa] + b[i * sb];
      break;
    case BIN_SUB:
      for (i = 0; i < n; i++) y[i] = a[i * sa] - b[i * sb];
      break;
    case BIN_MUL:
      for (i = 0; i < n; i++) y[i] = a[i * sa] * b[i * sb];
      break;
    case BIN_DIV:
      for (i = 0; i < n; i++) y[i] = a[i * sa] / b[i * sb];
      break;
    case BIN_MAX:
      for (i = 0; i < n; i++) y[i] = a[i * sa] > b[i * sb] ? a[i * sa] : b[i * sb];
      break;
    case BIN_MIN:
      for (i = 0; i < n; i++) y[i] = a[i * sa] < b[i * sb] ? a[i * sa] : b[i * sb];
      break;
    case BIN_SQDIFF:
      for (i = 0; i < n; i++) {
        d = a[i * sa] - b[i * sb];
        y[i] = d * d;
      }
      break;
  }
}

static rayo_status binary_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                              const rayo_exec *exec) {
  const binary_kind k = *(const binary_kind *)params;
  rayo_i32 a[RAYO_MAX_RANK], b[RAYO_MAX_RANK], y[RAYO_MAX_RANK];
  size_t sa[RAYO_MAX_RANK], sb[RAYO_MAX_RANK], na = 1, nb = 1, i0, i1, i2, at = 0;
  const float *pa = in[0].data, *pb = in[1].data;
  float *py = out->data;
  rayo_i32 i;
  (void)exec;
  pad4(&in[0].shape, a);
  pad4(&in[1].shape, b);
  pad4(&out->shape, y);
  if (rayo_shape_equal(&in[0].shape, &in[1].shape)) {
    apply(k, pa, 1, pb, 1, py, rayo_shape_count(&out->shape));
    return RAYO_OK;
  }
  /* Strides, 0 where a dimension repeats. */
  for (i = RAYO_MAX_RANK - 1; i >= 0; i--) {
    sa[i] = a[i] == 1 ? 0 : na;
    sb[i] = b[i] == 1 ? 0 : nb;
    na *= (size_t)a[i];
    nb *= (size_t)b[i];
  }
  for (i0 = 0; i0 < (size_t)y[0]; i0++) {
    for (i1 = 0; i1 < (size_t)y[1]; i1++) {
      for (i2 = 0; i2 < (size_t)y[2]; i2++) {
        const size_t oa = i0 * sa[0] + i1 * sa[1] + i2 * sa[2];
        const size_t ob = i0 * sb[0] + i1 * sb[1] + i2 * sb[2];
        apply(k, pa + oa, sa[3], pb + ob, sb[3], py + at, (size_t)y[3]);
        at += (size_t)y[3];
      }
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_add = {"add", 2, 2, add_prepare, binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_sub = {"sub", 2, 2, sub_prepare, binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_mul = {"mul", 2, 2, mul_prepare, binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_div = {"div", 2, 2, div_prepare, binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_maximum = {"maximum",    2,          2,         max_prepare,
                                 binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_minimum = {"minimum",    2,          2,         min_prepare,
                                 binary_infer, binary_run, RAYO_FALSE};
const rayo_op rayo_op_squared_difference = {
    "squared_difference", 2, 2, sqdiff_prepare, binary_infer, binary_run, RAYO_FALSE};

/* Unary ops. */

typedef enum unary_kind {
  UN_NEG,
  UN_ABS,
  UN_SQUARE,
  UN_SQRT,
  UN_RSQRT,
  UN_EXP,
  UN_LOG,
  UN_TANH,
  UN_SIGMOID,
  UN_RELU,
  UN_RELU6,
  UN_GELU_ERF,
  UN_GELU_TANH
} unary_kind;

static const unary_kind unary_kinds[] = {UN_NEG,   UN_ABS,      UN_SQUARE,   UN_SQRT,    UN_RSQRT,
                                         UN_EXP,   UN_LOG,      UN_TANH,     UN_SIGMOID, UN_RELU,
                                         UN_RELU6, UN_GELU_ERF, UN_GELU_TANH};

static float unary(unary_kind k, float x) {
  switch (k) {
    case UN_NEG:
      return -x;
    case UN_ABS:
      return x < 0.0f ? -x : x;
    case UN_SQUARE:
      return x * x;
    case UN_SQRT:
      return rayo_sqrtf(x);
    case UN_RSQRT:
      return 1.0f / rayo_sqrtf(x);
    case UN_EXP:
      return rayo_expf(x);
    case UN_LOG:
      return (float)log((double)x);
    case UN_TANH:
      return (float)tanh((double)x);
    case UN_SIGMOID:
      return (float)(1.0 / (1.0 + exp(-(double)x)));
    case UN_RELU:
      return x > 0.0f ? x : 0.0f;
    case UN_RELU6:
      return x > 0.0f ? (x < 6.0f ? x : 6.0f) : 0.0f;
    case UN_GELU_ERF:
      return 0.5f * x * (1.0f + rayo_erff(x * 0.70710678118654752f));
    case UN_GELU_TANH:
      return (float)(0.5 * x *
                     (1.0 + tanh(0.79788456080286535588 * (x + 0.044715 * (double)x * x * x))));
  }
  return x;
}

static rayo_status unary_prepare(const rayo_op_spec *spec, const unary_kind *kind,
                                 const void **params, rayo_message *msg) {
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: takes an activation", spec->output);
  }
  *params = kind;
  return RAYO_OK;
}

#define UNARY_OP(var, fn, name, k)                                                        \
  static rayo_status fn(const rayo_op_spec *spec, rayo_arena *arena, const void **params, \
                        rayo_message *msg) {                                              \
    (void)arena;                                                                          \
    return unary_prepare(spec, &unary_kinds[k], params, msg);                             \
  }                                                                                       \
  const rayo_op var = {name, 1, 1, fn, unary_infer, unary_run, RAYO_FALSE};

static rayo_status unary_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32) return rayo_fail(msg, RAYO_ERR_SHAPE, "a float32 input");
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status unary_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                             const rayo_exec *exec) {
  const unary_kind k = *(const unary_kind *)params;
  const float *x = in[0].data;
  float *y = out->data;
  const size_t n = rayo_shape_count(&out->shape);
  size_t i;
  (void)exec;
  for (i = 0; i < n; i++) y[i] = unary(k, x[i]);
  return RAYO_OK;
}

UNARY_OP(rayo_op_neg, neg_prepare, "neg", UN_NEG)
UNARY_OP(rayo_op_abs, abs_prepare, "abs", UN_ABS)
UNARY_OP(rayo_op_square, square_prepare, "square", UN_SQUARE)
UNARY_OP(rayo_op_sqrt, sqrt_prepare, "sqrt", UN_SQRT)
UNARY_OP(rayo_op_rsqrt, rsqrt_prepare, "rsqrt", UN_RSQRT)
UNARY_OP(rayo_op_exp, exp_prepare, "exp", UN_EXP)
UNARY_OP(rayo_op_log, log_prepare, "log", UN_LOG)
UNARY_OP(rayo_op_tanh, tanh_prepare, "tanh", UN_TANH)
UNARY_OP(rayo_op_sigmoid, sigmoid_prepare, "sigmoid", UN_SIGMOID)
UNARY_OP(rayo_op_relu, relu_prepare, "relu", UN_RELU)
UNARY_OP(rayo_op_relu6, relu6_prepare, "relu6", UN_RELU6)
UNARY_OP(rayo_op_gelu_erf, gelu_erf_prepare, "gelu_erf", UN_GELU_ERF)
UNARY_OP(rayo_op_gelu_tanh, gelu_tanh_prepare, "gelu_tanh", UN_GELU_TANH)
