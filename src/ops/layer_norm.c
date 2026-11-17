/* layer_norm: x [..., C] normalised over the innermost dimension, then
 * scaled by gamma and shifted by beta (float32 weights [C]). The arithmetic
 * follows TFLite's decomposition of Keras' LayerNormalization: mean and
 * variance (MEAN, SQUARED_DIFFERENCE), m = rsqrt(var + eps) * gamma, then
 * x * m + (beta - mean * m). */
#include "core/mathx.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct norm_params {
  const float *gamma;
  const float *beta;
  size_t width;
  float eps;
} norm_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  const rayo_weight *g = spec->inputs[1].weight, *b = spec->inputs[2].weight;
  norm_params *p;
  if (spec->inputs[0].weight || !g || !b || g->dtype != RAYO_DTYPE_F32 ||
      b->dtype != RAYO_DTYPE_F32 || g->count != b->count || g->count < 1) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: layer_norm takes an input and float32 gamma and beta of one size",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  p->gamma = g->data;
  p->beta = b->data;
  p->width = g->count;
  *params = p;
  return rayo_attr_float(spec, "epsilon", 1e-5f, &p->eps, msg);
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const norm_params *p = params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank < 1 ||
      rayo_shape_inner(&in[0].shape) != p->width) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "layer_norm: float32 [..., %lu]",
                     (unsigned long)p->width);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static void norm_row(const norm_params *p, const float *RAYO_RESTRICT x, float *RAYO_RESTRICT y) {
  const size_t c = p->width;
  const float inv_c = 1.0f / (float)c;
  float sum = 0.0f, sq = 0.0f, mean, rs;
  size_t i;
  for (i = 0; i < c; i++) sum += x[i];
  mean = sum * inv_c;
  for (i = 0; i < c; i++) sq += (x[i] - mean) * (x[i] - mean);
  rs = 1.0f / rayo_sqrtf(sq * inv_c + p->eps);
  for (i = 0; i < c; i++) {
    const float m = rs * p->gamma[i];
    y[i] = x[i] * m + (-mean * m + p->beta[i]);
  }
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const norm_params *p = params;
  const size_t rows = rayo_shape_outer(&in[0].shape), c = p->width;
  size_t r;
  (void)exec;
  for (r = 0; r < rows; r++) {
    norm_row(p, (const float *)in[0].data + r * c, (float *)out->data + r * c);
  }
  return RAYO_OK;
}

const rayo_op rayo_op_layer_norm = {"layer_norm", 3, 3, prepare, infer, run, RAYO_FALSE};
