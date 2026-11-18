/* The ops of decoder-only language models that the older families lack:
 * rms_norm, rope (rotary position embeddings) and swiglu (a SiLU-gated
 * product). Each is general: its sizes and constants come from the node's
 * attributes and weights, which a converter takes from the model's own
 * configuration (docs/OPS.md). */
#include <math.h>
#include <string.h>

#include "core/mathx.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"
#include "ops/scratch.h"

/* rms_norm: y = gamma * (x * rsqrt(mean(x²) + eps)) over the innermost
 * dimension, with the sum of squares in double. */

typedef struct rms_params {
  const float *gamma;
  size_t width;
  float eps;
  float offset; /* added to gamma: 0, or 1 for norms that store gamma - 1 */
} rms_params;

static rayo_status rms_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                               rayo_message *msg) {
  const rayo_weight *g = spec->inputs[1].weight;
  rms_params *p;
  rayo_status s;
  if (spec->inputs[0].weight || !g || g->dtype != RAYO_DTYPE_F32 || g->count < 1) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: rms_norm takes an input and float32 gamma",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  p->gamma = g->data;
  p->width = g->count;
  *params = p;
  s = rayo_attr_float(spec, "epsilon", 1e-6f, &p->eps, msg);
  if (s == RAYO_OK) s = rayo_attr_float(spec, "gamma_offset", 0.0f, &p->offset, msg);
  return s;
}

static rayo_status rms_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                             rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const rms_params *p = params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank < 1 ||
      rayo_shape_inner(&in[0].shape) != p->width) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "rms_norm: float32 [..., %lu]", (unsigned long)p->width);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status rms_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                           const rayo_exec *exec) {
  const rms_params *p = params;
  const size_t rows = rayo_shape_outer(&in[0].shape), c = p->width;
  size_t r, i;
  (void)exec;
  for (r = 0; r < rows; r++) {
    const float *x = (const float *)in[0].data + r * c;
    float *y = (float *)out->data + r * c;
    double sq = 0.0;
    float rs;
    for (i = 0; i < c; i++) sq += (double)x[i] * (double)x[i];
    rs = (float)(1.0 / sqrt((double)(float)(sq / (double)c) + (double)p->eps));
    for (i = 0; i < c; i++) {
      const float n = x[i] * rs;
      y[i] = (p->gamma[i] + p->offset) * n;
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_rms_norm = {"rms_norm", 2, 2, rms_prepare, rms_infer, rms_run, RAYO_FALSE};

/* rope: x [T, heads * D] at positions [T] (int32), each head's first
 * `rotary_dims` dimensions rotated by the angle position * freq[i]:
 * - "half" (the default; GPT-NeoX, Llama, Qwen): dimension i pairs with
 *   i + R/2;
 * - "interleaved" (GPT-J): dimension 2i pairs with 2i + 1.
 * freq[i] is the optional third input (a float32 weight [R/2]: a converter
 * writes it from the model's rope scaling), else theta^(-2i / R). cos and
 * sin are scaled by `attention_scaling` (default 1). */

typedef struct rope_params {
  rayo_i32 heads;
  rayo_i32 rotary; /* R; 0: the head's size */
  rayo_bool interleaved;
  double theta;
  float scaling;
  const float *freq; /* or NULL */
} rope_params;

static rayo_status rope_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                rayo_message *msg) {
  const rayo_weight *f = spec->input_count > 2 ? spec->inputs[2].weight : NULL;
  const char *style;
  rope_params *p;
  rayo_status s;
  if (spec->inputs[0].weight || spec->inputs[1].weight || (spec->input_count > 2 && !f) ||
      (f && (f->dtype != RAYO_DTYPE_F32 || f->shape.rank != 1))) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: rope takes x, int32 positions and float32 frequencies [R/2]",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  p->freq = f ? f->data : NULL;
  s = rayo_attr_int(spec, "heads", 1, 1, (rayo_i32)RAYO_MAX_DIM, &p->heads, msg);
  if (s == RAYO_OK)
    s = rayo_attr_int(spec, "rotary_dims", 0, 0, (rayo_i32)RAYO_MAX_DIM, &p->rotary, msg);
  if (s == RAYO_OK) s = rayo_attr_double(spec, "theta", 10000.0, &p->theta, msg);
  if (s == RAYO_OK) s = rayo_attr_float(spec, "attention_scaling", 1.0f, &p->scaling, msg);
  if (s == RAYO_OK) s = rayo_attr_string(spec, "style", "half", &style, msg);
  if (s != RAYO_OK) return s;
  if (strcmp(style, "half") != 0 && strcmp(style, "interleaved") != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: rope style %s", spec->output, style);
  }
  p->interleaved = strcmp(style, "interleaved") == 0;
  if (p->rotary % 2 != 0 || !(p->theta > 0.0)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: rope needs even rotary_dims and theta > 0",
                     spec->output);
  }
  if (f && p->rotary && f->count * 2 != (size_t)p->rotary) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: %lu frequencies for %ld rotary dimensions",
                     spec->output, (unsigned long)f->count, (long)p->rotary);
  }
  return RAYO_OK;
}

static size_t rope_dims(const rope_params *p, size_t head) {
  return p->rotary ? (size_t)p->rotary : head;
}

static rayo_status rope_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                              rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const rope_params *p = params;
  const rayo_shape *x = &in[0].shape;
  size_t head;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || x->rank != 2 || in[1].dtype != RAYO_DTYPE_I32 ||
      in[1].shape.rank != 1 || in[1].shape.dims[0] != x->dims[0] || x->dims[1] % p->heads != 0) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "rope: x float32 [T, heads * D] and int32 positions [T]");
  }
  head = (size_t)(x->dims[1] / p->heads);
  if (rope_dims(p, head) > head || rope_dims(p, head) % 2 != 0 ||
      (p->freq && in[2].weight->count * 2 != rope_dims(p, head))) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "rope: rotary dimensions do not fit a head of %lu",
                     (unsigned long)head);
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = *x;
  *scratch = 0;
  rayo_scratch_add(scratch, (size_t)x->dims[0] * rope_dims(p, head) * sizeof(float)); /* cos, sin */
  return RAYO_OK;
}

static rayo_status rope_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                            const rayo_exec *exec) {
  const rope_params *p = params;
  const size_t t = (size_t)in[0].shape.dims[0], d = (size_t)in[0].shape.dims[1];
  const size_t heads = (size_t)p->heads, head = d / heads, r = rope_dims(p, head), half = r / 2;
  const float *x = in[0].data;
  const rayo_i32 *pos = in[1].data;
  float *y = out->data, *cs = exec->scratch;
  size_t i, h, j;
  /* cos and sin of each position and frequency, once for all heads. The
   * angle is a float product, as the models compute it. */
  for (i = 0; i < t; i++) {
    for (j = 0; j < half; j++) {
      const float f =
          p->freq ? p->freq[j] : (float)(1.0 / pow(p->theta, (double)(2 * j) / (double)r));
      const float angle = (float)pos[i] * f;
      cs[i * r + j] = (float)cos((double)angle) * p->scaling;
      cs[i * r + half + j] = (float)sin((double)angle) * p->scaling;
    }
  }
  for (i = 0; i < t; i++) {
    const float *c = cs + i * r, *s = c + half;
    for (h = 0; h < heads; h++) {
      const float *xh = x + i * d + h * head;
      float *yh = y + i * d + h * head;
      for (j = 0; j < half; j++) {
        const size_t a = p->interleaved ? 2 * j : j, b = p->interleaved ? 2 * j + 1 : j + half;
        const float xa = xh[a], xb = xh[b];
        const float ac = xa * c[j], bs = xb * s[j], bc = xb * c[j], as = xa * s[j];
        yh[a] = ac - bs;
        yh[b] = bc + as;
      }
      for (j = r; j < head; j++) yh[j] = xh[j];
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_rope = {"rope", 2, 3, rope_prepare, rope_infer, rope_run, RAYO_FALSE};

/* swiglu: silu(gate) * up, for two float32 activations of one shape. */

static rayo_status swiglu_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                  rayo_message *msg) {
  (void)arena;
  *params = NULL;
  if (spec->inputs[0].weight || spec->inputs[1].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: swiglu takes two activations", spec->output);
  }
  return RAYO_OK;
}

static rayo_status swiglu_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                                rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  (void)params;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32 ||
      !rayo_shape_equal(&in[0].shape, &in[1].shape)) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "swiglu: gate and up float32, of one shape");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status swiglu_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                              const rayo_exec *exec) {
  const float *g = in[0].data, *u = in[1].data;
  float *y = out->data;
  const size_t n = rayo_shape_count(&out->shape);
  size_t i;
  (void)params;
  (void)exec;
  for (i = 0; i < n; i++) y[i] = rayo_silu(g[i]) * u[i];
  return RAYO_OK;
}

const rayo_op rayo_op_swiglu = {"swiglu",     2,          2,         swiglu_prepare,
                                swiglu_infer, swiglu_run, RAYO_FALSE};
