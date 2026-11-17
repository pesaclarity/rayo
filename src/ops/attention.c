/* attention: multi-head scaled dot-product attention. q is [T, D]; k and v
 * are [S, D] with the T queries at positions S - T .. S - 1 (the keys of
 * earlier steps come from a cache). Each head h uses dimensions
 * h * D/heads .. (h + 1) * D/heads. With causal, query i sees only keys
 * at or before its own position. The output is [T, D].
 *
 * As TFLite runs Keras' attention: scores = (q · k) * scale, softmax, then
 * the weighted sum of v; a hidden key adds -1e9 there, which is exactly a
 * zero weight in float32, so hidden keys are skipped here. */
#include "core/mathx.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"
#include "ops/scratch.h"

typedef struct attention_params {
  rayo_i32 heads;
  rayo_i32 causal;
  float scale; /* 0: 1 / sqrt(D / heads) */
} attention_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  attention_params *p;
  rayo_status s;
  int i;
  for (i = 0; i < 3; i++) {
    if (spec->inputs[i].weight) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: attention takes q, k and v activations",
                       spec->output);
    }
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_int(spec, "heads", 1, 1, (rayo_i32)RAYO_MAX_DIM, &p->heads, msg);
  if (s == RAYO_OK) s = rayo_attr_int(spec, "causal", 0, 0, 1, &p->causal, msg);
  if (s == RAYO_OK) s = rayo_attr_float(spec, "scale", 0.0f, &p->scale, msg);
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const attention_params *p = params;
  const rayo_shape *q = &in[0].shape, *k = &in[1].shape, *v = &in[2].shape;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32 ||
      in[2].dtype != RAYO_DTYPE_F32 || q->rank != 2 || k->rank != 2 || v->rank != 2 ||
      k->dims[1] != q->dims[1] || v->dims[1] != q->dims[1] || v->dims[0] != k->dims[0] ||
      k->dims[0] < q->dims[0] || q->dims[1] % p->heads != 0) {
    return rayo_fail(msg, RAYO_ERR_SHAPE,
                     "attention: q [T, D], k and v [S, D] with S >= T and D a multiple of heads");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = *q;
  *scratch = 0;
  rayo_scratch_add(scratch, (size_t)k->dims[0] * sizeof(float));
  return RAYO_OK;
}

/* One query, one head: out[0..hd) from keys and values 0 .. seen - 1, with
 * w (seen floats) for the weights. */
static void attend(const float *qi, const float *k, const float *v, size_t d, size_t hd,
                   size_t seen, float scale, float *w, float *out) {
  float top = -RAYO_FLT_BIG, sum = 0.0f, inv;
  size_t j, e;
  for (j = 0; j < seen; j++) {
    const float *kj = k + j * d;
    float dot = 0.0f;
    for (e = 0; e < hd; e++) dot += qi[e] * kj[e];
    w[j] = dot * scale;
    top = w[j] > top ? w[j] : top;
  }
  for (j = 0; j < seen; j++) {
    w[j] = rayo_expf(w[j] - top);
    sum += w[j];
  }
  inv = 1.0f / sum;
  for (e = 0; e < hd; e++) out[e] = 0.0f;
  for (j = 0; j < seen; j++) {
    const float pj = w[j] * inv;
    const float *vj = v + j * d;
    for (e = 0; e < hd; e++) out[e] += pj * vj[e];
  }
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const attention_params *p = params;
  const size_t t = (size_t)in[0].shape.dims[0], d = (size_t)in[0].shape.dims[1];
  const size_t s = (size_t)in[1].shape.dims[0], heads = (size_t)p->heads, hd = d / heads;
  const float scale = p->scale != 0.0f ? p->scale : 1.0f / rayo_sqrtf((float)hd);
  const float *q = in[0].data, *k = in[1].data, *v = in[2].data;
  float *y = out->data;
  size_t i, h;
  for (i = 0; i < t; i++) {
    const size_t seen = p->causal ? s - t + i + 1 : s;
    for (h = 0; h < heads; h++) {
      attend(q + i * d + h * hd, k + h * hd, v + h * hd, d, hd, seen, scale, (float *)exec->scratch,
             y + i * d + h * hd);
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_attention = {"attention", 3, 3, prepare, infer, run, RAYO_FALSE};
