/* attention: multi-head scaled dot-product attention. q is [T, D]; k and v
 * are [S, D] with the T queries at positions S - T .. S - 1 (the keys of
 * earlier steps come from a cache). Each head h uses dimensions
 * h * D/heads .. (h + 1) * D/heads. With causal, query i sees only keys
 * at or before its own position. The output is [T, D].
 *
 * As TFLite runs Keras' attention: scores = (q · k) * scale, softmax, then
 * the weighted sum of v; a hidden key adds -1e9 there, which is exactly a
 * zero weight in float32, so hidden keys are skipped here.
 *
 * k and v may be int8 with a scale per row and group (a session's int8
 * cache, from cache_write): each group's products are summed in float32
 * over the int8 values and then scaled, `scale_g * sum(q[e] * k8[e])`, and
 * the weighted sum adds `(p_j * scale_g) * v8[e]`. */
#include "core/mathx.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"
#include "ops/scratch.h"

typedef struct attention_params {
  rayo_i32 heads;
  rayo_i32 kv_heads;
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
  if (s == RAYO_OK) s = rayo_attr_int(spec, "kv_heads", p->heads, 1, p->heads, &p->kv_heads, msg);
  if (s == RAYO_OK && p->heads % p->kv_heads != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: heads must be a multiple of kv_heads",
                     spec->output);
  }
  if (s == RAYO_OK) s = rayo_attr_int(spec, "causal", 0, 0, 1, &p->causal, msg);
  if (s == RAYO_OK) s = rayo_attr_float(spec, "scale", 0.0f, &p->scale, msg);
  return s;
}

/* k or v: float32, or int8 whose groups hold whole heads or split them
 * evenly. */
static rayo_bool kv_ok(const rayo_tensor *t, rayo_i32 hd) {
  if (t->dtype == RAYO_DTYPE_F32) return RAYO_TRUE;
  return t->dtype == RAYO_DTYPE_I8 && t->group > 0 && hd > 0 &&
         (t->group % hd == 0 || hd % t->group == 0);
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const attention_params *p = params;
  const rayo_shape *q = &in[0].shape, *k = &in[1].shape, *v = &in[2].shape;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || !kv_ok(&in[1], q->dims[1] / p->heads) ||
      !kv_ok(&in[2], q->dims[1] / p->heads) || in[1].dtype != in[2].dtype ||
      in[1].group != in[2].group || q->rank != 2 || k->rank != 2 || v->rank != 2 ||
      q->dims[1] % p->heads != 0 || k->dims[1] != q->dims[1] / p->heads * p->kv_heads ||
      v->dims[1] != k->dims[1] || v->dims[0] != k->dims[0] || k->dims[0] < q->dims[0]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE,
                     "attention: q [T, heads * H], k and v [S, kv_heads * H] with S >= T");
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = *q;
  *scratch = 0;
  rayo_scratch_add(scratch, (size_t)p->heads * (size_t)k->dims[0] * sizeof(float));
  return RAYO_OK;
}

/* One query, one head: out[0..hd) from keys and values 0 .. seen - 1, with
 * w (seen floats) for the weights. */
static void attend(const float *qi, const float *k, const float *v, size_t d, size_t hd,
                   size_t seen, float scale, float *w, float *out) {
  /* d: the stride of k and v's rows. */
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

/* attend with int8 k and v (rows of d values, groups of `group` with
 * `ks` and `vs` holding d / group scales a row), the head's first column
 * at `col`. */
static void attend_i8(const float *qi, const rayo_i8 *k, const float *ks, const rayo_i8 *v,
                      const float *vs, size_t d, size_t col, size_t group, size_t hd, size_t seen,
                      float scale, float *w, float *out) {
  const size_t sub = group < hd ? group : hd, per_row = d / group;
  float top = -RAYO_FLT_BIG, sum = 0.0f, inv;
  size_t j, e, b;
  for (j = 0; j < seen; j++) {
    const rayo_i8 *kj = k + j * d + col;
    float dot = 0.0f;
    for (b = 0; b < hd; b += sub) {
      float part = 0.0f;
      for (e = b; e < b + sub; e++) part += qi[e] * (float)kj[e];
      dot += part * ks[j * per_row + (col + b) / group];
    }
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
    const rayo_i8 *vj = v + j * d + col;
    for (b = 0; b < hd; b += sub) {
      const float ps = pj * vs[j * per_row + (col + b) / group];
      for (e = b; e < b + sub; e++) out[e] += ps * (float)vj[e];
    }
  }
}

typedef struct attention_task {
  const attention_params *p;
  const float *q;
  const rayo_tensor *k, *v;
  float *y, *w;
  size_t t, s, d, kd, hd;
  float scale;
} attention_task;

/* Heads [begin, end), every query. */
static void heads_of(void *context, size_t begin, size_t end) {
  const attention_task *a = context;
  const size_t group = (size_t)(a->p->heads / a->p->kv_heads);
  size_t i, h;
  for (h = begin; h < end; h++) {
    const size_t kh = h / group;
    for (i = 0; i < a->t; i++) {
      const size_t seen = a->p->causal ? a->s - a->t + i + 1 : a->s;
      const float *qi = a->q + i * a->d + h * a->hd;
      float *yi = a->y + i * a->d + h * a->hd;
      if (a->k->dtype == RAYO_DTYPE_F32 && a->v->dtype == RAYO_DTYPE_F32) {
        attend(qi, (const float *)a->k->data + kh * a->hd, (const float *)a->v->data + kh * a->hd,
               a->kd, a->hd, seen, a->scale, a->w + h * a->s, yi);
      } else {
        attend_i8(qi, a->k->data, a->k->scales, a->v->data, a->v->scales, a->kd, kh * a->hd,
                  (size_t)a->k->group, a->hd, seen, a->scale, a->w + h * a->s, yi);
      }
    }
  }
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const attention_params *p = params;
  attention_task a;
  a.p = p;
  a.t = (size_t)in[0].shape.dims[0];
  a.d = (size_t)in[0].shape.dims[1];
  a.s = (size_t)in[1].shape.dims[0];
  a.kd = (size_t)in[1].shape.dims[1];
  a.hd = a.d / (size_t)p->heads;
  a.scale = p->scale != 0.0f ? p->scale : 1.0f / rayo_sqrtf((float)a.hd);
  a.q = in[0].data;
  a.k = &in[1];
  a.v = &in[2];
  a.y = out->data;
  a.w = exec->scratch;
  rayo_pool_run(exec->pool, (size_t)p->heads, 1, heads_of, &a);
  return RAYO_OK;
}

rayo_i32 rayo_attention_kv_heads(const void *params) {
  return ((const attention_params *)params)->kv_heads;
}

const rayo_op rayo_op_attention = {"attention", 3, 3, prepare, infer, run, RAYO_FALSE};
