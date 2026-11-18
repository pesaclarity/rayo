/* matmul: a [..., M, K] times b [..., K, N] to [..., M, N], float32, summed
 * in float32 in order. `transpose_a` / `transpose_b` 1 read a as [..., K, M]
 * or b as [..., N, K]. The dimensions before the last two broadcast as
 * NumPy's do. b may be a float32 weight. */
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct matmul_params {
  rayo_i32 ta;
  rayo_i32 tb;
} matmul_params;

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  matmul_params *p;
  rayo_status s;
  if (spec->inputs[0].weight ||
      (spec->inputs[1].weight && spec->inputs[1].weight->dtype != RAYO_DTYPE_F32)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "node %s: matmul takes an activation and an activation or float32 weight",
                     spec->output);
  }
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_int(spec, "transpose_a", 0, 0, 1, &p->ta, msg);
  if (s == RAYO_OK) s = rayo_attr_int(spec, "transpose_b", 0, 0, 1, &p->tb, msg);
  return s;
}

/* The batch dimensions of `s` (all but the last two) as rank 2, ones in
 * front. */
static void batch2(const rayo_shape *s, rayo_i32 *d) {
  rayo_i32 i, k = 2 - (s->rank - 2);
  for (i = 0; i < 2; i++) d[i] = i < k ? 1 : s->dims[i - k];
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const matmul_params *p = params;
  const rayo_shape *a = &in[0].shape, *b = &in[1].shape;
  rayo_i32 ba[2], bb[2], m, k, kb, n, i;
  const rayo_i32 rank = a->rank > b->rank ? a->rank : b->rank;
  (void)symbols;
  if (in[0].dtype != RAYO_DTYPE_F32 || in[1].dtype != RAYO_DTYPE_F32 || a->rank < 2 ||
      b->rank < 2) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "matmul: float32 inputs of rank 2 or more");
  }
  m = a->dims[a->rank - (p->ta ? 1 : 2)];
  k = a->dims[a->rank - (p->ta ? 2 : 1)];
  kb = b->dims[b->rank - (p->tb ? 1 : 2)];
  n = b->dims[b->rank - (p->tb ? 2 : 1)];
  batch2(a, ba);
  batch2(b, bb);
  if (k != kb) return rayo_fail(msg, RAYO_ERR_SHAPE, "matmul: %ld by %ld", (long)k, (long)kb);
  out->dtype = RAYO_DTYPE_F32;
  out->shape.rank = rank;
  for (i = 0; i < 2; i++) {
    if (ba[i] != bb[i] && ba[i] != 1 && bb[i] != 1) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "matmul: batch sizes %ld and %ld", (long)ba[i],
                       (long)bb[i]);
    }
    if (i >= 2 - (rank - 2)) out->shape.dims[i - (2 - (rank - 2))] = ba[i] == 1 ? bb[i] : ba[i];
  }
  out->shape.dims[rank - 2] = m;
  out->shape.dims[rank - 1] = n;
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const matmul_params *p = params;
  const rayo_shape *sa = &in[0].shape, *sb = &in[1].shape, *so = &out->shape;
  const size_t m = (size_t)so->dims[so->rank - 2], n = (size_t)so->dims[so->rank - 1];
  const size_t k = (size_t)sa->dims[sa->rank - (p->ta ? 2 : 1)];
  /* Element strides of a's rows and columns, and of b's. */
  const size_t ar = p->ta ? 1 : k, ac = p->ta ? m : 1, br = p->tb ? 1 : n, bc = p->tb ? k : 1;
  rayo_i32 ba[2], bb[2], bo[2];
  const float *a = in[0].data, *b = in[1].data;
  float *y = out->data;
  size_t i0, i1, i, j, e;
  (void)exec;
  batch2(sa, ba);
  batch2(sb, bb);
  batch2(so, bo);
  for (i0 = 0; i0 < (size_t)bo[0]; i0++) {
    for (i1 = 0; i1 < (size_t)bo[1]; i1++) {
      const size_t xa = (ba[0] == 1 ? 0 : i0) * (size_t)ba[1] + (ba[1] == 1 ? 0 : i1);
      const size_t xb = (bb[0] == 1 ? 0 : i0) * (size_t)bb[1] + (bb[1] == 1 ? 0 : i1);
      const float *pa = a + xa * m * k, *pb = b + xb * k * n;
      float *py = y + (i0 * (size_t)bo[1] + i1) * m * n;
      for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
          float acc = 0.0f;
          for (e = 0; e < k; e++) acc += pa[i * ar + e * ac] * pb[e * br + j * bc];
          py[i * n + j] = acc;
        }
      }
    }
  }
  return RAYO_OK;
}

const rayo_op rayo_op_matmul = {"matmul", 2, 2, prepare, infer, run, RAYO_FALSE};
