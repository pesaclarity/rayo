/* Ops that move values without computing: float32 or int32, any rank up to
 * RAYO_MAX_RANK.
 *
 * reshape: the same values in another shape. `shape` lists sizes (whole
 *   numbers or size symbols) and at most one -1, the size that makes the
 *   count match.
 * transpose: dimension i of the output is dimension perm[i] of the input.
 * slice: the block starting at `begin` (sizes) of `size` (sizes, or -1 for
 *   the rest of that dimension). */
#include <string.h>

#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

typedef struct sizes_params {
  rayo_sizes_attr a;
  rayo_sizes_attr b;
} sizes_params;

static rayo_status activation_only(const rayo_op_spec *spec, rayo_message *msg) {
  if (spec->inputs[0].weight) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: takes an activation", spec->output);
  }
  return RAYO_OK;
}

static rayo_bool four_bytes(rayo_dtype d) {
  return d == RAYO_DTYPE_F32 || d == RAYO_DTYPE_I32;
}

static rayo_status reshape_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                   rayo_message *msg) {
  sizes_params *p;
  rayo_status s = activation_only(spec, msg);
  rayo_i32 i, wild = 0;
  if (s != RAYO_OK) return s;
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_sizes(spec, "shape", -1, &p->a, msg);
  for (i = 0; s == RAYO_OK && i < p->a.count; i++) wild += p->a.items[i].value == -1;
  if (s == RAYO_OK && wild > 1) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: reshape takes at most one -1", spec->output);
  }
  return s;
}

static rayo_status reshape_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                                 rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const sizes_params *p = params;
  const size_t count = rayo_shape_count(&in[0].shape);
  size_t known = 1;
  rayo_i32 i, wild = -1;
  if (!four_bytes(in[0].dtype)) return rayo_fail(msg, RAYO_ERR_SHAPE, "reshape: a dtype");
  out->dtype = in[0].dtype;
  out->shape.rank = p->a.count;
  for (i = 0; i < p->a.count; i++) {
    const rayo_i32 v = rayo_size_value(p->a.items[i], symbols);
    out->shape.dims[i] = v;
    if (v == -1) {
      wild = i;
    } else {
      known *= (size_t)v;
    }
  }
  if (wild >= 0 && known > 0 && count % known == 0 && count / known <= (size_t)RAYO_MAX_DIM) {
    out->shape.dims[wild] = (rayo_i32)(count / known);
    known = count;
  }
  if (known != count) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "reshape: %lu values into %lu", (unsigned long)count,
                     (unsigned long)known);
  }
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status copy_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                            const rayo_exec *exec) {
  (void)params;
  (void)exec;
  memcpy(out->data, in[0].data, rayo_tensor_bytes(out));
  return RAYO_OK;
}

static rayo_status transpose_prepare(const rayo_op_spec *spec, rayo_arena *arena,
                                     const void **params, rayo_message *msg) {
  sizes_params *p;
  rayo_status s = activation_only(spec, msg);
  rayo_i32 i, j;
  if (s != RAYO_OK) return s;
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_sizes(spec, "perm", 0, &p->a, msg);
  for (i = 0; s == RAYO_OK && i < p->a.count; i++) {
    rayo_bool ok = p->a.items[i].symbol < 0 && p->a.items[i].value < p->a.count;
    for (j = 0; ok && j < i; j++) ok = p->a.items[j].value != p->a.items[i].value;
    if (!ok) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: perm must order the dimensions",
                       spec->output);
    }
  }
  return s;
}

static rayo_status transpose_infer(const void *params, const rayo_tensor *in,
                                   const rayo_i32 *symbols, rayo_tensor *out, size_t *scratch,
                                   rayo_message *msg) {
  const sizes_params *p = params;
  rayo_i32 i;
  (void)symbols;
  if (!four_bytes(in[0].dtype) || in[0].shape.rank != p->a.count) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "transpose: an input of rank %ld", (long)p->a.count);
  }
  out->dtype = in[0].dtype;
  out->shape.rank = p->a.count;
  for (i = 0; i < p->a.count; i++) out->shape.dims[i] = in[0].shape.dims[p->a.items[i].value];
  *scratch = 0;
  return RAYO_OK;
}

/* Strides of `s` as rank 4 (ones in front), in elements. */
static void strides4(const rayo_shape *s, size_t *st) {
  rayo_i32 i;
  size_t n = 1;
  for (i = RAYO_MAX_RANK - 1; i >= 0; i--) {
    const rayo_i32 k = i - (RAYO_MAX_RANK - s->rank);
    st[i] = n;
    if (k >= 0) n *= (size_t)s->dims[k];
  }
}

/* Copies the [d0, d1, d2, d3] block of `x` read with strides `sx` from
 * offset `at`, into `y` in order. */
static void gather4(const rayo_u32 *x, size_t at, const size_t *sx, const rayo_i32 *d,
                    rayo_u32 *y) {
  size_t i0, i1, i2, i3, o = 0;
  for (i0 = 0; i0 < (size_t)d[0]; i0++) {
    for (i1 = 0; i1 < (size_t)d[1]; i1++) {
      for (i2 = 0; i2 < (size_t)d[2]; i2++) {
        const rayo_u32 *row = x + at + i0 * sx[0] + i1 * sx[1] + i2 * sx[2];
        if (sx[3] == 1) {
          memcpy(y + o, row, (size_t)d[3] * sizeof *y);
          o += (size_t)d[3];
        } else {
          for (i3 = 0; i3 < (size_t)d[3]; i3++) y[o++] = row[i3 * sx[3]];
        }
      }
    }
  }
}

static rayo_status transpose_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                                 const rayo_exec *exec) {
  const sizes_params *p = params;
  size_t st[RAYO_MAX_RANK], sx[RAYO_MAX_RANK];
  rayo_i32 d[RAYO_MAX_RANK], i;
  const rayo_i32 pad = RAYO_MAX_RANK - p->a.count;
  (void)exec;
  strides4(&in[0].shape, st);
  for (i = 0; i < RAYO_MAX_RANK; i++) {
    d[i] = i < pad ? 1 : out->shape.dims[i - pad];
    sx[i] = i < pad ? 0 : st[pad + p->a.items[i - pad].value];
  }
  gather4(in[0].data, 0, sx, d, out->data);
  return RAYO_OK;
}

static rayo_status slice_prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                                 rayo_message *msg) {
  sizes_params *p;
  rayo_status s = activation_only(spec, msg);
  if (s != RAYO_OK) return s;
  p = rayo_arena_alloc(arena, sizeof *p);
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *params = p;
  s = rayo_attr_sizes(spec, "begin", 0, &p->a, msg);
  if (s == RAYO_OK) s = rayo_attr_sizes(spec, "size", -1, &p->b, msg);
  if (s == RAYO_OK && p->a.count != p->b.count) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: begin and size of one rank", spec->output);
  }
  return s;
}

static rayo_status slice_infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                               rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const sizes_params *p = params;
  rayo_i32 i;
  if (!four_bytes(in[0].dtype) || in[0].shape.rank != p->a.count) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "slice: an input of rank %ld", (long)p->a.count);
  }
  out->dtype = in[0].dtype;
  out->shape.rank = p->a.count;
  for (i = 0; i < p->a.count; i++) {
    const rayo_i32 n = in[0].shape.dims[i], at = rayo_size_value(p->a.items[i], symbols);
    rayo_i32 size = rayo_size_value(p->b.items[i], symbols);
    if (size == -1 && at <= n) size = n - at;
    if (at > n || size < 0 || size > n - at) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "slice: %ld at %ld of %ld", (long)size, (long)at,
                       (long)n);
    }
    out->shape.dims[i] = size;
  }
  *scratch = 0;
  return RAYO_OK;
}

static rayo_status slice_run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                             const rayo_exec *exec) {
  const sizes_params *p = params;
  size_t st[RAYO_MAX_RANK], at = 0;
  rayo_i32 d[RAYO_MAX_RANK], i;
  const rayo_i32 pad = RAYO_MAX_RANK - p->a.count;
  strides4(&in[0].shape, st);
  for (i = 0; i < RAYO_MAX_RANK; i++) {
    d[i] = i < pad ? 1 : out->shape.dims[i - pad];
    if (i >= pad) at += (size_t)rayo_size_value(p->a.items[i - pad], exec->symbols) * st[i];
  }
  gather4(in[0].data, at, st, d, out->data);
  return RAYO_OK;
}

const rayo_op rayo_op_reshape = {"reshape",     1,        1,         reshape_prepare,
                                 reshape_infer, copy_run, RAYO_FALSE};
const rayo_op rayo_op_transpose = {"transpose",   1,         1, transpose_prepare, transpose_infer,
                                   transpose_run, RAYO_FALSE};
const rayo_op rayo_op_slice = {"slice", 1, 1, slice_prepare, slice_infer, slice_run, RAYO_FALSE};
