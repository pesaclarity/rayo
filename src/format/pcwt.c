#include "format/pcwt.h"

#include <string.h>

#include "core/message.h"

static const rayo_u8 magic[4] = {'P', 'C', 'W', 'T'};

static rayo_u32 read_u32(const rayo_u8 *p) {
  return (rayo_u32)p[0] | (rayo_u32)p[1] << 8 | (rayo_u32)p[2] << 16 | (rayo_u32)p[3] << 24;
}

/* [offset, offset + bytes) inside `data_size`, with offset 4-aligned. */
static rayo_bool in_data(rayo_i32 offset, rayo_i32 bytes, size_t data_size) {
  return offset >= 0 && bytes >= 0 && offset % 4 == 0 && (size_t)offset <= data_size &&
         (size_t)bytes <= data_size - (size_t)offset;
}

/* The end of [offset, offset + bytes) padded to 4, for the "data ends after
 * the last tensor" check. */
static size_t padded_end(rayo_i32 offset, rayo_i32 bytes) {
  return (size_t)offset + (size_t)bytes + (size_t)((4 - bytes % 4) % 4);
}

static rayo_status read_shape(const rayo_json *v, rayo_shape *shape, size_t *count,
                              const char *name, rayo_message *msg) {
  size_t i, n = 1;
  if (!v || v->kind != RAYO_JSON_ARRAY || v->length > RAYO_MAX_RANK) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad shape", name);
  }
  shape->rank = (rayo_i32)v->length;
  for (i = 0; i < v->length; i++) {
    rayo_i32 d;
    if (!rayo_json_as_int(v->items[i], &d) || d < 0) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad dimension", name);
    }
    shape->dims[i] = d;
    /* Keep every element count, and its bytes, below 2^31. */
    if (d != 0 && n > (size_t)(RAYO_I32_MAX / 4) / (size_t)d) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: too large", name);
    }
    n *= (size_t)d;
  }
  *count = n;
  return RAYO_OK;
}

static rayo_status read_quantization(const rayo_json *q, const rayo_u8 *data, size_t data_size,
                                     rayo_weight *w, size_t *end, rayo_message *msg) {
  rayo_i32 zero, s_offset, s_bytes, axis = -1;
  rayo_i32 n;
  const rayo_json *axis_v = rayo_json_get(q, "axis");
  if (!rayo_json_as_int(rayo_json_get(q, "zero_point"), &zero) || zero < -128 || zero > 127 ||
      !rayo_json_as_int(rayo_json_get(q, "scales_offset"), &s_offset) ||
      !rayo_json_as_int(rayo_json_get(q, "scales_bytes"), &s_bytes) || !axis_v ||
      (axis_v->kind != RAYO_JSON_NULL &&
       (!rayo_json_as_int(axis_v, &axis) || axis < 0 || axis >= w->shape.rank))) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad quantization", w->name);
  }
  n = axis < 0 ? 1 : w->shape.dims[axis];
  if (s_bytes != n * 4 || !in_data(s_offset, s_bytes, data_size)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: scales outside the data", w->name);
  }
  w->scales = (const float *)(const void *)(data + s_offset);
  w->scale_count = (size_t)n;
  w->zero_point = zero;
  w->axis = axis;
  if (padded_end(s_offset, s_bytes) > *end) *end = padded_end(s_offset, s_bytes);
  return RAYO_OK;
}

/* int4 and int3: "quantization" is {"group_size": 32, "zero_point": 8 (int3:
 * 4), "scales_offset", "scales_bytes", and optionally "scale_dtype":
 * "float32" (the default) or "float16"}, a scale per group of the innermost
 * dimension. */
static rayo_status read_groups(const rayo_json *q, const rayo_u8 *data, size_t data_size,
                               rayo_weight *w, size_t *end, rayo_message *msg) {
  rayo_i32 group, zero, s_offset, s_bytes, inner;
  const rayo_i32 want_zero = w->dtype == RAYO_DTYPE_I4 ? 8 : 4;
  const char *scale_dtype;
  size_t scale_size = 4;
  if (!q || !rayo_json_as_int(rayo_json_get(q, "group_size"), &group) ||
      !rayo_json_as_int(rayo_json_get(q, "zero_point"), &zero) ||
      !rayo_json_as_int(rayo_json_get(q, "scales_offset"), &s_offset) ||
      !rayo_json_as_int(rayo_json_get(q, "scales_bytes"), &s_bytes) || w->shape.rank < 1) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad group quantization", w->name);
  }
  scale_dtype = rayo_json_as_string(rayo_json_get(q, "scale_dtype"));
  if (rayo_json_get(q, "scale_dtype") && (!scale_dtype || (strcmp(scale_dtype, "float32") != 0 &&
                                                           strcmp(scale_dtype, "float16") != 0))) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "tensor %s: scale_dtype float32 or float16",
                     w->name);
  }
  if (scale_dtype && strcmp(scale_dtype, "float16") == 0) scale_size = 2;
  inner = w->shape.dims[w->shape.rank - 1];
  if (group != RAYO_PCWT_INT4_GROUP || zero != want_zero || inner % group != 0) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "tensor %s: %s needs groups of %d along a multiple of it, zero point %d",
                     w->name, w->dtype == RAYO_DTYPE_I4 ? "int4" : "int3", RAYO_PCWT_INT4_GROUP,
                     (int)want_zero);
  }
  if ((size_t)s_bytes != w->count / (size_t)group * scale_size ||
      !in_data(s_offset, s_bytes, data_size) || (scale_size == 4 && s_offset % 4 != 0)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: scales outside the data", w->name);
  }
  w->group_scales = data + s_offset;
  w->scales_half = scale_size == 2;
  w->scales = scale_size == 4 ? (const float *)(const void *)(data + s_offset) : NULL;
  w->scale_count = (size_t)s_bytes / scale_size;
  w->zero_point = zero;
  w->group_size = group;
  if (padded_end(s_offset, s_bytes) > *end) *end = padded_end(s_offset, s_bytes);
  return RAYO_OK;
}

static rayo_status read_tensor(const rayo_json *t, const rayo_u8 *data, size_t data_size,
                               rayo_weight *w, size_t *end, rayo_message *msg) {
  const char *dtype;
  const rayo_json *q;
  size_t elem, need;
  rayo_i32 offset, bytes;
  rayo_status s;
  w->name = rayo_json_as_string(rayo_json_get(t, "name"));
  dtype = rayo_json_as_string(rayo_json_get(t, "dtype"));
  if (!w->name || !dtype)
    return rayo_fail(msg, RAYO_ERR_FORMAT, "a tensor without a name or dtype");
  if (strcmp(dtype, "float32") == 0) {
    w->dtype = RAYO_DTYPE_F32;
    elem = 4;
  } else if (strcmp(dtype, "int8") == 0) {
    w->dtype = RAYO_DTYPE_I8;
    elem = 1;
  } else if (strcmp(dtype, "int4") == 0) {
    w->dtype = RAYO_DTYPE_I4;
    elem = 0; /* two per byte */
  } else if (strcmp(dtype, "int3") == 0) {
    w->dtype = RAYO_DTYPE_I3;
    elem = 0; /* twelve bytes per group of 32 */
  } else {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "tensor %s: dtype %s", w->name, dtype);
  }
  s = read_shape(rayo_json_get(t, "shape"), &w->shape, &w->count, w->name, msg);
  if (s != RAYO_OK) return s;
  need = elem ? w->count * elem
              : (w->dtype == RAYO_DTYPE_I4
                     ? w->count / 2
                     : w->count / RAYO_PCWT_INT4_GROUP * RAYO_PCWT_INT3_GROUP_BYTES);
  if (!rayo_json_as_int(rayo_json_get(t, "offset"), &offset) ||
      !rayo_json_as_int(rayo_json_get(t, "bytes"), &bytes) || bytes < 0 || (size_t)bytes != need ||
      (!elem && w->count % (w->dtype == RAYO_DTYPE_I4 ? 2 : RAYO_PCWT_INT4_GROUP) != 0) ||
      !in_data(offset, bytes, data_size)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: outside the data or the wrong size",
                     w->name);
  }
  w->data = data + offset;
  w->axis = -1;
  if (padded_end(offset, bytes) > *end) *end = padded_end(offset, bytes);
  q = rayo_json_get(t, "quantization");
  if (w->dtype == RAYO_DTYPE_I4 || w->dtype == RAYO_DTYPE_I3) {
    return read_groups(q, data, data_size, w, end, msg);
  }
  if ((q != NULL) != (w->dtype == RAYO_DTYPE_I8)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: quantization only, and always, on int8",
                     w->name);
  }
  return q ? read_quantization(q, data, data_size, w, end, msg) : RAYO_OK;
}

static rayo_status read_tensors(const rayo_json *tensors, const rayo_u8 *data, size_t data_size,
                                rayo_weight *weights, size_t end, rayo_message *msg) {
  size_t i, j;
  for (i = 0; i < tensors->length; i++) {
    rayo_status s = read_tensor(tensors->items[i], data, data_size, &weights[i], &end, msg);
    if (s != RAYO_OK) return s;
    for (j = 0; j < i; j++) {
      if (strcmp(weights[j].name, weights[i].name) == 0) {
        return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s twice", weights[i].name);
      }
    }
  }
  if (end != data_size) return rayo_fail(msg, RAYO_ERR_FORMAT, "data after the last tensor");
  return RAYO_OK;
}

rayo_status rayo_pcwt_read(const rayo_u8 *file, size_t size, rayo_arena *arena, rayo_pcwt *out,
                           rayo_message *msg) {
  rayo_u32 version, header_len;
  rayo_json *header;
  rayo_i32 hv;
  const rayo_json *tensors;
  rayo_weight *weights;
  rayo_status s;
  memset(out, 0, sizeof *out);
  if (size < 12 || memcmp(file, magic, 4) != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "not a plain-weights (PCWT) file");
  }
  version = read_u32(file + 4);
  if (version != RAYO_PCWT_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "weights format version %lu, rayo reads %d",
                     (unsigned long)version, RAYO_PCWT_VERSION);
  }
  header_len = read_u32(file + 8);
  if (header_len > size - 12 || (12 + (size_t)header_len) % 4 != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "bad header length");
  }
  s = rayo_json_parse((const char *)file + 12, header_len, arena, &header, msg);
  if (s != RAYO_OK) return s;
  if (!rayo_json_as_int(rayo_json_get(header, "format_version"), &hv) || (rayo_u32)hv != version) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "header and file disagree on the format version");
  }
  tensors = rayo_json_get(header, "tensors");
  if (!tensors || tensors->kind != RAYO_JSON_ARRAY) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "no tensors in the header");
  }
  weights = (rayo_weight *)rayo_arena_array(arena, tensors->length ? tensors->length : 1,
                                            sizeof *weights);
  if (!weights) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  /* The tokeniser's table, {"offset", "bytes"} in the data, if any. */
  {
    const rayo_json *tok = rayo_json_get(header, "tokenizer");
    const size_t data_size = size - 12 - header_len;
    rayo_i32 offset, bytes;
    if (tok) {
      if (!rayo_json_as_int(rayo_json_get(tok, "offset"), &offset) ||
          !rayo_json_as_int(rayo_json_get(tok, "bytes"), &bytes) ||
          !in_data(offset, bytes, data_size) || bytes % 4 != 0) {
        return rayo_fail(msg, RAYO_ERR_FORMAT, "the tokeniser's table is outside the data");
      }
      out->tokenizer = file + 12 + header_len + offset;
      out->tokenizer_size = (size_t)bytes;
    }
  }
  s = read_tensors(
      tensors, file + 12 + header_len, size - 12 - header_len, weights,
      out->tokenizer ? (size_t)(out->tokenizer - file - 12 - header_len) + out->tokenizer_size : 0,
      msg);
  if (s != RAYO_OK) return s;
  out->header = header;
  out->model = rayo_json_as_string(rayo_json_get(header, "model"));
  out->weights = weights;
  out->weight_count = tensors->length;
  return RAYO_OK;
}

const rayo_weight *rayo_pcwt_find(const rayo_pcwt *pcwt, const char *name) {
  size_t i;
  for (i = 0; i < pcwt->weight_count; i++) {
    if (strcmp(pcwt->weights[i].name, name) == 0) return &pcwt->weights[i];
  }
  return NULL;
}
