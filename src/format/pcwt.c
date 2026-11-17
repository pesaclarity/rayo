#include "format/pcwt.h"

#include <string.h>

#include "core/message.h"

static const uint8_t magic[4] = {'P', 'C', 'W', 'T'};

static uint32_t read_u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* [offset, offset + bytes) inside `data_size`, with offset 4-aligned. */
static bool in_data(int64_t offset, int64_t bytes, size_t data_size) {
  return offset >= 0 && bytes >= 0 && offset % 4 == 0 && (uint64_t)offset <= data_size &&
         (uint64_t)bytes <= data_size - (uint64_t)offset;
}

static rayo_status read_shape(const rayo_json *v, rayo_shape *shape, int64_t *count,
                              const char *name, rayo_message *msg) {
  if (!v || v->kind != RAYO_JSON_ARRAY || v->length > RAYO_MAX_RANK) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad shape", name);
  }
  shape->rank = (int32_t)v->length;
  int64_t n = 1;
  for (size_t i = 0; i < v->length; i++) {
    int64_t d;
    if (!rayo_json_as_int(v->items[i], &d) || d < 0 || d > ((int64_t)1 << 32)) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad dimension", name);
    }
    shape->dims[i] = d;
    if (d != 0 && n > ((int64_t)1 << 40) / d) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: too large", name);
    }
    n *= d;
  }
  *count = n;
  return RAYO_OK;
}

static rayo_status read_quantization(const rayo_json *q, const uint8_t *data, size_t data_size,
                                     rayo_weight *w, int64_t *end, rayo_message *msg) {
  int64_t zero, s_offset, s_bytes, axis = -1;
  const rayo_json *axis_v = rayo_json_get(q, "axis");
  if (!rayo_json_as_int(rayo_json_get(q, "zero_point"), &zero) || zero < -128 || zero > 127 ||
      !rayo_json_as_int(rayo_json_get(q, "scales_offset"), &s_offset) ||
      !rayo_json_as_int(rayo_json_get(q, "scales_bytes"), &s_bytes) || !axis_v ||
      (axis_v->kind != RAYO_JSON_NULL &&
       (!rayo_json_as_int(axis_v, &axis) || axis < 0 || axis >= w->shape.rank))) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: bad quantization", w->name);
  }
  int64_t n = axis < 0 ? 1 : w->shape.dims[axis];
  if (s_bytes != n * 4 || !in_data(s_offset, s_bytes, data_size)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: scales outside the data", w->name);
  }
  w->scales = (const float *)(const void *)(data + s_offset);
  w->scale_count = n;
  w->zero_point = (int32_t)zero;
  w->axis = (int32_t)axis;
  int64_t e = s_offset + s_bytes + ((4 - s_bytes % 4) % 4);
  if (e > *end) *end = e;
  return RAYO_OK;
}

static rayo_status read_tensor(const rayo_json *t, const uint8_t *data, size_t data_size,
                               rayo_weight *w, int64_t *end, rayo_message *msg) {
  w->name = rayo_json_as_string(rayo_json_get(t, "name"));
  const char *dtype = rayo_json_as_string(rayo_json_get(t, "dtype"));
  if (!w->name || !dtype)
    return rayo_fail(msg, RAYO_ERR_FORMAT, "a tensor without a name or dtype");
  int64_t elem;
  if (strcmp(dtype, "float32") == 0) {
    w->dtype = RAYO_DTYPE_F32;
    elem = 4;
  } else if (strcmp(dtype, "int8") == 0) {
    w->dtype = RAYO_DTYPE_I8;
    elem = 1;
  } else {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "tensor %s: dtype %s", w->name, dtype);
  }
  rayo_status s = read_shape(rayo_json_get(t, "shape"), &w->shape, &w->count, w->name, msg);
  if (s != RAYO_OK) return s;
  int64_t offset, bytes;
  if (!rayo_json_as_int(rayo_json_get(t, "offset"), &offset) ||
      !rayo_json_as_int(rayo_json_get(t, "bytes"), &bytes) || bytes != w->count * elem ||
      !in_data(offset, bytes, data_size)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: outside the data or the wrong size",
                     w->name);
  }
  w->data = data + offset;
  w->axis = -1;
  int64_t e = offset + bytes + ((4 - bytes % 4) % 4);
  if (e > *end) *end = e;
  const rayo_json *q = rayo_json_get(t, "quantization");
  if ((q != NULL) != (w->dtype == RAYO_DTYPE_I8)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s: quantization only, and always, on int8",
                     w->name);
  }
  return q ? read_quantization(q, data, data_size, w, end, msg) : RAYO_OK;
}

rayo_status rayo_pcwt_read(const uint8_t *file, size_t size, rayo_arena *arena, rayo_pcwt *out,
                           rayo_message *msg) {
  memset(out, 0, sizeof *out);
  if (size < 12 || memcmp(file, magic, 4) != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "not a plain-weights (PCWT) file");
  }
  uint32_t version = read_u32(file + 4);
  if (version != RAYO_PCWT_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "weights format version %u, rayo reads %d", version,
                     RAYO_PCWT_VERSION);
  }
  uint32_t header_len = read_u32(file + 8);
  if (header_len > size - 12 || (12 + (size_t)header_len) % 4 != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "bad header length");
  }
  rayo_json *header;
  rayo_status s = rayo_json_parse((const char *)file + 12, header_len, arena, &header, msg);
  if (s != RAYO_OK) return s;
  int64_t hv;
  if (!rayo_json_as_int(rayo_json_get(header, "format_version"), &hv) || hv != version) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "header and file disagree on the format version");
  }
  const rayo_json *tensors = rayo_json_get(header, "tensors");
  if (!tensors || tensors->kind != RAYO_JSON_ARRAY) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "no tensors in the header");
  }
  const uint8_t *data = file + 12 + header_len;
  size_t data_size = size - 12 - header_len;
  rayo_weight *weights =
      rayo_arena_array(arena, tensors->length ? tensors->length : 1, sizeof *weights);
  if (!weights) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  int64_t end = 0;
  for (size_t i = 0; i < tensors->length; i++) {
    s = read_tensor(tensors->items[i], data, data_size, &weights[i], &end, msg);
    if (s != RAYO_OK) return s;
    for (size_t j = 0; j < i; j++) {
      if (strcmp(weights[j].name, weights[i].name) == 0) {
        return rayo_fail(msg, RAYO_ERR_FORMAT, "tensor %s twice", weights[i].name);
      }
    }
  }
  if ((uint64_t)end != data_size)
    return rayo_fail(msg, RAYO_ERR_FORMAT, "data after the last tensor");
  out->header = header;
  out->model = rayo_json_as_string(rayo_json_get(header, "model"));
  out->weights = weights;
  out->weight_count = tensors->length;
  return RAYO_OK;
}

const rayo_weight *rayo_pcwt_find(const rayo_pcwt *pcwt, const char *name) {
  for (size_t i = 0; i < pcwt->weight_count; i++) {
    if (strcmp(pcwt->weights[i].name, name) == 0) return &pcwt->weights[i];
  }
  return NULL;
}
