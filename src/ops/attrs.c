#include "ops/attrs.h"

#include <string.h>

#include "core/message.h"

rayo_status rayo_attr_int(const rayo_op_spec *spec, const char *key, rayo_i32 fallback, rayo_i32 lo,
                          rayo_i32 hi, rayo_i32 *out, rayo_message *msg) {
  const rayo_json *v = rayo_json_get(spec->json, key);
  if (!v) {
    *out = fallback;
    return RAYO_OK;
  }
  if (!rayo_json_as_int(v, out) || *out < lo || *out > hi) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s must be a whole number from %ld to %ld",
                     spec->output, key, (long)lo, (long)hi);
  }
  return RAYO_OK;
}

rayo_status rayo_attr_float(const rayo_op_spec *spec, const char *key, float fallback, float *out,
                            rayo_message *msg) {
  const rayo_json *v = rayo_json_get(spec->json, key);
  if (!v) {
    *out = fallback;
    return RAYO_OK;
  }
  if (v->kind != RAYO_JSON_NUMBER || v->number > 3.0e38 || v->number < -3.0e38) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s must be a number", spec->output, key);
  }
  *out = (float)v->number;
  return RAYO_OK;
}

rayo_status rayo_attr_string(const rayo_op_spec *spec, const char *key, const char *fallback,
                             const char **out, rayo_message *msg) {
  const rayo_json *v = rayo_json_get(spec->json, key);
  if (!v) {
    *out = fallback;
    return RAYO_OK;
  }
  *out = rayo_json_as_string(v);
  if (!*out)
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s must be a string", spec->output, key);
  return RAYO_OK;
}

rayo_status rayo_attr_activation(const rayo_op_spec *spec, rayo_activation *out,
                                 rayo_message *msg) {
  const char *s;
  rayo_status st = rayo_attr_string(spec, "activation", "none", &s, msg);
  if (st != RAYO_OK) return st;
  if (strcmp(s, "none") == 0) {
    *out = RAYO_ACT_NONE;
  } else if (strcmp(s, "relu") == 0) {
    *out = RAYO_ACT_RELU;
  } else if (strcmp(s, "gelu_erf") == 0) {
    *out = RAYO_ACT_GELU_ERF;
  } else {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: activation %s", spec->output, s);
  }
  return RAYO_OK;
}

rayo_status rayo_attr_size(const rayo_op_spec *spec, const char *key, rayo_i32 fallback,
                           rayo_size_attr *out, rayo_message *msg) {
  const rayo_json *v = rayo_json_get(spec->json, key);
  const char *name;
  out->symbol = -1;
  out->value = fallback;
  if (!v) return RAYO_OK;
  name = rayo_json_as_string(v);
  if (name) {
    out->symbol = spec->symbol(spec->graph, name);
    if (out->symbol < 0) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s names no symbol %s", spec->output, key,
                       name);
    }
    return RAYO_OK;
  }
  return rayo_attr_int(spec, key, fallback, 0, 1 << 24, &out->value, msg);
}

rayo_i32 rayo_size_value(rayo_size_attr a, const rayo_i32 *symbols) {
  return a.symbol >= 0 ? symbols[a.symbol] : a.value;
}
