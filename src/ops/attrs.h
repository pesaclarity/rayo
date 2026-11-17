/* Reading a node's attributes. */
#ifndef RAYO_OPS_ATTRS_H
#define RAYO_OPS_ATTRS_H

#include <stdbool.h>
#include <stdint.h>

#include "ops/op.h"

typedef enum rayo_activation {
  RAYO_ACT_NONE,
  RAYO_ACT_RELU,
  RAYO_ACT_GELU_ERF, /* 0.5 x (1 + erf(x / sqrt 2)), as Keras' gelu */
} rayo_activation;

/* A size given as a whole number or a size symbol's name. */
typedef struct rayo_size_attr {
  int64_t value;
  int32_t symbol; /* >= 0: the symbol's current value instead */
} rayo_size_attr;

static inline int64_t rayo_size_value(rayo_size_attr a, const int64_t *symbols) {
  return a.symbol >= 0 ? symbols[a.symbol] : a.value;
}

/* Attribute `key` as an integer in [lo, hi], or `fallback` when absent. */
rayo_status rayo_attr_int(const rayo_op_spec *spec, const char *key, int64_t fallback, int64_t lo,
                          int64_t hi, int64_t *out, rayo_message *msg);

/* Attribute `key` as a number, or `fallback` when absent. */
rayo_status rayo_attr_float(const rayo_op_spec *spec, const char *key, float fallback, float *out,
                            rayo_message *msg);

/* Attribute `key` as a whole number (0 to 1 << 24) or a symbol's name;
 * `fallback` when absent. */
rayo_status rayo_attr_size(const rayo_op_spec *spec, const char *key, int64_t fallback,
                           rayo_size_attr *out, rayo_message *msg);

/* "activation": "none" (or absent), "relu" or "gelu_erf". */
rayo_status rayo_attr_activation(const rayo_op_spec *spec, rayo_activation *out, rayo_message *msg);

/* Attribute `key` as a string, or `fallback` when absent. */
rayo_status rayo_attr_string(const rayo_op_spec *spec, const char *key, const char *fallback,
                             const char **out, rayo_message *msg);

#endif
