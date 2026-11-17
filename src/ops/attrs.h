/* Reading a node's attributes. */
#ifndef RAYO_OPS_ATTRS_H
#define RAYO_OPS_ATTRS_H

#include <stdbool.h>
#include <stdint.h>

#include "ops/op.h"

typedef enum rayo_activation {
  RAYO_ACT_NONE,
  RAYO_ACT_RELU,
} rayo_activation;

/* Attribute `key` as an integer in [lo, hi], or `fallback` when absent. */
rayo_status rayo_attr_int(const rayo_op_spec *spec, const char *key, int64_t fallback, int64_t lo,
                          int64_t hi, int64_t *out, rayo_message *msg);

/* Attribute `key` as a number, or `fallback` when absent. */
rayo_status rayo_attr_float(const rayo_op_spec *spec, const char *key, float fallback, float *out,
                            rayo_message *msg);

/* "activation": "none" (or absent) or "relu". */
rayo_status rayo_attr_activation(const rayo_op_spec *spec, rayo_activation *out, rayo_message *msg);

/* Attribute `key` as a string, or `fallback` when absent. */
rayo_status rayo_attr_string(const rayo_op_spec *spec, const char *key, const char *fallback,
                             const char **out, rayo_message *msg);

#endif
