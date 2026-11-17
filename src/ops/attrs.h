/* Reading a node's attributes. */
#ifndef RAYO_OPS_ATTRS_H
#define RAYO_OPS_ATTRS_H

#include "ops/op.h"

typedef enum rayo_activation {
  RAYO_ACT_NONE,
  RAYO_ACT_RELU,
  RAYO_ACT_GELU_ERF /* 0.5 x (1 + erf(x / sqrt 2)), as Keras' gelu */
} rayo_activation;

/* A size given as a whole number or a size symbol's name. */
typedef struct rayo_size_attr {
  rayo_i32 value;
  rayo_i32 symbol; /* >= 0: the symbol's current value instead */
} rayo_size_attr;

/* The size now: the number, or the symbol's current value. */
rayo_i32 rayo_size_value(rayo_size_attr a, const rayo_i32 *symbols);

/* Attribute `key` as an integer in [lo, hi], or `fallback` when absent. */
rayo_status rayo_attr_int(const rayo_op_spec *spec, const char *key, rayo_i32 fallback, rayo_i32 lo,
                          rayo_i32 hi, rayo_i32 *out, rayo_message *msg);

/* Attribute `key` as a number, or `fallback` when absent. */
rayo_status rayo_attr_float(const rayo_op_spec *spec, const char *key, float fallback, float *out,
                            rayo_message *msg);

/* Attribute `key` as a whole number (0 to RAYO_MAX_DIM) or a symbol's name;
 * `fallback` when absent. */
rayo_status rayo_attr_size(const rayo_op_spec *spec, const char *key, rayo_i32 fallback,
                           rayo_size_attr *out, rayo_message *msg);

/* "activation": "none" (or absent), "relu" or "gelu_erf". */
rayo_status rayo_attr_activation(const rayo_op_spec *spec, rayo_activation *out, rayo_message *msg);

/* Attribute `key` as a number in double, or `fallback` when absent. */
rayo_status rayo_attr_double(const rayo_op_spec *spec, const char *key, double fallback,
                             double *out, rayo_message *msg);

/* Whether attribute `key` is there. */
rayo_bool rayo_attr_has(const rayo_op_spec *spec, const char *key);

/* Attribute `key` as a string, or `fallback` when absent. */
rayo_status rayo_attr_string(const rayo_op_spec *spec, const char *key, const char *fallback,
                             const char **out, rayo_message *msg);

#endif
