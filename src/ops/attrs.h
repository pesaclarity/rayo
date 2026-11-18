/* Reading a node's attributes. */
#ifndef RAYO_OPS_ATTRS_H
#define RAYO_OPS_ATTRS_H

#include "ops/op.h"

typedef enum rayo_activation {
  RAYO_ACT_NONE,
  RAYO_ACT_RELU,
  RAYO_ACT_GELU_ERF, /* 0.5 x (1 + erf(x / sqrt 2)), as Keras' gelu */
  RAYO_ACT_SILU      /* x / (1 + exp(-x)) */
} rayo_activation;

/* A size given as a whole number or a size symbol's name. */
typedef struct rayo_size_attr {
  rayo_i32 value;
  rayo_i32 symbol; /* >= 0: the symbol's current value instead */
} rayo_size_attr;

/* A list attribute of sizes (reshape's shape, slice's begin and size,
 * transpose's perm): at most RAYO_MAX_RANK of them. */
typedef struct rayo_sizes_attr {
  rayo_i32 count;
  rayo_size_attr items[RAYO_MAX_RANK];
} rayo_sizes_attr;

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

/* Attribute `key` as a list of at most RAYO_MAX_RANK sizes, each a whole
 * number from `lo` (-1 or 0) to RAYO_MAX_DIM or a size symbol's name. It
 * must be there. */
rayo_status rayo_attr_sizes(const rayo_op_spec *spec, const char *key, rayo_i32 lo,
                            rayo_sizes_attr *out, rayo_message *msg);

/* "activation": "none" (or absent), "relu", "gelu_erf" or "silu". */
rayo_status rayo_attr_activation(const rayo_op_spec *spec, rayo_activation *out, rayo_message *msg);

/* x / (1 + exp(-x)), in double, rounded once (SiLU, also called swish). */
float rayo_silu(float x);

/* Attribute `key` as a number in double, or `fallback` when absent. */
rayo_status rayo_attr_double(const rayo_op_spec *spec, const char *key, double fallback,
                             double *out, rayo_message *msg);

/* Whether attribute `key` is there. */
rayo_bool rayo_attr_has(const rayo_op_spec *spec, const char *key);

/* Attribute `key` as a string, or `fallback` when absent. */
rayo_status rayo_attr_string(const rayo_op_spec *spec, const char *key, const char *fallback,
                             const char **out, rayo_message *msg);

#endif
