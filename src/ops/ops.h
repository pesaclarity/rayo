/* Every op rayo runs, one per family file. */
#ifndef RAYO_OPS_OPS_H
#define RAYO_OPS_OPS_H

#include "ops/op.h"

extern const rayo_op rayo_op_add; /* ops/elementwise.c: binary */
extern const rayo_op rayo_op_sub;
extern const rayo_op rayo_op_mul;
extern const rayo_op rayo_op_div;
extern const rayo_op rayo_op_maximum;
extern const rayo_op rayo_op_minimum;
extern const rayo_op rayo_op_squared_difference;
extern const rayo_op rayo_op_neg; /* ops/elementwise.c: unary */
extern const rayo_op rayo_op_abs;
extern const rayo_op rayo_op_square;
extern const rayo_op rayo_op_sqrt;
extern const rayo_op rayo_op_rsqrt;
extern const rayo_op rayo_op_exp;
extern const rayo_op rayo_op_log;
extern const rayo_op rayo_op_tanh;
extern const rayo_op rayo_op_sigmoid;
extern const rayo_op rayo_op_relu;
extern const rayo_op rayo_op_relu6;
extern const rayo_op rayo_op_gelu_erf;
extern const rayo_op rayo_op_gelu_tanh;
extern const rayo_op rayo_op_matmul;        /* ops/matmul.c */
extern const rayo_op rayo_op_reduce;        /* ops/reduce.c */
extern const rayo_op rayo_op_reshape;       /* ops/shape.c */
extern const rayo_op rayo_op_slice;         /* ops/shape.c */
extern const rayo_op rayo_op_transpose;     /* ops/shape.c */
extern const rayo_op rayo_op_attention;     /* ops/attention.c */
extern const rayo_op rayo_op_cache_write;   /* ops/sequence.c */
extern const rayo_op rayo_op_last_row;      /* ops/sequence.c */
extern const rayo_op rayo_op_layer_norm;    /* ops/layer_norm.c */
extern const rayo_op rayo_op_range;         /* ops/sequence.c */
extern const rayo_op rayo_op_concat;        /* ops/concat.c */
extern const rayo_op rayo_op_conv1d;        /* ops/conv1d.c */
extern const rayo_op rayo_op_dense;         /* ops/dense.c */
extern const rayo_op rayo_op_gather;        /* ops/gather.c */
extern const rayo_op rayo_op_masked_reduce; /* ops/pool.c */
extern const rayo_op rayo_op_compare;       /* ops/compare.c */
extern const rayo_op rayo_op_custom;        /* ops/custom.c: a program's own op */
extern const rayo_op rayo_op_softmax;       /* ops/softmax.c */

#endif
