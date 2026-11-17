/* Every op rayo runs, one per family file. */
#ifndef RAYO_OPS_OPS_H
#define RAYO_OPS_OPS_H

#include "ops/op.h"

extern const rayo_op rayo_op_add;         /* ops/elementwise.c */
extern const rayo_op rayo_op_attention;   /* ops/attention.c */
extern const rayo_op rayo_op_cache_write; /* ops/sequence.c */
extern const rayo_op rayo_op_last_row;    /* ops/sequence.c */
extern const rayo_op rayo_op_layer_norm;  /* ops/layer_norm.c */
extern const rayo_op rayo_op_range;       /* ops/sequence.c */
extern const rayo_op rayo_op_mask;        /* ops/elementwise.c */
extern const rayo_op rayo_op_concat;      /* ops/concat.c */
extern const rayo_op rayo_op_conv1d;      /* ops/conv1d.c */
extern const rayo_op rayo_op_dense;       /* ops/dense.c */
extern const rayo_op rayo_op_gather;      /* ops/gather.c */
extern const rayo_op rayo_op_masked_max;  /* ops/pool.c */
extern const rayo_op rayo_op_masked_mean; /* ops/pool.c */
extern const rayo_op rayo_op_softmax;     /* ops/softmax.c */

#endif
