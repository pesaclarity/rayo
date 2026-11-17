/* rayo/config.h: rayo's compile-time options, each with its default. To
 * change one, define it when building rayo and every program that includes
 * its headers, such as -DRAYO_MAX_RANK=6.
 *
 * Everything else a model may hold (nodes, inputs, outputs, states, size
 * symbols, a node's inputs, a kernel's size, a convolution's dilation) is
 * read from the file, bounded only by the checks that keep every count in
 * rayo_i32 and every tensor below 2^31 bytes. */
#ifndef RAYO_CONFIG_H
#define RAYO_CONFIG_H

/* The most dimensions a tensor may have. rayo_shape holds this many, so it
 * is part of the C API; the Dart package's bindings assume 4. */
#ifndef RAYO_MAX_RANK
#define RAYO_MAX_RANK 4
#endif

/* The largest dimension, or size symbol, a model file may give, and the
 * largest size attribute: 2^24 by default. At most 2^31 - 1. */
#ifndef RAYO_MAX_DIM
#define RAYO_MAX_DIM 16777216L
#endif

/* The alignment, in bytes, of the model's weights and of every buffer a
 * session plans: a power of two, at least 16 (the SIMD kernels' loads). */
#ifndef RAYO_ALIGN
#define RAYO_ALIGN 64
#endif

/* How deeply the JSON in a model's header may nest. The parser recurses
 * once per level, so this bounds its stack. */
#ifndef RAYO_JSON_MAX_DEPTH
#define RAYO_JSON_MAX_DEPTH 32
#endif

/* The most inputs an op of a program's own (rayo/op.h) may take. Built-in
 * ops read any number a file gives. */
#ifndef RAYO_OP_MAX_INPUTS
#define RAYO_OP_MAX_INPUTS 16
#endif

/* Compile-time checks: a negative array size fails the build. */
typedef char rayo_check_rank[RAYO_MAX_RANK >= 1 && RAYO_MAX_RANK <= 16 ? 1 : -1];
typedef char rayo_check_dim[RAYO_MAX_DIM >= 1 && RAYO_MAX_DIM <= 2147483647L ? 1 : -1];
typedef char rayo_check_op_inputs[RAYO_OP_MAX_INPUTS >= 1 ? 1 : -1];
typedef char rayo_check_align[RAYO_ALIGN >= 16 && (RAYO_ALIGN & (RAYO_ALIGN - 1)) == 0 ? 1 : -1];

#endif
