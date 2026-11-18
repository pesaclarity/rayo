/* rayo/config.h: rayo's compile-time options, each with its default. To
 * change one, define it when building rayo and every program that includes
 * its headers, such as -DRAYO_MAX_RANK=6.
 *
 * Model files and text are untrusted (docs/SECURITY.md): every count, size
 * and offset they give is checked before it is used, so that no tensor
 * reaches 2^31 bytes and no count leaves rayo_i32, and the limits below
 * bound what one file or text can make rayo allocate or compute. Going over
 * a limit fails with RAYO_ERR_LIMIT, naming it; raise one for a model that
 * needs more. */
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

/* The most bytes of a model file's JSON header (64 MiB), and the most
 * values (numbers, strings, lists and objects) in it (2^20). A language
 * model's header holds a few thousand; the parse costs about 100 bytes a
 * value. */
#ifndef RAYO_MAX_HEADER_BYTES
#define RAYO_MAX_HEADER_BYTES 67108864L
#endif
#ifndef RAYO_MAX_HEADER_VALUES
#define RAYO_MAX_HEADER_VALUES 1048576L
#endif

/* The most tensors in a model file (16384), nodes in its graph (8192, and
 * as many inputs, states and outputs), and size symbols (64). A decoder of
 * 28 layers has about 300 tensors and 1100 nodes. */
#ifndef RAYO_MAX_TENSORS
#define RAYO_MAX_TENSORS 16384L
#endif
#ifndef RAYO_MAX_NODES
#define RAYO_MAX_NODES 8192L
#endif
#ifndef RAYO_MAX_SYMBOLS
#define RAYO_MAX_SYMBOLS 64
#endif

/* The most bytes a session may plan for its activations, states and
 * scratch (one allocation, made when the session is): 2 GiB - 1. Lower it
 * to fail early on a phone. */
#ifndef RAYO_MAX_ARENA_BYTES
#define RAYO_MAX_ARENA_BYTES 2147483647L
#endif

/* A tokenizer.json (rayo_tokenizer_create): its bytes (64 MiB) and values
 * (2^23; Qwen2.5's has about 600 000). */
#ifndef RAYO_MAX_TOKENIZER_JSON_BYTES
#define RAYO_MAX_TOKENIZER_JSON_BYTES 67108864L
#endif
#ifndef RAYO_MAX_TOKENIZER_JSON_VALUES
#define RAYO_MAX_TOKENIZER_JSON_VALUES 8388608L
#endif

/* The largest vocabulary a tokeniser may have: 2^22 tokens. */
#ifndef RAYO_MAX_VOCAB
#define RAYO_MAX_VOCAB 4194304L
#endif

/* The most bytes of text rayo_tokenizer_encode takes, and holds once
 * normalised and pre-tokenised: 16 MiB. */
#ifndef RAYO_MAX_TEXT_BYTES
#define RAYO_MAX_TEXT_BYTES 16777216L
#endif

/* A tokeniser's patterns (Split): the most bytes of one (4096), and the
 * work matching may do, in steps a code point of the text (4096). A
 * pattern that backtracks more than that on a text fails the encoding with
 * RAYO_ERR_LIMIT instead of taking time without bound; the patterns of
 * real tokenisers take a few steps a code point. */
#ifndef RAYO_MAX_PATTERN_BYTES
#define RAYO_MAX_PATTERN_BYTES 4096
#endif
#ifndef RAYO_PATTERN_STEPS
#define RAYO_PATTERN_STEPS 4096L
#endif

/* Compile-time checks: a negative array size fails the build. */
typedef char rayo_check_rank[RAYO_MAX_RANK >= 1 && RAYO_MAX_RANK <= 16 ? 1 : -1];
typedef char rayo_check_dim[RAYO_MAX_DIM >= 1 && RAYO_MAX_DIM <= 2147483647L ? 1 : -1];
typedef char rayo_check_op_inputs[RAYO_OP_MAX_INPUTS >= 1 ? 1 : -1];
typedef char rayo_check_limits[RAYO_MAX_HEADER_BYTES >= 1024 && RAYO_MAX_HEADER_VALUES >= 64 &&
                                       RAYO_MAX_TENSORS >= 1 && RAYO_MAX_TENSORS <= 16777216L &&
                                       RAYO_MAX_NODES >= 1 && RAYO_MAX_SYMBOLS >= 1 &&
                                       RAYO_MAX_NODES <= 16777216L && RAYO_MAX_ARENA_BYTES >= 1 &&
                                       RAYO_MAX_ARENA_BYTES <= 2147483647L &&
                                       RAYO_MAX_TEXT_BYTES <= 268435456L &&
                                       RAYO_MAX_VOCAB <= 268435456L
                                   ? 1
                                   : -1];
typedef char rayo_check_align[RAYO_ALIGN >= 16 && (RAYO_ALIGN & (RAYO_ALIGN - 1)) == 0 ? 1 : -1];

#endif
