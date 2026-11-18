/* rayo's fuzz targets (tests/fuzz/README.md): each defines
 * LLVMFuzzerTestOneInput, for libFuzzer (clang -fsanitize=fuzzer), and
 * driver.c runs one on files or standard input, for AFL and for replaying a
 * corpus with any C89 compiler. C89, as the library. */
#ifndef RAYO_FUZZ_H
#define RAYO_FUZZ_H

#include <stddef.h>

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size);

/* Optional set-up, once, before the first input (libFuzzer's signature). */
int LLVMFuzzerInitialize(int *argc, char ***argv);

/* What a run may give a session: larger arenas are skipped (a fuzzer's
 * memory limit is not a bug), and each size symbol is bound at most this
 * high. */
#define RAYO_FUZZ_MAX_ARENA (64L * 1024 * 1024)
#define RAYO_FUZZ_MAX_SYMBOL 3

/* The whole file at `path`, NUL-terminated, or NULL; free() it. */
unsigned char *rayo_fuzz_read(const char *path, size_t *size);

/* Encodes `size` bytes of text with `tokenizer` and decodes every id it
 * gives, as an app would. */
struct rayo_tokenizer;
void rayo_fuzz_encode(const struct rayo_tokenizer *tokenizer, const char *text, size_t size);

/* Loads, plans and runs the model in `data` with small sizes, and its
 * tokeniser if it has one. */
void rayo_fuzz_model(const unsigned char *data, size_t size);

#endif
