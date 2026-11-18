/* rayo/text.h: text in and out of a language model. A tokeniser read from
 * the model's own tokenizer.json (the Hugging Face tokenizers format), and a
 * sampler that picks the next token from a model's logits.
 *
 * The tokeniser reads the kinds of tokenizer.json that decoder-only models
 * ship (docs/RUNNING-SLMS.md says which): a BPE model (byte-level, as GPT-2,
 * Llama 3, Qwen and SmolLM; or SentencePiece-style with byte fallback, as
 * Llama 2 and Mistral) or a Unigram model; Split, ByteLevel, Digits and
 * Metaspace pre-tokenisers; Replace and Prepend normalisers (NFC is taken
 * to hold already); and added tokens, which are matched in the text as
 * themselves. A file with anything else is refused when it is read, naming
 * what. */
#ifndef RAYO_TEXT_H
#define RAYO_TEXT_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/model.h"
#include "rayo/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rayo_tokenizer rayo_tokenizer;

/* Reads the `size` bytes of tokenizer.json at `json`. rayo keeps what it
 * needs: the caller may free `json` when this returns. Reading allocates
 * memory for the file's parse, then frees it. */
RAYO_API rayo_status rayo_tokenizer_create(const char *json, size_t size, rayo_tokenizer **out,
                                           rayo_message *msg);

/* The same, from the file at `path`. */
RAYO_API rayo_status rayo_tokenizer_load(const char *path, rayo_tokenizer **out, rayo_message *msg);

/* The tokeniser whose table rayo-convert wrote into `model`'s file
 * (docs/FORMAT.md), read where it is in the mapped file: nothing is parsed
 * or copied but its few patterns. `model` must outlive it.
 * RAYO_ERR_UNSUPPORTED when the file has no table. */
RAYO_API rayo_status rayo_tokenizer_from_model(const rayo_model *model, rayo_tokenizer **out,
                                               rayo_message *msg);

/* The tokeniser of a table (docs/FORMAT.md) of `size` bytes at `table`,
 * 4-byte aligned, which must outlive it: read where it is. */
RAYO_API rayo_status rayo_tokenizer_create_table(const void *table, size_t size,
                                                 rayo_tokenizer **out, rayo_message *msg);

/* The tokeniser's table: `*size` bytes at `*table`, owned by the tokeniser
 * (or its model). A tokeniser read from a tokenizer.json has one too, the
 * same bytes rayo-convert writes for that file. */
RAYO_API rayo_status rayo_tokenizer_table(const rayo_tokenizer *tokenizer, const void **table,
                                          size_t *size);

RAYO_API void rayo_tokenizer_destroy(rayo_tokenizer *tokenizer);

/* One more than the largest token id. */
RAYO_API rayo_i32 rayo_tokenizer_vocab_size(const rayo_tokenizer *tokenizer);

/* The token ids of the `size` bytes of UTF-8 text at `text`, into `ids`
 * (room for `capacity`). `*count` is how many there are, also when they do
 * not fit (then RAYO_ERR_SHAPE, and the first `capacity` are written).
 * Encoding allocates working memory and frees it before it returns. */
RAYO_API rayo_status rayo_tokenizer_encode(const rayo_tokenizer *tokenizer, const char *text,
                                           size_t size, rayo_i32 *ids, size_t capacity,
                                           size_t *count, rayo_message *msg);

/* The bytes token `id` stands for in text (`*size` of them, not
 * NUL-terminated), owned by the tokeniser; NULL for an id it does not have.
 * A multi-byte character may be split across tokens: join the pieces
 * before showing them. */
RAYO_API const char *rayo_tokenizer_piece(const rayo_tokenizer *tokenizer, rayo_i32 id,
                                          size_t *size);

/* The id of the token whose text is `token` (an added token such as
 * "<|im_end|>", or a vocabulary entry), or -1. */
RAYO_API rayo_i32 rayo_tokenizer_find(const rayo_tokenizer *tokenizer, const char *token);

/* Choosing the next token. Zero the options (memset) and set what you
 * need: zeros mean greedy decoding (the most likely token). */
typedef struct rayo_sampler_options {
  float temperature;        /* <= 0: greedy */
  rayo_i32 top_k;           /* keep the k most likely (0: all) */
  float top_p;              /* then the fewest whose probabilities add up to p (0 or 1: all) */
  float min_p;              /* then those at least min_p times the most likely (0: all) */
  float repetition_penalty; /* > 1: recent tokens' logits divided (positive) or multiplied */
  rayo_u32 seed;            /* the same seed and inputs give the same tokens */
} rayo_sampler_options;

typedef struct rayo_sampler rayo_sampler;

/* A sampler for logits of `vocab` values. It holds all its working memory:
 * rayo_sampler_next allocates nothing. */
RAYO_API rayo_status rayo_sampler_create(rayo_i32 vocab, const rayo_sampler_options *options,
                                         rayo_sampler **out, rayo_message *msg);

RAYO_API void rayo_sampler_destroy(rayo_sampler *sampler);

/* The next token from `logits` (`vocab` floats, not changed), with the
 * repetition penalty on the `recent_count` ids at `recent`. */
RAYO_API rayo_i32 rayo_sampler_next(rayo_sampler *sampler, const float *logits,
                                    const rayo_i32 *recent, size_t recent_count);

#ifdef __cplusplus
}
#endif

#endif
