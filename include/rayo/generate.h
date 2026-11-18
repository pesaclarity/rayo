/* rayo/generate.h: text from a language model, a token at a time. A
 * generator joins a session of a decoder converted by rayo-convert (its
 * input the tokens, its output the last position's logits, symbols P and
 * T), its tokeniser and a sampler; it keeps the key/value cache across
 * calls, so a chat goes on where it left off.
 *
 *   rayo_generator_create(model, tokenizer, &options, &g, &msg);
 *   rayo_generator_chat(g, NULL, "Hello!", &msg);       (or _prompt: raw text)
 *   while (rayo_generator_next(g, &token, &msg) == RAYO_OK && token >= 0) {
 *     piece = rayo_tokenizer_piece(tokenizer, token, &size);  (show it)
 *   }
 *
 * Each call does a bounded amount of work (a prompt chunk, or one token),
 * so a caller cancels by not calling again, and a UI thread or a Dart
 * isolate stays responsive. Running the model allocates nothing; only the
 * tokeniser's encoding of new text takes working memory, and frees it. */
#ifndef RAYO_GENERATE_H
#define RAYO_GENERATE_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/model.h"
#include "rayo/text.h"
#include "rayo/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rayo_generator rayo_generator;

/* Zero them (memset), then set what you need. */
typedef struct rayo_generator_options {
  rayo_sampler_options sampling; /* zeros: greedy */
  int threads;                   /* as rayo_session_options: 0 or 1 one, -1 the fast cores */
  rayo_i32 max_tokens;           /* per reply: 0 for no limit but the context */
} rayo_generator_options;

/* A generator for `model` (a decoder converted by rayo-convert) and
 * `tokenizer` (its tokenizer.json), both of which must outlive it. The stop
 * tokens, the chat template's pieces and whether a prompt starts with BOS
 * come from the model's metadata. */
RAYO_API rayo_status rayo_generator_create(const rayo_model *model, const rayo_tokenizer *tokenizer,
                                           const rayo_generator_options *options,
                                           rayo_generator **out, rayo_message *msg);

RAYO_API void rayo_generator_destroy(rayo_generator *generator);

/* Adds a user's turn (and, before the first, an optional system text) in
 * the model's own chat template, and runs it through the model. Then call
 * rayo_generator_next for the reply. RAYO_ERR_UNSUPPORTED when the model
 * has no chat template; RAYO_ERR_SHAPE when the context is full. */
RAYO_API rayo_status rayo_generator_chat(rayo_generator *generator, const char *system,
                                         const char *user, rayo_message *msg);

/* Adds `size` bytes of raw text (BOS first, at the start, if the model
 * wants one) and runs it through the model. */
RAYO_API rayo_status rayo_generator_prompt(rayo_generator *generator, const char *text, size_t size,
                                           rayo_message *msg);

/* The next token of the reply in `*token`, or -1 when the reply has ended
 * (a stop token, max_tokens, or a full context). */
RAYO_API rayo_status rayo_generator_next(rayo_generator *generator, rayo_i32 *token,
                                         rayo_message *msg);

/* Forgets the conversation: an empty cache, position 0. */
RAYO_API void rayo_generator_reset(rayo_generator *generator);

/* Tokens in the cache, and the most it holds. */
RAYO_API rayo_i32 rayo_generator_position(const rayo_generator *generator);
RAYO_API rayo_i32 rayo_generator_context(const rayo_generator *generator);

/* Whether the model's metadata has a chat template (rayo-convert with
 * jinja2 installed writes it). */
RAYO_API rayo_bool rayo_generator_can_chat(const rayo_generator *generator);

/* The session's threads and kernels, for reports. */
RAYO_API int rayo_generator_threads(const rayo_generator *generator);
RAYO_API const char *rayo_generator_kernels(const rayo_generator *generator);

#ifdef __cplusplus
}
#endif

#endif
