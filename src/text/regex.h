/* The patterns of tokenizer.json pre-tokenisers (Split with a Regex, and
 * ByteLevel's own), matched as the tokenizers library's regex engine
 * (Oniguruma) matches them, on code points: alternation tried left to
 * right, greedy quantifiers that give back, lookahead.
 *
 * What a pattern may use: literal characters; . ; \s \S \d \D \w \W \p{L}
 * \p{N} \P{L} \P{N} (\pL, \pN) and escaped characters (\r \n \t \f \v \\
 * \' and the like); classes [...] and [^...] with ranges; groups ( ),
 * (?: ), (?i: ) (case-insensitive ASCII) and lookahead (?= ), (?! );
 * quantifiers ? * + {n} {n,} {n,m}, greedy, on a single character or class.
 * Anything else is refused when the pattern is read, naming it, as is a
 * pattern of more than RAYO_MAX_PATTERN_BYTES or groups nested more than
 * 16 deep. */
#ifndef RAYO_TEXT_REGEX_H
#define RAYO_TEXT_REGEX_H

#include <stddef.h>

#include "core/arena.h"
#include "rayo/error.h"
#include "rayo/types.h"

typedef struct rayo_rx_node rayo_rx_node;

typedef struct rayo_regex {
  const rayo_rx_node *root;
} rayo_regex;

/* Reads the pattern (UTF-8, NUL-terminated) into `out`, in `arena`. */
rayo_status rayo_regex_compile(const char *pattern, rayo_arena *arena, rayo_regex *out,
                               rayo_message *msg);

/* What rayo_regex_match returns when `*budget` runs out. */
#define RAYO_REGEX_TOO_COSTLY (-2L)

/* The end of the match starting at code point `at` of `text` (`n` of them),
 * or -1 when none starts there. Each step of the matcher (a node tried, a
 * character a repetition reads) takes one from `*budget`; when it runs out,
 * or the match nests too deeply, RAYO_REGEX_TOO_COSTLY: a pattern from an
 * untrusted file can backtrack polynomially, and its time is bounded by
 * the caller's budget instead. */
long rayo_regex_match(const rayo_regex *rx, const rayo_u32 *text, size_t n, size_t at,
                      long *budget);

#endif
