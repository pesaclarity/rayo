/* What the tokenisers need of Unicode: UTF-8, and whether a code point is a
 * letter, a number or white space, as the patterns of tokenizer.json files
 * (\p{L}, \p{N}, \s) mean them. */
#ifndef RAYO_TEXT_UNICODE_H
#define RAYO_TEXT_UNICODE_H

#include <stddef.h>

#include "rayo/types.h"

typedef struct rayo_unicode_range {
  rayo_u32 first, last;
} rayo_unicode_range;

extern const rayo_unicode_range rayo_unicode_letters[];
extern const size_t rayo_unicode_letters_count;
extern const rayo_unicode_range rayo_unicode_numbers[];
extern const size_t rayo_unicode_numbers_count;

rayo_bool rayo_unicode_is_letter(rayo_u32 c);
rayo_bool rayo_unicode_is_number(rayo_u32 c);
rayo_bool rayo_unicode_is_space(rayo_u32 c);

/* The code point starting at s[0] (of n > 0 bytes) and its length in bytes.
 * Bytes that are not UTF-8 read as themselves, one at a time, so every
 * input splits into pieces that join back to it. */
rayo_u32 rayo_utf8_next(const unsigned char *s, size_t n, size_t *len);

#endif
