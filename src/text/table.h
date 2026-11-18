/* The tokeniser's table (docs/FORMAT.md, "The tokeniser's table"): what a
 * tokenizer.json says, in one block of little-endian 32-bit words and
 * bytes that rayo reads where it is, such as in a mapped model file, with
 * no parsing. rayo-convert writes it into the model file; rayo builds the
 * same bytes from a tokenizer.json (tokenizer_json.c). */
#ifndef RAYO_TEXT_TABLE_H
#define RAYO_TEXT_TABLE_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/types.h"

#define RAYO_TOKTAB_VERSION 1

/* The header's words. */
enum {
  TT_MAGIC,
  TT_VERSION,
  TT_FLAGS,
  TT_UNK,
  TT_VOCAB_SIZE,
  TT_MAX_LEN,
  TT_MIN_SCORE, /* float32 bits */
  TT_VOCAB_SLOTS,
  TT_MERGE_SLOTS,
  TT_ADDED_COUNT,
  TT_PRE_COUNT,
  TT_NORM_COUNT,
  TT_POOL_BYTES,
  TT_VOCAB_OFF,
  TT_MERGES_OFF,
  TT_PIECES_OFF,
  TT_SCORES_OFF,
  TT_ADDED_OFF,
  TT_PRE_OFF,
  TT_NORM_OFF,
  TT_BYTE_IDS_OFF,
  TT_POOL_OFF,
  TT_TOTAL_BYTES,
  TT_HEADER_WORDS = 32
};

/* TT_FLAGS */
#define TT_UNIGRAM 1u
#define TT_BYTE_LEVEL 2u
#define TT_BYTE_FALLBACK 4u
#define TT_IGNORE_MERGES 8u
#define TT_METASPACE_DECODE 16u

/* Words of each record: a vocabulary slot (string offset, length, id; id -1
 * when empty), a merge slot (a, b, rank, merged; a -1 when empty), a piece
 * (offset, length; offset 0xFFFFFFFF for none), an added token (offset,
 * length, id), a pre-tokeniser step (kind, flags, string offset, length)
 * and a normaliser step (kind, from offset, from length, to offset, to
 * length). */
#define TT_VOCAB_WORDS 3
#define TT_MERGE_WORDS 4
#define TT_PIECE_WORDS 2
#define TT_ADDED_WORDS 3
#define TT_PRE_WORDS 4
#define TT_NORM_WORDS 5
#define TT_NO_PIECE 0xFFFFFFFFu

/* Pre-tokeniser kinds and flags, normaliser kinds. */
enum { TT_SPLIT, TT_BYTELEVEL, TT_DIGITS, TT_METASPACE };
#define TT_USE_REGEX 1u
#define TT_PREFIX 2u
#define TT_REMOVED 4u
#define TT_INDIVIDUAL 8u
#define TT_SPLIT_WORDS 16u
#define TT_FIRST_ONLY 32u
enum { TT_REPLACE, TT_PREPEND };

/* The most steps of each kind. */
#define TT_MAX_STEPS 8

/* The hashes of the vocabulary's strings (FNV-1a) and of merge pairs. */
rayo_u32 rayo_toktab_hash_bytes(const char *s, size_t n);
rayo_u32 rayo_toktab_hash_pair(rayo_i32 a, rayo_i32 b);

/* Slots for n entries: a power of two, at least 2n + 1 and 16. */
size_t rayo_toktab_slots(size_t n);

/* The table of the tokenizer.json at `json`, in `*out` (`*size` bytes, to
 * free()). */
rayo_status rayo_toktab_from_json(const char *json, size_t size, unsigned char **out,
                                  size_t *out_size, rayo_message *msg);

#endif
