/* The tokeniser (rayo/text.h): its table (text/table.h), read where it is
 * (in a mapped model file, or built from a tokenizer.json by
 * tokenizer_json.c), then each text normalised, split into words by the
 * pre-tokenisers and each word encoded by the model (BPE merges, or the
 * Unigram model's best segmentation). Nothing of the table is copied: a
 * tokeniser read from a mapped file costs only its compiled patterns. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/message.h"
#include "model.h"
#include "rayo/text.h"
#include "text/regex.h"
#include "text/table.h"
#include "text/unicode.h"

/* GPT-2's byte-level pre-tokeniser pattern (ByteLevel with use_regex). */
static const char gpt2_pattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

typedef struct pre_step {
  rayo_u32 kind;           /* TT_SPLIT, ... */
  rayo_regex rx;           /* SPLIT, and BYTELEVEL with use_regex */
  rayo_bool use_regex;     /* BYTELEVEL */
  rayo_bool prefix;        /* BYTELEVEL add_prefix_space; METASPACE prepend (first or always) */
  rayo_bool removed;       /* SPLIT: the matches are dropped (behaviour Removed) */
  rayo_bool individual;    /* DIGITS */
  rayo_bool split;         /* METASPACE */
  rayo_bool first_only;    /* METASPACE prepend_scheme first: only before the text's start */
  const char *replacement; /* METASPACE */
  size_t replacement_len;
} pre_step;

typedef struct norm_step {
  rayo_u32 kind; /* TT_REPLACE or TT_PREPEND */
  const char *from, *to;
  size_t from_len, to_len;
} norm_step;

struct rayo_tokenizer {
  rayo_arena arena;           /* the compiled patterns */
  const unsigned char *table; /* the table: in a model's file, or `owned` */
  size_t table_size;
  unsigned char *owned; /* built from a tokenizer.json, freed with the tokeniser */
  rayo_bool unigram;
  rayo_bool byte_level; /* the vocabulary is in GPT-2's byte-to-character mapping */
  rayo_bool byte_fallback;
  rayo_bool ignore_merges;
  rayo_i32 unk;
  rayo_i32 vocab_size;
  size_t max_len; /* the longest token, in bytes */
  float min_score;
  size_t vocab_mask, merge_mask; /* slots - 1; merge_mask 0: no merges */
  const unsigned char *vocab, *merges, *pieces, *scores, *added, *byte_ids, *pool;
  size_t added_count, pool_bytes;
  pre_step pre[TT_MAX_STEPS];
  int pre_count;
  norm_step norm[TT_MAX_STEPS];
  int norm_count;
};

/* ---- the table ------------------------------------------------------------------------------- */

static rayo_u32 rd(const unsigned char *p) {
  return (rayo_u32)p[0] | (rayo_u32)p[1] << 8 | (rayo_u32)p[2] << 16 | (rayo_u32)p[3] << 24;
}

static rayo_i32 rdi(const unsigned char *p) {
  const rayo_u32 v = rd(p);
  return v <= (rayo_u32)RAYO_I32_MAX ? (rayo_i32)v : -(rayo_i32)(0xFFFFFFFFu - v) - 1;
}

static rayo_i32 str_get(const rayo_tokenizer *t, const char *s, size_t n) {
  size_t i = rayo_toktab_hash_bytes(s, n) & t->vocab_mask;
  for (;;) {
    const unsigned char *e = t->vocab + i * TT_VOCAB_WORDS * 4;
    const rayo_i32 id = rdi(e + 8);
    if (id < 0) return -1;
    if (rd(e + 4) == n && memcmp(t->pool + rd(e), s, n) == 0) return id;
    i = (i + 1) & t->vocab_mask;
  }
}

/* The rank and result of merging a and b; rank -1 when they do not merge. */
static rayo_i32 pair_get(const rayo_tokenizer *t, rayo_i32 a, rayo_i32 b, rayo_i32 *merged) {
  size_t i;
  if (!t->merge_mask) return -1;
  i = rayo_toktab_hash_pair(a, b) & t->merge_mask;
  for (;;) {
    const unsigned char *e = t->merges + i * TT_MERGE_WORDS * 4;
    const rayo_i32 ea = rdi(e);
    if (ea < 0) return -1;
    if (ea == a && rdi(e + 4) == b) {
      *merged = rdi(e + 12);
      return rdi(e + 8);
    }
    i = (i + 1) & t->merge_mask;
  }
}

static float score_of(const rayo_tokenizer *t, rayo_i32 id) {
  const rayo_u32 bits = rd(t->scores + (size_t)id * 4);
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

static rayo_i32 byte_id(const rayo_tokenizer *t, unsigned b) {
  return rdi(t->byte_ids + (size_t)b * 4);
}

/* Whether [off, off + len) of the pool, and [at, at + count * words * 4)
 * of the table, are inside. */
static rayo_bool in_pool(const rayo_tokenizer *t, rayo_u32 off, rayo_u32 len) {
  return off <= t->pool_bytes && len <= t->pool_bytes - off;
}

static rayo_bool in_table(size_t size, rayo_u32 at, size_t count, size_t words) {
  return at % 4 == 0 && at >= TT_HEADER_WORDS * 4 && at <= size &&
         count <= (size - at) / (words * 4);
}

/* Every record of the table checked against it, as a model file is: the
 * table is untrusted. */
static rayo_status check_table(rayo_tokenizer *t, rayo_message *msg) {
  const unsigned char *b = t->table;
  const size_t size = t->table_size;
  size_t vslots, mslots, i;
  rayo_u32 flags;
  if (size < TT_HEADER_WORDS * 4 || memcmp(b, "RTOK", 4) != 0 || size % 4 != 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "not a tokeniser table");
  }
  if (rd(b + 4 * TT_VERSION) != RAYO_TOKTAB_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "tokeniser table version %lu, rayo reads %d",
                     (unsigned long)rd(b + 4 * TT_VERSION), RAYO_TOKTAB_VERSION);
  }
  flags = rd(b + 4 * TT_FLAGS);
  t->unigram = (flags & TT_UNIGRAM) != 0;
  t->byte_level = (flags & TT_BYTE_LEVEL) != 0;
  t->byte_fallback = (flags & TT_BYTE_FALLBACK) != 0;
  t->ignore_merges = (flags & TT_IGNORE_MERGES) != 0;
  t->unk = rdi(b + 4 * TT_UNK);
  t->vocab_size = rdi(b + 4 * TT_VOCAB_SIZE);
  t->max_len = rd(b + 4 * TT_MAX_LEN);
  {
    const rayo_u32 bits = rd(b + 4 * TT_MIN_SCORE);
    memcpy(&t->min_score, &bits, sizeof t->min_score);
  }
  vslots = rd(b + 4 * TT_VOCAB_SLOTS);
  mslots = rd(b + 4 * TT_MERGE_SLOTS);
  t->added_count = rd(b + 4 * TT_ADDED_COUNT);
  t->pre_count = (int)rd(b + 4 * TT_PRE_COUNT);
  t->norm_count = (int)rd(b + 4 * TT_NORM_COUNT);
  t->pool_bytes = rd(b + 4 * TT_POOL_BYTES);
  if (t->vocab_size < 1 || vslots < 16 || (vslots & (vslots - 1)) != 0 ||
      vslots < (size_t)t->vocab_size || (mslots != 0 && (mslots & (mslots - 1)) != 0) ||
      rd(b + 4 * TT_PRE_COUNT) > TT_MAX_STEPS || rd(b + 4 * TT_NORM_COUNT) > TT_MAX_STEPS ||
      rd(b + 4 * TT_TOTAL_BYTES) != size || t->unk < -1 || t->unk >= t->vocab_size ||
      !in_table(size, rd(b + 4 * TT_VOCAB_OFF), vslots, TT_VOCAB_WORDS) ||
      (mslots && !in_table(size, rd(b + 4 * TT_MERGES_OFF), mslots, TT_MERGE_WORDS)) ||
      !in_table(size, rd(b + 4 * TT_PIECES_OFF), (size_t)t->vocab_size, TT_PIECE_WORDS) ||
      (t->unigram && !in_table(size, rd(b + 4 * TT_SCORES_OFF), (size_t)t->vocab_size, 1)) ||
      !in_table(size, rd(b + 4 * TT_ADDED_OFF), t->added_count, TT_ADDED_WORDS) ||
      !in_table(size, rd(b + 4 * TT_PRE_OFF), (size_t)t->pre_count, TT_PRE_WORDS) ||
      !in_table(size, rd(b + 4 * TT_NORM_OFF), (size_t)t->norm_count, TT_NORM_WORDS) ||
      !in_table(size, rd(b + 4 * TT_BYTE_IDS_OFF), 256, 1) || rd(b + 4 * TT_POOL_OFF) > size ||
      t->pool_bytes > size - rd(b + 4 * TT_POOL_OFF)) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: bad header");
  }
  t->vocab = b + rd(b + 4 * TT_VOCAB_OFF);
  t->vocab_mask = vslots - 1;
  t->merges = b + rd(b + 4 * TT_MERGES_OFF);
  t->merge_mask = mslots ? mslots - 1 : 0;
  t->pieces = b + rd(b + 4 * TT_PIECES_OFF);
  t->scores = b + rd(b + 4 * TT_SCORES_OFF);
  t->added = b + rd(b + 4 * TT_ADDED_OFF);
  t->byte_ids = b + rd(b + 4 * TT_BYTE_IDS_OFF);
  t->pool = b + rd(b + 4 * TT_POOL_OFF);
  /* Every slot's string in the pool and id in range, with an empty slot to
   * end each probe. */
  {
    rayo_bool empty = RAYO_FALSE;
    for (i = 0; i < vslots; i++) {
      const unsigned char *e = t->vocab + i * 12;
      const rayo_i32 id = rdi(e + 8);
      if (id < 0) {
        empty = RAYO_TRUE;
      } else if (id >= t->vocab_size || !in_pool(t, rd(e), rd(e + 4))) {
        return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: vocabulary slot %lu",
                         (unsigned long)i);
      }
    }
    if (!empty) return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: a full vocabulary");
    empty = mslots == 0;
    for (i = 0; i < mslots; i++) {
      const unsigned char *e = t->merges + i * 16;
      if (rdi(e) < 0) {
        empty = RAYO_TRUE;
      } else if (rdi(e) >= t->vocab_size || rdi(e + 4) < 0 || rdi(e + 4) >= t->vocab_size ||
                 rdi(e + 8) < 0 || rdi(e + 12) < 0 || rdi(e + 12) >= t->vocab_size) {
        return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: merge slot %lu", (unsigned long)i);
      }
    }
    if (!empty) return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: full merges");
  }
  for (i = 0; i < (size_t)t->vocab_size; i++) {
    const unsigned char *e = t->pieces + i * 8;
    if (rd(e) != TT_NO_PIECE && !in_pool(t, rd(e), rd(e + 4))) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: piece %lu", (unsigned long)i);
    }
  }
  for (i = 0; i < t->added_count; i++) {
    const unsigned char *e = t->added + i * 12;
    if (!in_pool(t, rd(e), rd(e + 4)) || rd(e + 4) == 0 || rdi(e + 8) < 0 ||
        rdi(e + 8) >= t->vocab_size) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: added token %lu", (unsigned long)i);
    }
  }
  for (i = 0; i < 256; i++) {
    if (byte_id(t, (unsigned)i) < -1 || byte_id(t, (unsigned)i) >= t->vocab_size) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: byte %lu", (unsigned long)i);
    }
  }
  return RAYO_OK;
}

/* The steps, their patterns compiled. */
static rayo_status read_steps(rayo_tokenizer *t, rayo_message *msg) {
  const unsigned char *b = t->table;
  int k;
  for (k = 0; k < t->pre_count; k++) {
    const unsigned char *e = b + rd(b + 4 * TT_PRE_OFF) + (size_t)k * 16;
    pre_step *st = &t->pre[k];
    const rayo_u32 flags = rd(e + 4), off = rd(e + 8), len = rd(e + 12);
    char *str;
    rayo_status s;
    if (!in_pool(t, off, len) || rd(e) > TT_METASPACE) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: step %d", k);
    }
    str = rayo_arena_strndup(&t->arena, (const char *)t->pool + off, len);
    if (!str) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    memset(st, 0, sizeof *st);
    st->kind = rd(e);
    st->use_regex = (flags & TT_USE_REGEX) != 0;
    st->prefix = (flags & TT_PREFIX) != 0;
    st->removed = (flags & TT_REMOVED) != 0;
    st->individual = (flags & TT_INDIVIDUAL) != 0;
    st->split = (flags & TT_SPLIT_WORDS) != 0;
    st->first_only = (flags & TT_FIRST_ONLY) != 0;
    st->replacement = str;
    st->replacement_len = len;
    if (st->kind == TT_SPLIT || (st->kind == TT_BYTELEVEL && st->use_regex)) {
      s = rayo_regex_compile(st->kind == TT_SPLIT ? str : gpt2_pattern, &t->arena, &st->rx, msg);
      if (s != RAYO_OK) return s;
    }
  }
  for (k = 0; k < t->norm_count; k++) {
    const unsigned char *e = b + rd(b + 4 * TT_NORM_OFF) + (size_t)k * 20;
    norm_step *st = &t->norm[k];
    if (rd(e) > TT_PREPEND || !in_pool(t, rd(e + 4), rd(e + 8)) ||
        !in_pool(t, rd(e + 12), rd(e + 16)) || (rd(e) == TT_REPLACE && rd(e + 8) == 0)) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "tokeniser table: normaliser %d", k);
    }
    st->kind = rd(e);
    st->from = (const char *)t->pool + rd(e + 4);
    st->from_len = rd(e + 8);
    st->to = (const char *)t->pool + rd(e + 12);
    st->to_len = rd(e + 16);
  }
  return RAYO_OK;
}

/* A tokeniser of the table at `table`, kept (or owned, with `owned`). */
static rayo_status open_table(const unsigned char *table, size_t size, unsigned char *owned,
                              rayo_tokenizer **out, rayo_message *msg) {
  rayo_tokenizer *t;
  rayo_status s;
  *out = NULL;
  t = calloc(1, sizeof *t);
  if (!t) {
    free(owned);
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  t->table = table;
  t->table_size = size;
  t->owned = owned;
  s = check_table(t, msg);
  if (s == RAYO_OK) s = read_steps(t, msg);
  if (s != RAYO_OK) {
    rayo_tokenizer_destroy(t);
    return s;
  }
  *out = t;
  return RAYO_OK;
}

/* ---- growable buffers (encoding only) -------------------------------------------------------- */

typedef struct buffer {
  char *data;
  size_t len, cap;
  rayo_bool failed;
} buffer;

static void buf_add(buffer *b, const char *s, size_t n) {
  if (b->failed) return;
  if (b->len + n > b->cap) {
    size_t cap = b->cap ? b->cap * 2 : 256;
    char *p;
    while (cap < b->len + n) cap *= 2;
    p = realloc(b->data, cap);
    if (!p) {
      b->failed = RAYO_TRUE;
      return;
    }
    b->data = p;
    b->cap = cap;
  }
  if (n) memcpy(b->data + b->len, s, n);
  b->len += n;
}

/* Words: [start, end) byte ranges of a buffer. */
typedef struct span {
  size_t start, end;
} span;

typedef struct spans {
  span *items;
  size_t count, cap;
  rayo_bool failed;
} spans;

static void spans_add(spans *v, size_t start, size_t end) {
  if (v->failed || start >= end) return;
  if (v->count == v->cap) {
    size_t cap = v->cap ? v->cap * 2 : 64;
    span *p = realloc(v->items, cap * sizeof *p);
    if (!p) {
      v->failed = RAYO_TRUE;
      return;
    }
    v->items = p;
    v->cap = cap;
  }
  v->items[v->count].start = start;
  v->items[v->count].end = end;
  v->count++;
}

rayo_status rayo_tokenizer_create(const char *json, size_t size, rayo_tokenizer **out,
                                  rayo_message *msg) {
  unsigned char *table;
  size_t n;
  rayo_status s;
  if (!out || !json) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *out = NULL;
  s = rayo_toktab_from_json(json, size, &table, &n, msg);
  if (s != RAYO_OK) return s;
  return open_table(table, n, table, out, msg);
}

rayo_status rayo_tokenizer_create_table(const void *table, size_t size, rayo_tokenizer **out,
                                        rayo_message *msg) {
  if (!out || !table) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  if ((size_t)table % 4 != 0) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "the table must be 4-byte aligned");
  }
  return open_table(table, size, NULL, out, msg);
}

rayo_status rayo_tokenizer_from_model(const rayo_model *model, rayo_tokenizer **out,
                                      rayo_message *msg) {
  if (!model || !out) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *out = NULL;
  if (!model->tokenizer) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "the model file has no tokeniser table (convert it again, or load its "
                     "tokenizer.json)");
  }
  return open_table(model->tokenizer, model->tokenizer_size, NULL, out, msg);
}

rayo_status rayo_tokenizer_table(const rayo_tokenizer *tokenizer, const void **table,
                                 size_t *size) {
  if (!tokenizer || !table || !size) return RAYO_ERR_INVALID_ARGUMENT;
  *table = tokenizer->table;
  *size = tokenizer->table_size;
  return RAYO_OK;
}

rayo_status rayo_tokenizer_load(const char *path, rayo_tokenizer **out, rayo_message *msg) {
  FILE *f;
  long n;
  char *p;
  rayo_status s;
  if (!path || !out) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  f = fopen(path, "rb");
  if (!f) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot open %s", path);
  if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return rayo_fail(msg, RAYO_ERR_FORMAT, "cannot read %s", path);
  }
  p = malloc((size_t)n);
  if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
    free(p);
    fclose(f);
    return rayo_fail(msg, p ? RAYO_ERR_FORMAT : RAYO_ERR_OUT_OF_MEMORY, "cannot read %s", path);
  }
  fclose(f);
  s = rayo_tokenizer_create(p, (size_t)n, out, msg);
  free(p);
  return s;
}

void rayo_tokenizer_destroy(rayo_tokenizer *tokenizer) {
  if (!tokenizer) return;
  rayo_arena_free(&tokenizer->arena);
  free(tokenizer->owned);
  free(tokenizer);
}

rayo_i32 rayo_tokenizer_vocab_size(const rayo_tokenizer *tokenizer) {
  return tokenizer ? tokenizer->vocab_size : 0;
}

const char *rayo_tokenizer_piece(const rayo_tokenizer *tokenizer, rayo_i32 id, size_t *size) {
  const unsigned char *e;
  if (!tokenizer || id < 0 || id >= tokenizer->vocab_size ||
      rd((e = tokenizer->pieces + (size_t)id * 8)) == TT_NO_PIECE) {
    if (size) *size = 0;
    return NULL;
  }
  if (size) *size = rd(e + 4);
  return (const char *)tokenizer->pool + rd(e);
}

/* Added token i: its text, length and id. */
static const char *added_at(const rayo_tokenizer *t, size_t i, size_t *len, rayo_i32 *id) {
  const unsigned char *e = t->added + i * 12;
  *len = rd(e + 4);
  *id = rdi(e + 8);
  return (const char *)t->pool + rd(e);
}

rayo_i32 rayo_tokenizer_find(const rayo_tokenizer *tokenizer, const char *token) {
  size_t i, n, len;
  rayo_i32 id;
  if (!tokenizer || !token) return -1;
  n = strlen(token);
  for (i = 0; i < tokenizer->added_count; i++) {
    const char *s = added_at(tokenizer, i, &len, &id);
    if (len == n && memcmp(s, token, n) == 0) return id;
  }
  return str_get(tokenizer, token, n);
}

/* ---- encoding -------------------------------------------------------------------------------- */

typedef struct encoder {
  const rayo_tokenizer *t;
  rayo_i32 *ids;
  size_t cap, count;
  rayo_bool failed;
  /* Working memory, reused across words. */
  rayo_i32 *sym;
  size_t sym_cap;
  rayo_u32 *cps;
  size_t *cp_at;
  size_t cp_cap;
  float *best;
  long *from;
  rayo_i32 *best_id;
  size_t dp_cap;
} encoder;

static void emit(encoder *e, rayo_i32 id) {
  if (e->count < e->cap) e->ids[e->count] = id;
  e->count++;
}

static rayo_bool grow_syms(encoder *e, size_t n) {
  rayo_i32 *p;
  if (n <= e->sym_cap) return RAYO_TRUE;
  p = realloc(e->sym, n * sizeof *p);
  if (!p) return RAYO_FALSE;
  e->sym = p;
  e->sym_cap = n;
  return RAYO_TRUE;
}

/* A character that has no token: its bytes' tokens, or the unknown one. */
static void unknown(encoder *e, const unsigned char *s, size_t n, rayo_i32 *out, size_t *count) {
  const rayo_tokenizer *t = e->t;
  size_t i;
  rayo_bool all = t->byte_fallback;
  for (i = 0; all && i < n; i++) all = byte_id(t, s[i]) >= 0;
  if (all) {
    for (i = 0; i < n; i++) out[(*count)++] = byte_id(t, s[i]);
  } else if (t->unk >= 0) {
    out[(*count)++] = t->unk;
  }
}

static void bpe_word(encoder *e, const char *w, size_t n) {
  const rayo_tokenizer *t = e->t;
  size_t i = 0, count = 0, len;
  rayo_i32 whole;
  if (t->ignore_merges && (whole = str_get(t, w, n)) >= 0) {
    emit(e, whole);
    return;
  }
  if (!grow_syms(e, 4 * n + 1)) {
    e->failed = RAYO_TRUE;
    return;
  }
  while (i < n) {
    rayo_i32 id;
    len = 1;
    if (!t->byte_level) rayo_utf8_next((const unsigned char *)w + i, n - i, &len);
    id = str_get(t, w + i, len);
    if (id >= 0) {
      e->sym[count++] = id;
    } else {
      unknown(e, (const unsigned char *)w + i, len, e->sym, &count);
    }
    i += len;
  }
  /* Merge the pair of lowest rank, leftmost first, until none is left. */
  for (;;) {
    rayo_i32 rank = RAYO_I32_MAX, merged = -1;
    size_t at = 0, j;
    for (j = 0; j + 1 < count; j++) {
      rayo_i32 m = -1;
      const rayo_i32 r = pair_get(t, e->sym[j], e->sym[j + 1], &m);
      if (r >= 0 && r < rank) {
        rank = r;
        merged = m;
        at = j;
      }
    }
    if (merged < 0) break;
    e->sym[at] = merged;
    memmove(e->sym + at + 1, e->sym + at + 2, (count - at - 2) * sizeof *e->sym);
    count--;
  }
  for (i = 0; i < count; i++) emit(e, e->sym[i]);
}

/* The Unigram model: the segmentation of the highest total score. */
static void unigram_word(encoder *e, const char *w, size_t n) {
  const rayo_tokenizer *t = e->t;
  const float unk_score = t->min_score - 10.0f;
  size_t i, len, k, count = 0;
  long at;
  rayo_bool last_unk = RAYO_FALSE;
  if (n + 1 > e->dp_cap) {
    float *b = realloc(e->best, (n + 1) * sizeof *b);
    long *f = b ? realloc(e->from, (n + 1) * sizeof *f) : NULL;
    rayo_i32 *bi = f ? realloc(e->best_id, (n + 1) * sizeof *bi) : NULL;
    if (b) e->best = b;
    if (f) e->from = f;
    if (bi) e->best_id = bi;
    if (!b || !f || !bi) {
      e->failed = RAYO_TRUE;
      return;
    }
    e->dp_cap = n + 1;
  }
  for (i = 0; i <= n; i++) {
    e->best[i] = -3.0e38f;
    e->from[i] = -1;
  }
  e->best[0] = 0.0f;
  for (i = 0; i < n; i++) {
    rayo_bool any = RAYO_FALSE;
    if (e->from[i] < 0 && i > 0) continue;
    rayo_utf8_next((const unsigned char *)w + i, n - i, &len);
    for (k = len; i + k <= n && k <= t->max_len;) {
      const rayo_i32 id = str_get(t, w + i, k);
      size_t next;
      if (id >= 0) {
        const float sc = e->best[i] + score_of(t, id);
        if (sc > e->best[i + k]) {
          e->best[i + k] = sc;
          e->from[i + k] = (long)i;
          e->best_id[i + k] = id;
        }
        if (k == len) any = RAYO_TRUE;
      }
      if (i + k >= n) break;
      rayo_utf8_next((const unsigned char *)w + i + k, n - i - k, &next);
      k += next;
    }
    if (!any && e->best[i] + unk_score > e->best[i + len]) {
      e->best[i + len] = e->best[i] + unk_score;
      e->from[i + len] = (long)i;
      e->best_id[i + len] = -1; /* unknown */
    }
  }
  if (!grow_syms(e, 4 * n + 1)) {
    e->failed = RAYO_TRUE;
    return;
  }
  /* Back from the end, then reversed. */
  for (at = (long)n; at > 0; at = e->from[at]) {
    if (e->best_id[at] >= 0) {
      e->sym[count++] = e->best_id[at];
      last_unk = RAYO_FALSE;
    } else if (last_unk && !t->byte_fallback) {
      /* Unknown characters in a row are one unknown token, as the
       * tokenizers library fuses them. */
    } else {
      rayo_i32 tmp[8];
      size_t c = 0, j;
      last_unk = RAYO_TRUE;
      unknown(e, (const unsigned char *)w + e->from[at], (size_t)(at - e->from[at]), tmp, &c);
      for (j = c; j > 0; j--) e->sym[count++] = tmp[j - 1];
    }
    if (e->from[at] < 0) break;
  }
  for (i = count; i > 0; i--) emit(e, e->sym[i - 1]);
}

static void word(encoder *e, const char *w, size_t n) {
  if (n == 0) return;
  if (e->t->unigram) {
    unigram_word(e, w, n);
  } else {
    bpe_word(e, w, n);
  }
}

/* Splits each span of `in` (bytes in `b`) by `rx`: the matches, and the
 * text between them, each its own span (the matches dropped if `removed`). */
static rayo_bool split_regex(encoder *e, const rayo_regex *rx, rayo_bool removed, const buffer *b,
                             const spans *in, spans *out) {
  size_t s;
  for (s = 0; s < in->count; s++) {
    const span sp = in->items[s];
    size_t n = 0, i = sp.start, len, pos = 0, gap = 0;
    if (sp.end - sp.start + 1 > e->cp_cap) {
      const size_t cap = sp.end - sp.start + 1;
      rayo_u32 *c = realloc(e->cps, cap * sizeof *c);
      size_t *at = c ? realloc(e->cp_at, cap * sizeof *at) : NULL;
      if (c) e->cps = c;
      if (at) e->cp_at = at;
      if (!c || !at) return RAYO_FALSE;
      e->cp_cap = cap;
    }
    while (i < sp.end) {
      e->cp_at[n] = i;
      e->cps[n++] = rayo_utf8_next((const unsigned char *)b->data + i, sp.end - i, &len);
      i += len;
    }
    e->cp_at[n] = sp.end;
    while (pos < n) {
      const long end = rayo_regex_match(rx, e->cps, n, pos);
      if (end > (long)pos) {
        spans_add(out, e->cp_at[gap], e->cp_at[pos]);
        if (!removed) spans_add(out, e->cp_at[pos], e->cp_at[end]);
        pos = (size_t)end;
        gap = pos;
      } else {
        pos++;
      }
    }
    spans_add(out, e->cp_at[gap], e->cp_at[n]);
  }
  return !out->failed;
}

static void digits(const buffer *b, rayo_bool individual, const spans *in, spans *out) {
  size_t s, i;
  for (s = 0; s < in->count; s++) {
    const span sp = in->items[s];
    size_t start = sp.start;
    for (i = sp.start; i < sp.end; i++) {
      const rayo_bool d = b->data[i] >= '0' && b->data[i] <= '9';
      const rayo_bool prev = i > sp.start && b->data[i - 1] >= '0' && b->data[i - 1] <= '9';
      if (i > sp.start && (d != prev || (d && individual))) {
        spans_add(out, start, i);
        start = i;
      }
    }
    spans_add(out, start, sp.end);
  }
}

/* Rewrites each span into `b` (appended) with `fn`, giving new spans. */
static void rewrite(buffer *b, const pre_step *st, rayo_bool first_word, const spans *in,
                    spans *out) {
  size_t s, i;
  for (s = 0; s < in->count; s++) {
    const span sp = in->items[s];
    const size_t start = b->len;
    if (st->kind == TT_BYTELEVEL) {
      if (st->prefix && s == 0 && b->data[sp.start] != ' ') buf_add(b, " ", 1);
      for (i = sp.start; i < sp.end && !b->failed; i++) buf_add(b, b->data + i, 1);
      spans_add(out, start, b->len);
    } else { /* METASPACE: spaces to the replacement, prepended, split before each */
      size_t word_start = start;
      /* Prepended unless the text starts with what becomes the replacement. */
      if (st->prefix && s == 0 && (first_word || !st->first_only) && b->data[sp.start] != ' ' &&
          !(sp.end - sp.start >= st->replacement_len &&
            memcmp(b->data + sp.start, st->replacement, st->replacement_len) == 0)) {
        buf_add(b, st->replacement, st->replacement_len);
      }
      for (i = sp.start; i < sp.end && !b->failed; i++) {
        if (b->data[i] == ' ') {
          if (st->split && b->len > word_start) {
            spans_add(out, word_start, b->len);
            word_start = b->len;
          }
          buf_add(b, st->replacement, st->replacement_len);
        } else {
          buf_add(b, b->data + i, 1);
        }
      }
      spans_add(out, word_start, b->len);
    }
  }
}

/* Normalises `text` into `b`. */
static void normalise(const rayo_tokenizer *t, const char *text, size_t n, buffer *b) {
  buffer tmp;
  int k;
  size_t i;
  memset(&tmp, 0, sizeof tmp);
  buf_add(b, text, n);
  for (k = 0; k < t->norm_count && !b->failed; k++) {
    const norm_step *st = &t->norm[k];
    tmp.len = 0;
    if (st->kind == TT_PREPEND) {
      if (b->len) buf_add(&tmp, st->to, st->to_len);
      buf_add(&tmp, b->data, b->len);
    } else {
      for (i = 0; i < b->len;) {
        if (i + st->from_len <= b->len && memcmp(b->data + i, st->from, st->from_len) == 0) {
          buf_add(&tmp, st->to, st->to_len);
          i += st->from_len;
        } else {
          buf_add(&tmp, b->data + i, 1);
          i++;
        }
      }
    }
    b->len = 0;
    buf_add(b, tmp.data, tmp.len);
    if (tmp.failed) b->failed = RAYO_TRUE;
  }
  free(tmp.data);
}

/* Text without added tokens: normalised, split into words, encoded. */
static rayo_bool segment(encoder *e, const char *text, size_t n, rayo_bool first) {
  const rayo_tokenizer *t = e->t;
  buffer b;
  spans a, c;
  int k;
  size_t i;
  rayo_bool ok = RAYO_TRUE;
  memset(&b, 0, sizeof b);
  memset(&a, 0, sizeof a);
  memset(&c, 0, sizeof c);
  normalise(t, text, n, &b);
  spans_add(&a, 0, b.len);
  for (k = 0; k < t->pre_count && ok && !b.failed; k++) {
    const pre_step *st = &t->pre[k];
    c.count = 0;
    switch (st->kind) {
      case TT_SPLIT:
        ok = split_regex(e, &st->rx, st->removed, &b, &a, &c);
        break;
      case TT_BYTELEVEL:
        rewrite(&b, st, first, &a, &c);
        if (st->use_regex) {
          spans d = a;
          a = c;
          c = d;
          c.count = 0;
          ok = split_regex(e, &st->rx, RAYO_FALSE, &b, &a, &c);
        }
        break;
      case TT_DIGITS:
        digits(&b, st->individual, &a, &c);
        break;
      case TT_METASPACE:
        rewrite(&b, st, first, &a, &c);
        break;
    }
    {
      spans d = a;
      a = c;
      c = d;
    }
    ok = ok && !a.failed;
  }
  ok = ok && !b.failed;
  for (i = 0; ok && i < a.count; i++)
    word(e, b.data + a.items[i].start, a.items[i].end - a.items[i].start);
  free(b.data);
  free(a.items);
  free(c.items);
  return ok && !e->failed;
}

rayo_status rayo_tokenizer_encode(const rayo_tokenizer *tokenizer, const char *text, size_t size,
                                  rayo_i32 *ids, size_t capacity, size_t *count,
                                  rayo_message *msg) {
  encoder e;
  size_t i = 0, start = 0, k;
  rayo_bool ok = RAYO_TRUE;
  if (!tokenizer || (!text && size) || (!ids && capacity) || !count) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  }
  memset(&e, 0, sizeof e);
  e.t = tokenizer;
  e.ids = ids;
  e.cap = capacity;
  /* Added tokens first, longest first at each position; the text between
   * them goes through the model. */
  while (i < size && ok) {
    rayo_i32 hit = -1;
    size_t hit_len = 0;
    for (k = 0; k < tokenizer->added_count; k++) {
      size_t len;
      rayo_i32 id;
      const char *a = added_at(tokenizer, k, &len, &id);
      if (len <= size - i && memcmp(text + i, a, len) == 0) {
        hit = id;
        hit_len = len;
        break;
      }
    }
    if (hit < 0) {
      i++;
      continue;
    }
    if (i > start) ok = segment(&e, text + start, i - start, start == 0);
    emit(&e, hit);
    i += hit_len;
    start = i;
  }
  if (ok && size > start) ok = segment(&e, text + start, size - start, start == 0);
  free(e.sym);
  free(e.cps);
  free(e.cp_at);
  free(e.best);
  free(e.from);
  free(e.best_id);
  *count = e.count;
  if (!ok) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (e.count > capacity) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "%lu tokens, room for %lu", (unsigned long)e.count,
                     (unsigned long)capacity);
  }
  return RAYO_OK;
}
