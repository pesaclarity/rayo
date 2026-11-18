/* The tokeniser of tokenizer.json files (rayo/text.h): what the file says,
 * read into hash tables, then each text normalised, split into words by the
 * pre-tokenisers and each word encoded by the model (BPE merges, or the
 * Unigram model's best segmentation). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "core/message.h"
#include "rayo/text.h"
#include "text/regex.h"
#include "text/unicode.h"

#define MAX_STEPS 8

/* GPT-2's byte-level pre-tokeniser pattern (ByteLevel with use_regex). */
static const char gpt2_pattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

typedef enum step_kind { STEP_SPLIT, STEP_BYTELEVEL, STEP_DIGITS, STEP_METASPACE } step_kind;

typedef enum norm_kind { NORM_REPLACE, NORM_PREPEND } norm_kind;

typedef struct pre_step {
  step_kind kind;
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
  norm_kind kind;
  const char *from, *to; /* REPLACE: from to to; PREPEND: to */
  size_t from_len, to_len;
} norm_step;

/* A hash table of byte strings to ids, open addressing. */
typedef struct str_entry {
  const char *s;
  size_t len;
  rayo_i32 id;
} str_entry;

typedef struct str_map {
  str_entry *slots;
  size_t mask;
} str_map;

/* A hash table of id pairs to a merge's rank and result. */
typedef struct pair_entry {
  rayo_i32 a, b, rank, merged; /* a < 0: empty */
} pair_entry;

typedef struct pair_map {
  pair_entry *slots;
  size_t mask;
} pair_map;

typedef struct added_token {
  const char *s;
  size_t len;
  rayo_i32 id;
} added_token;

struct rayo_tokenizer {
  rayo_arena arena;
  rayo_bool unigram;
  rayo_bool byte_level; /* the vocabulary is in GPT-2's byte-to-character mapping */
  rayo_bool byte_fallback;
  rayo_bool ignore_merges;
  rayo_bool metaspace_decode; /* show ▁ as a space */
  rayo_i32 unk;
  rayo_i32 vocab_size;
  const char **piece; /* per id: its text, as output */
  size_t *piece_len;
  str_map vocab; /* the model's tokens, in the form the model's words take */
  pair_map merges;
  float *scores; /* Unigram */
  float min_score;
  size_t max_len;         /* the longest token, in bytes */
  rayo_i32 byte_ids[256]; /* <0xXX> tokens, or -1 */
  rayo_i16 unbyte[512];   /* the byte each byte-level character below 512 stands for, or -1 */
  added_token *added;     /* longest first */
  size_t added_count;
  pre_step pre[MAX_STEPS];
  int pre_count;
  norm_step norm[MAX_STEPS];
  int norm_count;
};

/* ---- hash tables ---------------------------------------------------------------------------- */

static rayo_u32 hash_bytes(const char *s, size_t n) {
  rayo_u32 h = 2166136261u;
  size_t i;
  for (i = 0; i < n; i++) h = (h ^ (rayo_u8)s[i]) * 16777619u;
  return h;
}

static rayo_u32 hash_pair(rayo_i32 a, rayo_i32 b) {
  rayo_u32 h = (rayo_u32)a * 2654435761u ^ ((rayo_u32)b + 0x9E3779B9u);
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  return h;
}

static size_t table_size(size_t n) {
  size_t size = 16;
  while (size < 2 * n + 1) size *= 2;
  return size;
}

static rayo_bool str_init(rayo_arena *a, str_map *m, size_t n) {
  const size_t size = table_size(n);
  m->slots = rayo_arena_array(a, size, sizeof *m->slots);
  m->mask = size - 1;
  return m->slots != NULL;
}

static rayo_i32 str_get(const str_map *m, const char *s, size_t n) {
  size_t i = hash_bytes(s, n) & m->mask;
  while (m->slots[i].s) {
    if (m->slots[i].len == n && memcmp(m->slots[i].s, s, n) == 0) return m->slots[i].id;
    i = (i + 1) & m->mask;
  }
  return -1;
}

/* Adds s (kept by the caller) as id; an earlier entry for s stays. */
static void str_put(str_map *m, const char *s, size_t n, rayo_i32 id) {
  size_t i = hash_bytes(s, n) & m->mask;
  while (m->slots[i].s) {
    if (m->slots[i].len == n && memcmp(m->slots[i].s, s, n) == 0) return;
    i = (i + 1) & m->mask;
  }
  m->slots[i].s = s;
  m->slots[i].len = n;
  m->slots[i].id = id;
}

static rayo_bool pair_init(rayo_arena *a, pair_map *m, size_t n) {
  const size_t size = table_size(n);
  size_t i;
  m->slots = rayo_arena_array(a, size, sizeof *m->slots);
  m->mask = size - 1;
  if (!m->slots) return RAYO_FALSE;
  for (i = 0; i < size; i++) m->slots[i].a = -1;
  return RAYO_TRUE;
}

static const pair_entry *pair_get(const pair_map *m, rayo_i32 a, rayo_i32 b) {
  size_t i;
  if (!m->slots) return NULL;
  i = hash_pair(a, b) & m->mask;
  while (m->slots[i].a >= 0) {
    if (m->slots[i].a == a && m->slots[i].b == b) return &m->slots[i];
    i = (i + 1) & m->mask;
  }
  return NULL;
}

static void pair_put(pair_map *m, rayo_i32 a, rayo_i32 b, rayo_i32 rank, rayo_i32 merged) {
  size_t i = hash_pair(a, b) & m->mask;
  while (m->slots[i].a >= 0) {
    if (m->slots[i].a == a && m->slots[i].b == b) return; /* the first rank stays */
    i = (i + 1) & m->mask;
  }
  m->slots[i].a = a;
  m->slots[i].b = b;
  m->slots[i].rank = rank;
  m->slots[i].merged = merged;
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

/* ---- GPT-2's byte-to-character mapping --------------------------------------------------------
 */

static rayo_bool byte_prints(unsigned b) {
  return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
}

/* The character byte b is written as. */
static rayo_u32 byte_char(unsigned b) {
  unsigned i, n = 0;
  if (byte_prints(b)) return b;
  for (i = 0; i < b; i++) n += byte_prints(i) ? 0u : 1u;
  return 256u + n;
}

/* The byte character c stands for, or -1. */
static int char_byte(rayo_u32 c) {
  unsigned b;
  if (c < 256) return byte_prints(c) ? (int)c : -1;
  for (b = 0; b < 256; b++) {
    if (!byte_prints(b) && byte_char(b) == c) return (int)b;
  }
  return -1;
}

/* s in the byte-level mapping, back to its bytes, into out (at least n
 * bytes); its length. Characters that are not in the mapping stay as they
 * are. */
static size_t unmap_bytes(const rayo_i16 *table, const char *s, size_t n, char *out) {
  size_t i = 0, o = 0, len;
  rayo_u32 c;
  while (i < n) {
    c = rayo_utf8_next((const unsigned char *)s + i, n - i, &len);
    if (c < 512 && table[c] >= 0) {
      out[o++] = (char)table[c];
    } else {
      memcpy(out + o, s + i, len);
      o += len;
    }
    i += len;
  }
  return o;
}

/* Whether `needle` is in the `n` bytes at `s`. */
static rayo_bool strstr_n(const char *s, size_t n, const char *needle) {
  const size_t m = strlen(needle);
  size_t i;
  for (i = 0; i + m <= n; i++) {
    if (memcmp(s + i, needle, m) == 0) return RAYO_TRUE;
  }
  return RAYO_FALSE;
}

/* ---- reading tokenizer.json ------------------------------------------------------------------ */

typedef struct reader {
  rayo_tokenizer *t;
  rayo_message *msg;
} reader;

static const char *str_of(const rayo_json *j, const char *key) {
  return rayo_json_as_string(rayo_json_get(j, key));
}

static rayo_bool bool_of(const rayo_json *j, const char *key, rayo_bool fallback) {
  const rayo_json *v = rayo_json_get(j, key);
  return v && v->kind == RAYO_JSON_BOOL ? v->boolean : fallback;
}

static char *keep(rayo_tokenizer *t, const char *s, size_t n) {
  return rayo_arena_strndup(&t->arena, s, n);
}

static rayo_status read_normalizer(reader *r, const rayo_json *n) {
  rayo_tokenizer *t = r->t;
  const char *type;
  size_t i;
  if (!n || n->kind == RAYO_JSON_NULL) return RAYO_OK;
  type = str_of(n, "type");
  if (!type) return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: a normalizer without a type");
  if (strcmp(type, "Sequence") == 0) {
    const rayo_json *list = rayo_json_get(n, "normalizers");
    for (i = 0; list && list->kind == RAYO_JSON_ARRAY && i < list->length; i++) {
      rayo_status s = read_normalizer(r, list->items[i]);
      if (s != RAYO_OK) return s;
    }
    return RAYO_OK;
  }
  if (strcmp(type, "NFC") == 0) return RAYO_OK; /* text is taken to be NFC already */
  if (t->norm_count == MAX_STEPS) {
    return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: too many normalizers");
  }
  if (strcmp(type, "Replace") == 0) {
    const char *from = str_of(rayo_json_get(n, "pattern"), "String");
    const char *to = str_of(n, "content");
    norm_step *st = &t->norm[t->norm_count];
    if (!from || !to || !*from) {
      return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: Replace of a string only");
    }
    st->kind = NORM_REPLACE;
    st->from = keep(t, from, strlen(from));
    st->to = keep(t, to, strlen(to));
    st->from_len = strlen(from);
    st->to_len = strlen(to);
    t->norm_count++;
    return RAYO_OK;
  }
  if (strcmp(type, "Prepend") == 0) {
    const char *to = str_of(n, "prepend");
    norm_step *st = &t->norm[t->norm_count];
    if (!to) return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: Prepend without prepend");
    st->kind = NORM_PREPEND;
    st->to = keep(t, to, strlen(to));
    st->to_len = strlen(to);
    t->norm_count++;
    return RAYO_OK;
  }
  return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: normalizer %s", type);
}

static rayo_status read_pre(reader *r, const rayo_json *n) {
  rayo_tokenizer *t = r->t;
  const char *type;
  pre_step *st;
  rayo_status s;
  size_t i;
  if (!n || n->kind == RAYO_JSON_NULL) return RAYO_OK;
  type = str_of(n, "type");
  if (!type) return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: a pre_tokenizer without a type");
  if (strcmp(type, "Sequence") == 0) {
    const rayo_json *list = rayo_json_get(n, "pretokenizers");
    for (i = 0; list && list->kind == RAYO_JSON_ARRAY && i < list->length; i++) {
      s = read_pre(r, list->items[i]);
      if (s != RAYO_OK) return s;
    }
    return RAYO_OK;
  }
  if (t->pre_count == MAX_STEPS) {
    return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: too many pre_tokenizers");
  }
  st = &t->pre[t->pre_count];
  memset(st, 0, sizeof *st);
  if (strcmp(type, "Split") == 0) {
    const char *pattern = str_of(rayo_json_get(n, "pattern"), "Regex");
    const char *literal = str_of(rayo_json_get(n, "pattern"), "String");
    const char *behavior = str_of(n, "behavior");
    if (bool_of(n, "invert", RAYO_FALSE) || !behavior ||
        (strcmp(behavior, "Isolated") != 0 && strcmp(behavior, "Removed") != 0)) {
      return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED,
                       "tokenizer: Split with behavior %s (Isolated and Removed are read)",
                       behavior ? behavior : "(none)");
    }
    if (!pattern && literal) {
      /* A literal string: escape it as a pattern. */
      char *esc = rayo_arena_alloc(&t->arena, 2 * strlen(literal) + 1);
      size_t o = 0;
      if (!esc) return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
      for (i = 0; literal[i]; i++) {
        if (strchr("\\.^$|?*+()[]{}", literal[i])) esc[o++] = '\\';
        esc[o++] = literal[i];
      }
      pattern = esc;
    }
    if (!pattern) return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: Split without a pattern");
    s = rayo_regex_compile(pattern, &t->arena, &st->rx, r->msg);
    if (s != RAYO_OK) return s;
    st->kind = STEP_SPLIT;
    st->removed = strcmp(behavior, "Removed") == 0;
  } else if (strcmp(type, "ByteLevel") == 0) {
    st->kind = STEP_BYTELEVEL;
    st->prefix = bool_of(n, "add_prefix_space", RAYO_FALSE);
    st->use_regex = bool_of(n, "use_regex", RAYO_TRUE);
    if (st->use_regex) {
      s = rayo_regex_compile(gpt2_pattern, &t->arena, &st->rx, r->msg);
      if (s != RAYO_OK) return s;
    }
    t->byte_level = RAYO_TRUE;
  } else if (strcmp(type, "Digits") == 0) {
    st->kind = STEP_DIGITS;
    st->individual = bool_of(n, "individual_digits", RAYO_FALSE);
  } else if (strcmp(type, "Metaspace") == 0) {
    const char *rep = str_of(n, "replacement");
    const char *scheme = str_of(n, "prepend_scheme");
    st->kind = STEP_METASPACE;
    st->replacement = keep(t, rep ? rep : "\xe2\x96\x81", strlen(rep ? rep : "\xe2\x96\x81"));
    st->replacement_len = strlen(st->replacement);
    st->prefix = scheme ? strcmp(scheme, "never") != 0 : bool_of(n, "add_prefix_space", RAYO_TRUE);
    st->first_only = scheme && strcmp(scheme, "first") == 0;
    st->split = bool_of(n, "split", RAYO_TRUE);
    t->metaspace_decode = RAYO_TRUE;
  } else {
    return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: pre_tokenizer %s", type);
  }
  t->pre_count++;
  return RAYO_OK;
}

/* Gives id its output text: the bytes of a byte-level token, a
 * SentencePiece token with ▁ as a space, a <0xXX> token as its byte. */
static rayo_status set_piece(reader *r, rayo_i32 id, const char *s, size_t n, rayo_bool special) {
  rayo_tokenizer *t = r->t;
  char *out;
  size_t o = 0, i;
  unsigned v;
  if (t->piece[id]) return RAYO_OK;
  out = rayo_arena_alloc(&t->arena, n + 1);
  if (!out) return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (!t->byte_level && n == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>' &&
      sscanf(s + 3, "%2x", &v) == 1) {
    /* A byte token, <0xXX>: that byte. */
    out[o++] = (char)v;
    if (t->byte_ids[v & 255u] < 0) t->byte_ids[v & 255u] = id;
  } else if (special) {
    memcpy(out, s, n);
    o = n;
  } else if (t->byte_level) {
    o = unmap_bytes(t->unbyte, s, n, out);
  } else {
    for (i = 0; i < n; i++) {
      if (t->metaspace_decode && i + 3 <= n && memcmp(s + i, "\xe2\x96\x81", 3) == 0) {
        out[o++] = ' ';
        i += 2;
      } else {
        out[o++] = s[i];
      }
    }
  }
  t->piece[id] = out;
  t->piece_len[id] = o;
  return RAYO_OK;
}

/* A vocabulary entry: its form for the model's lookups (bytes, for a
 * byte-level vocabulary). */
static rayo_status add_token(reader *r, const char *s, size_t n, rayo_i32 id) {
  rayo_tokenizer *t = r->t;
  char *form = rayo_arena_alloc(&t->arena, n + 1);
  size_t len = n;
  if (!form) return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (t->byte_level) {
    len = unmap_bytes(t->unbyte, s, n, form);
  } else {
    memcpy(form, s, n);
  }
  str_put(&t->vocab, form, len, id);
  if (len > t->max_len) t->max_len = len;
  return set_piece(r, id, s, n, RAYO_FALSE);
}

static rayo_status read_merges(reader *r, const rayo_json *merges) {
  rayo_tokenizer *t = r->t;
  char *buf = NULL;
  size_t i, cap = 0;
  if (!merges || merges->kind != RAYO_JSON_ARRAY) return RAYO_OK;
  if (!pair_init(&t->arena, &t->merges, merges->length)) {
    return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (i = 0; i < merges->length; i++) {
    const rayo_json *m = merges->items[i];
    const char *a, *b;
    size_t na, nb, la, lb;
    rayo_i32 ia, ib, im;
    if (m->kind == RAYO_JSON_STRING) {
      const char *sp = strchr(m->string, ' ');
      if (!sp) continue;
      a = m->string;
      na = (size_t)(sp - m->string);
      b = sp + 1;
      nb = strlen(b);
    } else if (m->kind == RAYO_JSON_ARRAY && m->length == 2 && m->items[0]->string &&
               m->items[1]->string) {
      a = m->items[0]->string;
      na = m->items[0]->length;
      b = m->items[1]->string;
      nb = m->items[1]->length;
    } else {
      free(buf);
      return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: merge %lu", (unsigned long)i);
    }
    if (na + nb + 1 > cap) {
      char *p;
      cap = 2 * (na + nb + 1);
      p = realloc(buf, cap);
      if (!p) {
        free(buf);
        return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
      }
      buf = p;
    }
    if (t->byte_level) {
      la = unmap_bytes(t->unbyte, a, na, buf);
      ia = str_get(&t->vocab, buf, la);
      lb = unmap_bytes(t->unbyte, b, nb, buf + la);
      ib = str_get(&t->vocab, buf + la, lb);
    } else {
      memcpy(buf, a, na);
      la = na;
      ia = str_get(&t->vocab, buf, la);
      memcpy(buf + la, b, nb);
      lb = nb;
      ib = str_get(&t->vocab, buf + la, lb);
    }
    im = str_get(&t->vocab, buf, la + lb);
    if (ia >= 0 && ib >= 0 && im >= 0) pair_put(&t->merges, ia, ib, (rayo_i32)i, im);
  }
  free(buf);
  return RAYO_OK;
}

static rayo_status read_model(reader *r, const rayo_json *model) {
  rayo_tokenizer *t = r->t;
  const char *type = str_of(model, "type");
  const rayo_json *vocab = rayo_json_get(model, "vocab");
  size_t i;
  rayo_status s;
  if (!type || !vocab) return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: no model");
  t->byte_fallback = bool_of(model, "byte_fallback", RAYO_FALSE);
  t->ignore_merges = bool_of(model, "ignore_merges", RAYO_FALSE);
  if (strcmp(type, "BPE") == 0) {
    const char *unk = str_of(model, "unk_token");
    if (vocab->kind != RAYO_JSON_OBJECT) {
      return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: a BPE vocab is an object");
    }
    for (i = 0; i < vocab->length; i++) {
      rayo_i32 id;
      if (!rayo_json_as_int(vocab->items[i], &id) || id < 0 || id >= t->vocab_size) {
        return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: id of %s", vocab->keys[i]);
      }
      s = add_token(r, vocab->keys[i], strlen(vocab->keys[i]), id);
      if (s != RAYO_OK) return s;
    }
    if (unk) t->unk = str_get(&t->vocab, unk, strlen(unk));
    return read_merges(r, rayo_json_get(model, "merges"));
  }
  if (strcmp(type, "Unigram") == 0) {
    rayo_i32 unk_id = -1;
    if (vocab->kind != RAYO_JSON_ARRAY) {
      return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: a Unigram vocab is a list");
    }
    t->unigram = RAYO_TRUE;
    t->scores = rayo_arena_array(&t->arena, (size_t)t->vocab_size, sizeof *t->scores);
    if (!t->scores) return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    t->min_score = 0.0f;
    for (i = 0; i < vocab->length; i++) {
      const rayo_json *e = vocab->items[i];
      if (e->kind != RAYO_JSON_ARRAY || e->length != 2 || !e->items[0]->string ||
          e->items[1]->kind != RAYO_JSON_NUMBER) {
        return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: Unigram entry %lu", (unsigned long)i);
      }
      s = add_token(r, e->items[0]->string, e->items[0]->length, (rayo_i32)i);
      if (s != RAYO_OK) return s;
      t->scores[i] = (float)e->items[1]->number;
      if (t->scores[i] < t->min_score) t->min_score = t->scores[i];
    }
    if (rayo_json_as_int(rayo_json_get(model, "unk_id"), &unk_id)) t->unk = unk_id;
    return RAYO_OK;
  }
  return rayo_fail(r->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: model %s (BPE and Unigram are read)",
                   type);
}

/* The largest id in the vocabulary and the added tokens. */
static rayo_i32 largest_id(const rayo_json *model, const rayo_json *added) {
  const rayo_json *vocab = rayo_json_get(model, "vocab");
  rayo_i32 top = -1, id;
  size_t i;
  if (vocab && vocab->kind == RAYO_JSON_OBJECT) {
    for (i = 0; i < vocab->length; i++) {
      if (rayo_json_as_int(vocab->items[i], &id) && id > top) top = id;
    }
  } else if (vocab && vocab->kind == RAYO_JSON_ARRAY) {
    top = (rayo_i32)vocab->length - 1;
  }
  for (i = 0; added && added->kind == RAYO_JSON_ARRAY && i < added->length; i++) {
    if (rayo_json_as_int(rayo_json_get(added->items[i], "id"), &id) && id > top) top = id;
  }
  return top;
}

static int longer_first(const void *a, const void *b) {
  const added_token *x = a, *y = b;
  return x->len < y->len ? 1 : (x->len > y->len ? -1 : 0);
}

static rayo_status read_added(reader *r, const rayo_json *added) {
  rayo_tokenizer *t = r->t;
  size_t i;
  if (!added || added->kind != RAYO_JSON_ARRAY || added->length == 0) return RAYO_OK;
  t->added = rayo_arena_array(&t->arena, added->length, sizeof *t->added);
  if (!t->added) return rayo_fail(r->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (i = 0; i < added->length; i++) {
    const rayo_json *a = added->items[i];
    const rayo_json *c = rayo_json_get(a, "content");
    rayo_i32 id;
    rayo_status s;
    if (!c || !c->string || c->length == 0 || !rayo_json_as_int(rayo_json_get(a, "id"), &id) ||
        id < 0 || id >= t->vocab_size) {
      return rayo_fail(r->msg, RAYO_ERR_FORMAT, "tokenizer: added token %lu", (unsigned long)i);
    }
    t->added[t->added_count].s = keep(t, c->string, c->length);
    t->added[t->added_count].len = c->length;
    t->added[t->added_count].id = id;
    t->added_count++;
    s = set_piece(r, id, c->string, c->length, RAYO_TRUE);
    if (s != RAYO_OK) return s;
  }
  qsort(t->added, t->added_count, sizeof *t->added, longer_first);
  return RAYO_OK;
}

rayo_status rayo_tokenizer_create(const char *json, size_t size, rayo_tokenizer **out,
                                  rayo_message *msg) {
  rayo_arena parse;
  rayo_json *root;
  const rayo_json *model, *added;
  rayo_tokenizer *t;
  reader r;
  rayo_status s;
  rayo_i32 top;
  int i;
  if (!out || !json) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *out = NULL;
  memset(&parse, 0, sizeof parse);
  s = rayo_json_parse(json, size, &parse, &root, msg);
  if (s != RAYO_OK) {
    rayo_arena_free(&parse);
    return s;
  }
  model = rayo_json_get(root, "model");
  added = rayo_json_get(root, "added_tokens");
  top = largest_id(model, added);
  t = calloc(1, sizeof *t);
  if (!t || top < 0 || top >= RAYO_I32_MAX - 1) {
    free(t);
    rayo_arena_free(&parse);
    return t ? rayo_fail(msg, RAYO_ERR_FORMAT, "tokenizer: no vocabulary")
             : rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  t->vocab_size = top + 1;
  t->unk = -1;
  for (i = 0; i < 256; i++) t->byte_ids[i] = -1;
  for (i = 0; i < 512; i++) t->unbyte[i] = (rayo_i16)char_byte((rayo_u32)i);
  r.t = t;
  r.msg = msg;
  t->piece = rayo_arena_array(&t->arena, (size_t)t->vocab_size, sizeof *t->piece);
  t->piece_len = rayo_arena_array(&t->arena, (size_t)t->vocab_size, sizeof *t->piece_len);
  if (!t->piece || !t->piece_len || !str_init(&t->arena, &t->vocab, (size_t)t->vocab_size)) {
    s = rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  /* The decoder says byte-level too, when the pre-tokeniser does not. */
  if (s == RAYO_OK) {
    const rayo_json *decoder = rayo_json_get(root, "decoder");
    const char *dec = str_of(decoder, "type");
    if (dec && strcmp(dec, "ByteLevel") == 0) t->byte_level = RAYO_TRUE;
    /* A decoder that writes SentencePiece's ▁ as a space. */
    if (decoder && decoder->raw &&
        (strstr_n(decoder->raw, decoder->raw_length, "\\u2581") ||
         strstr_n(decoder->raw, decoder->raw_length, "\xe2\x96\x81"))) {
      t->metaspace_decode = RAYO_TRUE;
    }
    s = read_normalizer(&r, rayo_json_get(root, "normalizer"));
  }
  if (s == RAYO_OK) s = read_pre(&r, rayo_json_get(root, "pre_tokenizer"));
  if (s == RAYO_OK) s = read_added(&r, added);
  if (s == RAYO_OK) s = read_model(&r, model);
  rayo_arena_free(&parse);
  if (s != RAYO_OK) {
    rayo_tokenizer_destroy(t);
    return s;
  }
  *out = t;
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
  free(tokenizer);
}

rayo_i32 rayo_tokenizer_vocab_size(const rayo_tokenizer *tokenizer) {
  return tokenizer ? tokenizer->vocab_size : 0;
}

const char *rayo_tokenizer_piece(const rayo_tokenizer *tokenizer, rayo_i32 id, size_t *size) {
  if (!tokenizer || id < 0 || id >= tokenizer->vocab_size || !tokenizer->piece[id]) {
    if (size) *size = 0;
    return NULL;
  }
  if (size) *size = tokenizer->piece_len[id];
  return tokenizer->piece[id];
}

rayo_i32 rayo_tokenizer_find(const rayo_tokenizer *tokenizer, const char *token) {
  size_t i, n;
  if (!tokenizer || !token) return -1;
  n = strlen(token);
  for (i = 0; i < tokenizer->added_count; i++) {
    if (tokenizer->added[i].len == n && memcmp(tokenizer->added[i].s, token, n) == 0) {
      return tokenizer->added[i].id;
    }
  }
  return str_get(&tokenizer->vocab, token, n);
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
  for (i = 0; all && i < n; i++) all = t->byte_ids[s[i]] >= 0;
  if (all) {
    for (i = 0; i < n; i++) out[(*count)++] = t->byte_ids[s[i]];
  } else if (t->unk >= 0) {
    out[(*count)++] = t->unk;
  }
}

static void bpe_word(encoder *e, const char *w, size_t n) {
  const rayo_tokenizer *t = e->t;
  size_t i = 0, count = 0, len;
  rayo_i32 whole;
  if (t->ignore_merges && (whole = str_get(&t->vocab, w, n)) >= 0) {
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
    id = str_get(&t->vocab, w + i, len);
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
      const pair_entry *p = pair_get(&t->merges, e->sym[j], e->sym[j + 1]);
      if (p && p->rank < rank) {
        rank = p->rank;
        merged = p->merged;
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
      const rayo_i32 id = str_get(&t->vocab, w + i, k);
      size_t next;
      if (id >= 0) {
        const float sc = e->best[i] + t->scores[id];
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
      last_unk = RAYO_TRUE;
      rayo_i32 tmp[8];
      size_t c = 0, j;
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
    if (st->kind == STEP_BYTELEVEL) {
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
    if (st->kind == NORM_PREPEND) {
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
      case STEP_SPLIT:
        ok = split_regex(e, &st->rx, st->removed, &b, &a, &c);
        break;
      case STEP_BYTELEVEL:
        rewrite(&b, st, first, &a, &c);
        if (st->use_regex) {
          spans d = a;
          a = c;
          c = d;
          c.count = 0;
          ok = split_regex(e, &st->rx, RAYO_FALSE, &b, &a, &c);
        }
        break;
      case STEP_DIGITS:
        digits(&b, st->individual, &a, &c);
        break;
      case STEP_METASPACE:
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
    const added_token *hit = NULL;
    for (k = 0; k < tokenizer->added_count; k++) {
      const added_token *a = &tokenizer->added[k];
      if (a->len <= size - i && memcmp(text + i, a->s, a->len) == 0) {
        hit = a;
        break;
      }
    }
    if (!hit) {
      i++;
      continue;
    }
    if (i > start) ok = segment(&e, text + start, i - start, start == 0);
    emit(&e, hit->id);
    i += hit->len;
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
