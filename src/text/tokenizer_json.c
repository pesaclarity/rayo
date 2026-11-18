/* A tokenizer.json read into the tokeniser's table (text/table.h): what the
 * file says, gathered into hash tables, then laid out as the table's bytes
 * in one canonical order, which rayo-convert's writer
 * (convert/rayo_convert/tokenizer_table.py) follows to the byte. */
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "core/message.h"
#include "rayo/config.h"
#include "text/table.h"
#include "text/unicode.h"

#include "core/small_begin.h"

/* A hash table of byte strings to ids, open addressing. */
typedef struct str_entry {
  const char *s;
  size_t len;
  rayo_i32 id;
} str_entry;

typedef struct str_map {
  str_entry *slots;
  size_t size;
} str_map;

typedef struct pair_entry {
  rayo_i32 a, b, rank, merged; /* a < 0: empty */
} pair_entry;

typedef struct pair_map {
  pair_entry *slots;
  size_t size;
} pair_map;

typedef struct added_token {
  const char *s;
  size_t len;
  rayo_i32 id;
} added_token;

typedef struct step {
  rayo_u32 kind, flags;
  const char *s, *s2; /* pre: the pattern or replacement; norm: from, to */
  size_t len, len2;
} step;

typedef struct builder {
  rayo_arena arena;
  rayo_message *msg;
  rayo_u32 flags;
  rayo_i32 unk;
  rayo_i32 vocab_size;
  const char **piece; /* per id */
  size_t *piece_len;
  str_map vocab;
  pair_map merges;
  float *scores;
  float min_score;
  size_t max_len;
  rayo_i32 byte_ids[256];
  rayo_i16 unbyte[512];
  added_token *added;
  size_t added_count;
  step pre[TT_MAX_STEPS];
  int pre_count;
  step norm[TT_MAX_STEPS];
  int norm_count;
} builder;

rayo_u32 rayo_toktab_hash_bytes(const char *s, size_t n) {
  rayo_u32 h = 2166136261u;
  size_t i;
  for (i = 0; i < n; i++) h = ((h ^ (rayo_u8)s[i]) * 16777619u) & 0xFFFFFFFFu;
  return h;
}

rayo_u32 rayo_toktab_hash_pair(rayo_i32 a, rayo_i32 b) {
  rayo_u32 h = ((rayo_u32)a * 2654435761u ^ ((rayo_u32)b + 0x9E3779B9u)) & 0xFFFFFFFFu;
  h ^= h >> 15;
  h = (h * 2246822519u) & 0xFFFFFFFFu;
  h ^= h >> 13;
  return h;
}

size_t rayo_toktab_slots(size_t n) {
  size_t size = 16;
  while (size < 2 * n + 1) size *= 2;
  return size;
}

static rayo_i32 str_get(const str_map *m, const char *s, size_t n) {
  size_t i = rayo_toktab_hash_bytes(s, n) & (m->size - 1);
  while (m->slots[i].s) {
    if (m->slots[i].len == n && memcmp(m->slots[i].s, s, n) == 0) return m->slots[i].id;
    i = (i + 1) & (m->size - 1);
  }
  return -1;
}

/* Adds s (kept by the caller) as id; an earlier entry for s stays. */
static void str_put(str_map *m, const char *s, size_t n, rayo_i32 id) {
  size_t i = rayo_toktab_hash_bytes(s, n) & (m->size - 1);
  while (m->slots[i].s) {
    if (m->slots[i].len == n && memcmp(m->slots[i].s, s, n) == 0) return;
    i = (i + 1) & (m->size - 1);
  }
  m->slots[i].s = s;
  m->slots[i].len = n;
  m->slots[i].id = id;
}

static void pair_put(pair_map *m, rayo_i32 a, rayo_i32 b, rayo_i32 rank, rayo_i32 merged) {
  size_t i = rayo_toktab_hash_pair(a, b) & (m->size - 1);
  while (m->slots[i].a >= 0) {
    if (m->slots[i].a == a && m->slots[i].b == b) return; /* the first rank stays */
    i = (i + 1) & (m->size - 1);
  }
  m->slots[i].a = a;
  m->slots[i].b = b;
  m->slots[i].rank = rank;
  m->slots[i].merged = merged;
}

/* ---- GPT-2's byte-to-character mapping ----------------------------------------------------- */

static rayo_bool byte_prints(unsigned b) {
  return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
}

static rayo_u32 byte_char(unsigned b) {
  unsigned i, n = 0;
  if (byte_prints(b)) return b;
  for (i = 0; i < b; i++) n += byte_prints(i) ? 0u : 1u;
  return 256u + n;
}

static int char_byte(rayo_u32 c) {
  unsigned b;
  if (c < 256) return byte_prints(c) ? (int)c : -1;
  for (b = 0; b < 256; b++) {
    if (!byte_prints(b) && byte_char(b) == c) return (int)b;
  }
  return -1;
}

/* s in the byte-level mapping, back to its bytes, into out (at least n
 * bytes); its length. Characters not in the mapping stay as they are. */
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

static rayo_bool contains(const char *s, size_t n, const char *needle) {
  const size_t m = strlen(needle);
  size_t i;
  for (i = 0; i + m <= n; i++) {
    if (memcmp(s + i, needle, m) == 0) return RAYO_TRUE;
  }
  return RAYO_FALSE;
}

static int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* ---- reading tokenizer.json ------------------------------------------------------------------ */

static const char *str_of(const rayo_json *j, const char *key) {
  return rayo_json_as_string(rayo_json_get(j, key));
}

static rayo_bool bool_of(const rayo_json *j, const char *key, rayo_bool fallback) {
  const rayo_json *v = rayo_json_get(j, key);
  return v && v->kind == RAYO_JSON_BOOL ? v->boolean : fallback;
}

static rayo_status read_normalizer(builder *t, const rayo_json *n) {
  const char *type;
  size_t i;
  if (!n || n->kind == RAYO_JSON_NULL) return RAYO_OK;
  type = str_of(n, "type");
  if (!type) return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: a normalizer without a type");
  if (strcmp(type, "Sequence") == 0) {
    const rayo_json *list = rayo_json_get(n, "normalizers");
    for (i = 0; list && list->kind == RAYO_JSON_ARRAY && i < list->length; i++) {
      rayo_status s = read_normalizer(t, list->items[i]);
      if (s != RAYO_OK) return s;
    }
    return RAYO_OK;
  }
  if (strcmp(type, "NFC") == 0) return RAYO_OK; /* text is taken to be NFC already */
  if (t->norm_count == TT_MAX_STEPS) {
    return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: too many normalizers");
  }
  if (strcmp(type, "Replace") == 0) {
    const char *from = str_of(rayo_json_get(n, "pattern"), "String");
    const char *to = str_of(n, "content");
    step *st = &t->norm[t->norm_count];
    if (!from || !to || !*from) {
      return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: Replace of a string only");
    }
    st->kind = TT_REPLACE;
    st->s = from;
    st->len = strlen(from);
    st->s2 = to;
    st->len2 = strlen(to);
    t->norm_count++;
    return RAYO_OK;
  }
  if (strcmp(type, "Prepend") == 0) {
    const char *to = str_of(n, "prepend");
    step *st = &t->norm[t->norm_count];
    if (!to) return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: Prepend without prepend");
    st->kind = TT_PREPEND;
    st->s = "";
    st->len = 0;
    st->s2 = to;
    st->len2 = strlen(to);
    t->norm_count++;
    return RAYO_OK;
  }
  return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: normalizer %s", type);
}

static rayo_status read_pre(builder *t, const rayo_json *n) {
  const char *type;
  step *st;
  size_t i;
  if (!n || n->kind == RAYO_JSON_NULL) return RAYO_OK;
  type = str_of(n, "type");
  if (!type) return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: a pre_tokenizer without a type");
  if (strcmp(type, "Sequence") == 0) {
    const rayo_json *list = rayo_json_get(n, "pretokenizers");
    for (i = 0; list && list->kind == RAYO_JSON_ARRAY && i < list->length; i++) {
      rayo_status s = read_pre(t, list->items[i]);
      if (s != RAYO_OK) return s;
    }
    return RAYO_OK;
  }
  if (t->pre_count == TT_MAX_STEPS) {
    return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: too many pre_tokenizers");
  }
  st = &t->pre[t->pre_count];
  memset(st, 0, sizeof *st);
  st->s = "";
  if (strcmp(type, "Split") == 0) {
    const char *pattern = str_of(rayo_json_get(n, "pattern"), "Regex");
    const char *literal = str_of(rayo_json_get(n, "pattern"), "String");
    const char *behavior = str_of(n, "behavior");
    if (bool_of(n, "invert", RAYO_FALSE) || !behavior ||
        (strcmp(behavior, "Isolated") != 0 && strcmp(behavior, "Removed") != 0)) {
      return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED,
                       "tokenizer: Split with behavior %s (Isolated and Removed are read)",
                       behavior ? behavior : "(none)");
    }
    if (!pattern && literal) {
      /* A literal string: escaped as a pattern. */
      char *esc = rayo_arena_alloc(&t->arena, 2 * strlen(literal) + 1);
      size_t o = 0;
      if (!esc) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
      for (i = 0; literal[i]; i++) {
        if (strchr("\\.^$|?*+()[]{}", literal[i])) esc[o++] = '\\';
        esc[o++] = literal[i];
      }
      pattern = esc;
    }
    if (!pattern) return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: Split without a pattern");
    st->kind = TT_SPLIT;
    st->flags = strcmp(behavior, "Removed") == 0 ? TT_REMOVED : 0u;
    st->s = pattern;
    st->len = strlen(pattern);
  } else if (strcmp(type, "ByteLevel") == 0) {
    st->kind = TT_BYTELEVEL;
    if (bool_of(n, "add_prefix_space", RAYO_FALSE)) st->flags |= TT_PREFIX;
    if (bool_of(n, "use_regex", RAYO_TRUE)) st->flags |= TT_USE_REGEX;
    t->flags |= TT_BYTE_LEVEL;
  } else if (strcmp(type, "Digits") == 0) {
    st->kind = TT_DIGITS;
    if (bool_of(n, "individual_digits", RAYO_FALSE)) st->flags |= TT_INDIVIDUAL;
  } else if (strcmp(type, "Metaspace") == 0) {
    const char *rep = str_of(n, "replacement");
    const char *scheme = str_of(n, "prepend_scheme");
    st->kind = TT_METASPACE;
    st->s = rep ? rep : "\xe2\x96\x81";
    st->len = strlen(st->s);
    if (scheme ? strcmp(scheme, "never") != 0 : bool_of(n, "add_prefix_space", RAYO_TRUE)) {
      st->flags |= TT_PREFIX;
    }
    if (scheme && strcmp(scheme, "first") == 0) st->flags |= TT_FIRST_ONLY;
    if (bool_of(n, "split", RAYO_TRUE)) st->flags |= TT_SPLIT_WORDS;
    t->flags |= TT_METASPACE_DECODE;
  } else {
    return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: pre_tokenizer %s", type);
  }
  t->pre_count++;
  return RAYO_OK;
}

/* Gives id its output text: the bytes of a byte-level token, a
 * SentencePiece token with ▁ as a space, a <0xXX> token as its byte. */
static rayo_status set_piece(builder *t, rayo_i32 id, const char *s, size_t n, rayo_bool special) {
  char *out;
  size_t o = 0, i;
  if (t->piece[id]) return RAYO_OK;
  out = rayo_arena_alloc(&t->arena, n + 1);
  if (!out) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (!(t->flags & TT_BYTE_LEVEL) && n == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' &&
      s[5] == '>' && hex_digit(s[3]) >= 0 && hex_digit(s[4]) >= 0) {
    /* A byte token, <0xXX>: that byte. */
    const int v = hex_digit(s[3]) * 16 + hex_digit(s[4]);
    out[o++] = (char)v;
    if (t->byte_ids[v] < 0) t->byte_ids[v] = id;
  } else if (special) {
    memcpy(out, s, n);
    o = n;
  } else if (t->flags & TT_BYTE_LEVEL) {
    o = unmap_bytes(t->unbyte, s, n, out);
  } else {
    for (i = 0; i < n; i++) {
      if ((t->flags & TT_METASPACE_DECODE) && i + 3 <= n && memcmp(s + i, "\xe2\x96\x81", 3) == 0) {
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
static rayo_status add_token(builder *t, const char *s, size_t n, rayo_i32 id) {
  char *form = rayo_arena_alloc(&t->arena, n + 1);
  size_t len = n;
  if (!form) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (t->flags & TT_BYTE_LEVEL) {
    len = unmap_bytes(t->unbyte, s, n, form);
  } else {
    memcpy(form, s, n);
  }
  str_put(&t->vocab, form, len, id);
  if (len > t->max_len) t->max_len = len;
  return set_piece(t, id, s, n, RAYO_FALSE);
}

static rayo_status read_merges(builder *t, const rayo_json *merges) {
  char *buf = NULL;
  size_t i, cap = 0;
  if (!merges || merges->kind != RAYO_JSON_ARRAY) return RAYO_OK;
  t->merges.size = rayo_toktab_slots(merges->length);
  t->merges.slots = rayo_arena_array(&t->arena, t->merges.size, sizeof *t->merges.slots);
  if (!t->merges.slots) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (i = 0; i < t->merges.size; i++) t->merges.slots[i].a = -1;
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
      return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: merge %lu", (unsigned long)i);
    }
    if (!buf || na + nb + 1 > cap) {
      char *p;
      cap = 2 * (na + nb + 1);
      p = realloc(buf, cap);
      if (!p) {
        free(buf);
        return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
      }
      buf = p;
    }
    if (t->flags & TT_BYTE_LEVEL) {
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

static rayo_status read_model(builder *t, const rayo_json *model) {
  const char *type = str_of(model, "type");
  const rayo_json *vocab = rayo_json_get(model, "vocab");
  size_t i;
  rayo_status s;
  if (!type || !vocab) return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: no model");
  if (bool_of(model, "byte_fallback", RAYO_FALSE)) t->flags |= TT_BYTE_FALLBACK;
  if (bool_of(model, "ignore_merges", RAYO_FALSE)) t->flags |= TT_IGNORE_MERGES;
  if (strcmp(type, "BPE") == 0) {
    const char *unk = str_of(model, "unk_token");
    if (vocab->kind != RAYO_JSON_OBJECT) {
      return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: a BPE vocab is an object");
    }
    for (i = 0; i < vocab->length; i++) {
      rayo_i32 id;
      if (!rayo_json_as_int(vocab->items[i], &id) || id < 0 || id >= t->vocab_size) {
        return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: id of %s", vocab->keys[i]);
      }
      s = add_token(t, vocab->keys[i], strlen(vocab->keys[i]), id);
      if (s != RAYO_OK) return s;
    }
    if (unk) t->unk = str_get(&t->vocab, unk, strlen(unk));
    return read_merges(t, rayo_json_get(model, "merges"));
  }
  if (strcmp(type, "Unigram") == 0) {
    rayo_i32 unk_id = -1;
    if (vocab->kind != RAYO_JSON_ARRAY) {
      return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: a Unigram vocab is a list");
    }
    t->flags |= TT_UNIGRAM;
    t->scores = rayo_arena_array(&t->arena, (size_t)t->vocab_size, sizeof *t->scores);
    if (!t->scores) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    t->min_score = 0.0f;
    for (i = 0; i < vocab->length; i++) {
      const rayo_json *e = vocab->items[i];
      if (e->kind != RAYO_JSON_ARRAY || e->length != 2 || !e->items[0]->string ||
          e->items[1]->kind != RAYO_JSON_NUMBER) {
        return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: Unigram entry %lu", (unsigned long)i);
      }
      /* A log probability: finite, as the table's reader requires. */
      if (!(e->items[1]->number >= -1.0e6 && e->items[1]->number <= 1.0e6)) {
        return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: Unigram score %lu", (unsigned long)i);
      }
      s = add_token(t, e->items[0]->string, e->items[0]->length, (rayo_i32)i);
      if (s != RAYO_OK) return s;
      t->scores[i] = (float)e->items[1]->number;
      if (t->scores[i] < t->min_score) t->min_score = t->scores[i];
    }
    if (rayo_json_as_int(rayo_json_get(model, "unk_id"), &unk_id)) t->unk = unk_id;
    return RAYO_OK;
  }
  return rayo_fail(t->msg, RAYO_ERR_UNSUPPORTED, "tokenizer: model %s (BPE and Unigram are read)",
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

/* The entries of the vocabulary and the added tokens. */
static size_t entries(const rayo_json *model, const rayo_json *added) {
  const rayo_json *vocab = rayo_json_get(model, "vocab");
  size_t n = vocab && (vocab->kind == RAYO_JSON_OBJECT || vocab->kind == RAYO_JSON_ARRAY)
                 ? vocab->length
                 : 0;
  return n + (added && added->kind == RAYO_JSON_ARRAY ? added->length : 0);
}

static rayo_status read_added(builder *t, const rayo_json *added) {
  size_t i, j;
  if (!added || added->kind != RAYO_JSON_ARRAY || added->length == 0) return RAYO_OK;
  t->added = rayo_arena_array(&t->arena, added->length, sizeof *t->added);
  if (!t->added) return rayo_fail(t->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (i = 0; i < added->length; i++) {
    const rayo_json *a = added->items[i];
    const rayo_json *c = rayo_json_get(a, "content");
    rayo_i32 id;
    rayo_status s;
    if (!c || !c->string || c->length == 0 || !rayo_json_as_int(rayo_json_get(a, "id"), &id) ||
        id < 0 || id >= t->vocab_size) {
      return rayo_fail(t->msg, RAYO_ERR_FORMAT, "tokenizer: added token %lu", (unsigned long)i);
    }
    t->added[t->added_count].s = c->string;
    t->added[t->added_count].len = c->length;
    t->added[t->added_count].id = id;
    t->added_count++;
    s = set_piece(t, id, c->string, c->length, RAYO_TRUE);
    if (s != RAYO_OK) return s;
  }
  /* Longest first, the file's order among equals (a stable insertion sort). */
  for (i = 1; i < t->added_count; i++) {
    const added_token x = t->added[i];
    for (j = i; j > 0 && t->added[j - 1].len < x.len; j--) t->added[j] = t->added[j - 1];
    t->added[j] = x;
  }
  return RAYO_OK;
}

/* ---- the table's bytes ----------------------------------------------------------------------- */

typedef struct packer {
  unsigned char *out;
  size_t pool_at; /* where the pool starts */
  size_t pool_len;
  rayo_u32 *piece_at; /* each id's piece's offset in the pool */
} packer;

static void put32(unsigned char *p, rayo_u32 v) {
  p[0] = (unsigned char)(v & 255u);
  p[1] = (unsigned char)((v >> 8) & 255u);
  p[2] = (unsigned char)((v >> 16) & 255u);
  p[3] = (unsigned char)((v >> 24) & 255u);
}

/* Appends n bytes to the pool (or only counts them, before `out` exists);
 * their offset in it. */
static rayo_u32 pool_add(packer *k, const char *s, size_t n) {
  const size_t at = k->pool_len;
  if (k->out && n) memcpy(k->out + k->pool_at + at, s, n);
  k->pool_len += n;
  return (rayo_u32)at;
}

static rayo_bool same_as_piece(const builder *t, rayo_i32 id, const char *s, size_t n) {
  return id >= 0 && id < t->vocab_size && t->piece[id] && t->piece_len[id] == n &&
         memcmp(t->piece[id], s, n) == 0;
}

/* The pool, in its canonical order: each id's piece; then each vocabulary
 * slot's string unless it is its id's piece; then each added token's,
 * likewise; then the steps' strings. Writes the records that point into
 * it, at their offsets, when `k->out` is set. */
static void lay_out(const builder *t, packer *k, const size_t *off) {
  unsigned char *o = k->out;
  size_t i;
  int s;
  k->pool_len = 0;
  for (i = 0; i < (size_t)t->vocab_size; i++) {
    rayo_u32 at = TT_NO_PIECE, len = 0;
    if (t->piece[i]) {
      at = pool_add(k, t->piece[i], t->piece_len[i]);
      len = (rayo_u32)t->piece_len[i];
    }
    k->piece_at[i] = at;
    if (o) {
      put32(o + off[2] + i * 8, at);
      put32(o + off[2] + i * 8 + 4, len);
    }
  }
  for (i = 0; i < t->vocab.size; i++) {
    const str_entry *e = &t->vocab.slots[i];
    rayo_u32 at = 0;
    if (!e->s) {
      if (o) {
        put32(o + off[0] + i * 12, 0);
        put32(o + off[0] + i * 12 + 4, 0);
        put32(o + off[0] + i * 12 + 8, 0xFFFFFFFFu);
      }
      continue;
    }
    if (same_as_piece(t, e->id, e->s, e->len)) {
      at = k->piece_at[e->id];
    } else {
      at = pool_add(k, e->s, e->len);
    }
    if (o) {
      put32(o + off[0] + i * 12, at);
      put32(o + off[0] + i * 12 + 4, (rayo_u32)e->len);
      put32(o + off[0] + i * 12 + 8, (rayo_u32)e->id);
    }
  }
  for (i = 0; i < t->added_count; i++) {
    const added_token *a = &t->added[i];
    rayo_u32 at;
    if (same_as_piece(t, a->id, a->s, a->len)) {
      at = k->piece_at[a->id];
    } else {
      at = pool_add(k, a->s, a->len);
    }
    if (o) {
      put32(o + off[4] + i * 12, at);
      put32(o + off[4] + i * 12 + 4, (rayo_u32)a->len);
      put32(o + off[4] + i * 12 + 8, (rayo_u32)a->id);
    }
  }
  for (s = 0; s < t->pre_count; s++) {
    const step *p = &t->pre[s];
    const rayo_u32 at = pool_add(k, p->s, p->len);
    if (o) {
      put32(o + off[5] + (size_t)s * 16, p->kind);
      put32(o + off[5] + (size_t)s * 16 + 4, p->flags);
      put32(o + off[5] + (size_t)s * 16 + 8, at);
      put32(o + off[5] + (size_t)s * 16 + 12, (rayo_u32)p->len);
    }
  }
  for (s = 0; s < t->norm_count; s++) {
    const step *p = &t->norm[s];
    const rayo_u32 from = pool_add(k, p->s, p->len);
    const rayo_u32 to = pool_add(k, p->s2, p->len2);
    if (o) {
      put32(o + off[6] + (size_t)s * 20, p->kind);
      put32(o + off[6] + (size_t)s * 20 + 4, from);
      put32(o + off[6] + (size_t)s * 20 + 8, (rayo_u32)p->len);
      put32(o + off[6] + (size_t)s * 20 + 12, to);
      put32(o + off[6] + (size_t)s * 20 + 16, (rayo_u32)p->len2);
    }
  }
}

static rayo_status pack(const builder *t, unsigned char **out, size_t *out_size,
                        rayo_message *msg) {
  /* Sections: vocabulary, merges, pieces, scores, added, pre, norm, byte
   * ids, pool. */
  size_t off[9], at = (size_t)TT_HEADER_WORDS * 4, total, i;
  const rayo_bool unigram = (t->flags & TT_UNIGRAM) != 0;
  packer k;
  unsigned char *o;
  rayo_u32 bits;
  off[0] = at;
  at += t->vocab.size * TT_VOCAB_WORDS * 4;
  off[1] = at;
  at += t->merges.size * TT_MERGE_WORDS * 4;
  off[2] = at;
  at += (size_t)t->vocab_size * TT_PIECE_WORDS * 4;
  off[3] = unigram ? at : 0;
  if (unigram) at += (size_t)t->vocab_size * 4;
  off[4] = at;
  at += t->added_count * TT_ADDED_WORDS * 4;
  off[5] = at;
  at += (size_t)t->pre_count * TT_PRE_WORDS * 4;
  off[6] = at;
  at += (size_t)t->norm_count * TT_NORM_WORDS * 4;
  off[7] = at;
  at += 256 * 4;
  off[8] = at;
  memset(&k, 0, sizeof k);
  k.piece_at = malloc((size_t)t->vocab_size * sizeof *k.piece_at);
  if (!k.piece_at) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  lay_out(t, &k, off); /* to count the pool */
  total = at + (k.pool_len + 3) / 4 * 4;
  o = total <= (size_t)RAYO_I32_MAX ? calloc(1, total) : NULL;
  if (!o) {
    free(k.piece_at);
    return rayo_fail(msg,
                     total <= (size_t)RAYO_I32_MAX ? RAYO_ERR_OUT_OF_MEMORY : RAYO_ERR_UNSUPPORTED,
                     "tokenizer: no memory for its table, or too large");
  }
  memcpy(o, "RTOK", 4);
  put32(o + 4 * TT_VERSION, RAYO_TOKTAB_VERSION);
  put32(o + 4 * TT_FLAGS, t->flags);
  put32(o + 4 * TT_UNK, (rayo_u32)t->unk);
  put32(o + 4 * TT_VOCAB_SIZE, (rayo_u32)t->vocab_size);
  put32(o + 4 * TT_MAX_LEN, (rayo_u32)t->max_len);
  memcpy(&bits, &t->min_score, 4);
  put32(o + 4 * TT_MIN_SCORE, bits);
  put32(o + 4 * TT_VOCAB_SLOTS, (rayo_u32)t->vocab.size);
  put32(o + 4 * TT_MERGE_SLOTS, (rayo_u32)t->merges.size);
  put32(o + 4 * TT_ADDED_COUNT, (rayo_u32)t->added_count);
  put32(o + 4 * TT_PRE_COUNT, (rayo_u32)t->pre_count);
  put32(o + 4 * TT_NORM_COUNT, (rayo_u32)t->norm_count);
  put32(o + 4 * TT_POOL_BYTES, (rayo_u32)k.pool_len);
  for (i = 0; i < 9; i++) put32(o + 4 * (TT_VOCAB_OFF + i), (rayo_u32)off[i]);
  put32(o + 4 * TT_TOTAL_BYTES, (rayo_u32)total);
  for (i = 0; i < t->merges.size; i++) {
    const pair_entry *e = &t->merges.slots[i];
    put32(o + off[1] + i * 16, (rayo_u32)e->a);
    put32(o + off[1] + i * 16 + 4, e->a < 0 ? 0u : (rayo_u32)e->b);
    put32(o + off[1] + i * 16 + 8, e->a < 0 ? 0u : (rayo_u32)e->rank);
    put32(o + off[1] + i * 16 + 12, e->a < 0 ? 0u : (rayo_u32)e->merged);
  }
  for (i = 0; unigram && i < (size_t)t->vocab_size; i++) {
    memcpy(&bits, &t->scores[i], 4);
    put32(o + off[3] + i * 4, bits);
  }
  for (i = 0; i < 256; i++) put32(o + off[7] + i * 4, (rayo_u32)t->byte_ids[i]);
  k.out = o;
  k.pool_at = off[8];
  lay_out(t, &k, off);
  free(k.piece_at);
  *out = o;
  *out_size = total;
  return RAYO_OK;
}

rayo_status rayo_toktab_from_json(const char *json, size_t size, unsigned char **out,
                                  size_t *out_size, rayo_message *msg) {
  rayo_arena parse;
  rayo_json *root;
  const rayo_json *model, *added, *decoder;
  builder t;
  rayo_status s;
  rayo_i32 top;
  int i;
  *out = NULL;
  *out_size = 0;
  memset(&parse, 0, sizeof parse);
  memset(&t, 0, sizeof t);
  t.msg = msg;
  if (size > (size_t)RAYO_MAX_TOKENIZER_JSON_BYTES) {
    return rayo_fail(msg, RAYO_ERR_LIMIT,
                     "tokenizer: %lu bytes, at most %ld (RAYO_MAX_TOKENIZER_JSON_BYTES)",
                     (unsigned long)size, (long)RAYO_MAX_TOKENIZER_JSON_BYTES);
  }
  s = rayo_json_parse(json, size, (size_t)RAYO_MAX_TOKENIZER_JSON_VALUES, &parse, &root, msg);
  if (s != RAYO_OK) {
    rayo_arena_free(&parse);
    return s;
  }
  model = rayo_json_get(root, "model");
  added = rayo_json_get(root, "added_tokens");
  top = largest_id(model, added);
  if (top < 0 || top >= RAYO_I32_MAX - 1) {
    rayo_arena_free(&parse);
    return rayo_fail(msg, RAYO_ERR_FORMAT, "tokenizer: no vocabulary");
  }
  /* The tables are as long as the largest id: it must be one of a
   * vocabulary as large as the file's entries (with room for gaps), not an
   * id that would make rayo allocate gigabytes for a small file. */
  {
    const size_t n = entries(model, added);
    if ((size_t)top >= (size_t)RAYO_MAX_VOCAB || (size_t)top > 2 * n + 1024) {
      rayo_arena_free(&parse);
      return rayo_fail(msg, RAYO_ERR_LIMIT,
                       "tokenizer: id %ld, for %lu entries (at most %ld, RAYO_MAX_VOCAB)",
                       (long)top, (unsigned long)n, (long)RAYO_MAX_VOCAB);
    }
  }
  t.vocab_size = top + 1;
  t.unk = -1;
  for (i = 0; i < 256; i++) t.byte_ids[i] = -1;
  for (i = 0; i < 512; i++) t.unbyte[i] = (rayo_i16)char_byte((rayo_u32)i);
  t.piece = rayo_arena_array(&t.arena, (size_t)t.vocab_size, sizeof *t.piece);
  t.piece_len = rayo_arena_array(&t.arena, (size_t)t.vocab_size, sizeof *t.piece_len);
  t.vocab.size = rayo_toktab_slots((size_t)t.vocab_size);
  t.vocab.slots = rayo_arena_array(&t.arena, t.vocab.size, sizeof *t.vocab.slots);
  if (!t.piece || !t.piece_len || !t.vocab.slots) {
    s = rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  /* The decoder says byte-level too, when the pre-tokeniser does not, and
   * whether SentencePiece's ▁ shows as a space. */
  decoder = rayo_json_get(root, "decoder");
  if (s == RAYO_OK) {
    const char *dec = str_of(decoder, "type");
    if (dec && strcmp(dec, "ByteLevel") == 0) t.flags |= TT_BYTE_LEVEL;
    if (decoder && decoder->raw &&
        (contains(decoder->raw, decoder->raw_length, "\\u2581") ||
         contains(decoder->raw, decoder->raw_length, "\xe2\x96\x81"))) {
      t.flags |= TT_METASPACE_DECODE;
    }
    s = read_normalizer(&t, rayo_json_get(root, "normalizer"));
  }
  if (s == RAYO_OK) s = read_pre(&t, rayo_json_get(root, "pre_tokenizer"));
  if (s == RAYO_OK) s = read_added(&t, added);
  if (s == RAYO_OK) s = read_model(&t, model);
  if (s == RAYO_OK) s = pack(&t, out, out_size, msg);
  rayo_arena_free(&t.arena);
  rayo_arena_free(&parse);
  return s;
}

#include "core/small_end.h"
