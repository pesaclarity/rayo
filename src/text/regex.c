#include "text/regex.h"

#include <string.h>

#include "core/message.h"
#include "rayo/config.h"
#include "text/unicode.h"

#include "core/small_begin.h"

typedef enum rx_kind { RX_CHAR, RX_CLASS, RX_ANY, RX_SEQ, RX_ALT, RX_REPEAT, RX_LOOK } rx_kind;

typedef enum rx_set { SET_RANGE, SET_LETTER, SET_NUMBER, SET_SPACE, SET_DIGIT, SET_WORD } rx_set;

typedef struct rx_item {
  rx_set set;
  rayo_bool negate; /* \S, \P{L} and the like */
  rayo_u32 lo, hi;  /* SET_RANGE */
} rx_item;

#define RX_MAX_ITEMS 32
#define RX_MAX_KIDS 64
#define RX_MAX_GROUPS 16 /* groups nested */
#define RX_MAX_DEPTH 512 /* the matcher's recursion, a few frames a level */

struct rayo_rx_node {
  rx_kind kind;
  rayo_bool icase;
  rayo_bool negate; /* a class's ^, a lookahead's ! */
  rayo_u32 c;       /* RX_CHAR */
  rx_item *items;   /* RX_CLASS */
  int item_count;
  rayo_rx_node **kids; /* RX_SEQ, RX_ALT */
  int kid_count;
  rayo_rx_node *child; /* RX_REPEAT, RX_LOOK */
  long min, max;       /* RX_REPEAT; max -1: no limit */
};

typedef struct parser {
  const unsigned char *s;
  size_t at, n;
  rayo_arena *arena;
  rayo_message *msg;
  rayo_status status;
  int groups; /* open now */
} parser;

static rayo_rx_node *fail(parser *p, const char *what) {
  if (p->status == RAYO_OK) {
    p->status = rayo_fail(p->msg, RAYO_ERR_UNSUPPORTED, "tokenizer pattern: %s at %lu", what,
                          (unsigned long)p->at);
  }
  return NULL;
}

static rayo_rx_node *node(parser *p, rx_kind kind, rayo_bool icase) {
  rayo_rx_node *n = rayo_arena_alloc(p->arena, sizeof *n);
  if (!n) {
    if (p->status == RAYO_OK)
      p->status = rayo_fail(p->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    return NULL;
  }
  n->kind = kind;
  n->icase = icase;
  return n;
}

static rayo_bool more(const parser *p) {
  return p->at < p->n;
}
static unsigned char peek(const parser *p) {
  return more(p) ? p->s[p->at] : 0;
}

static rayo_u32 next_char(parser *p) {
  size_t len;
  const rayo_u32 c = rayo_utf8_next(p->s + p->at, p->n - p->at, &len);
  p->at += len;
  return c;
}

/* \p{L}, \pN and the like, after the p or P. */
static rayo_bool read_property(parser *p, rx_item *it) {
  char name[8];
  size_t i = 0;
  if (peek(p) == '{') {
    p->at++;
    while (more(p) && peek(p) != '}' && i + 1 < sizeof name) name[i++] = (char)p->s[p->at++];
    if (peek(p) != '}') return RAYO_FALSE;
    p->at++;
  } else if (more(p)) {
    name[i++] = (char)p->s[p->at++];
  }
  name[i] = 0;
  if (strcmp(name, "L") == 0) {
    it->set = SET_LETTER;
  } else if (strcmp(name, "N") == 0) {
    it->set = SET_NUMBER;
  } else {
    return RAYO_FALSE;
  }
  return RAYO_TRUE;
}

/* An escape after the backslash: a set (`*is_set`) or a character. */
static rayo_bool read_escape(parser *p, rx_item *it, rayo_bool *is_set) {
  const unsigned char e = peek(p);
  if (!more(p)) return RAYO_FALSE;
  p->at++;
  *is_set = RAYO_TRUE;
  it->negate = e == 'S' || e == 'D' || e == 'W' || e == 'P';
  switch (e) {
    case 's':
    case 'S':
      it->set = SET_SPACE;
      return RAYO_TRUE;
    case 'd':
    case 'D':
      it->set = SET_DIGIT;
      return RAYO_TRUE;
    case 'w':
    case 'W':
      it->set = SET_WORD;
      return RAYO_TRUE;
    case 'p':
    case 'P':
      return read_property(p, it);
    default:
      break;
  }
  *is_set = RAYO_FALSE;
  it->negate = RAYO_FALSE;
  it->set = SET_RANGE;
  switch (e) {
    case 'r':
      it->lo = '\r';
      break;
    case 'n':
      it->lo = '\n';
      break;
    case 't':
      it->lo = '\t';
      break;
    case 'f':
      it->lo = '\f';
      break;
    case 'v':
      it->lo = '\v';
      break;
    default:
      if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9')) {
        return RAYO_FALSE; /* an escape this reader does not know */
      }
      p->at--;
      it->lo = next_char(p);
  }
  it->hi = it->lo;
  return RAYO_TRUE;
}

static rayo_rx_node *parse_class(parser *p, rayo_bool icase) {
  rayo_rx_node *n = node(p, RX_CLASS, icase);
  if (!n) return NULL;
  n->items = rayo_arena_array(p->arena, RX_MAX_ITEMS, sizeof *n->items);
  if (!n->items) return fail(p, "out of memory");
  if (peek(p) == '^') {
    n->negate = RAYO_TRUE;
    p->at++;
  }
  while (more(p) && (peek(p) != ']' || n->item_count == 0)) {
    rx_item it;
    rayo_bool is_set = RAYO_FALSE;
    memset(&it, 0, sizeof it);
    if (n->item_count == RX_MAX_ITEMS) return fail(p, "a class too long");
    if (peek(p) == '[') return fail(p, "a class within a class");
    if (peek(p) == '\\') {
      p->at++;
      if (!read_escape(p, &it, &is_set)) return fail(p, "an escape");
    } else {
      it.set = SET_RANGE;
      it.lo = it.hi = next_char(p);
    }
    if (!is_set && peek(p) == '-' && p->at + 1 < p->n && p->s[p->at + 1] != ']') {
      rx_item hi;
      rayo_bool hs = RAYO_FALSE;
      p->at++;
      memset(&hi, 0, sizeof hi);
      if (peek(p) == '\\') {
        p->at++;
        if (!read_escape(p, &hi, &hs) || hs) return fail(p, "a range");
      } else {
        hi.lo = next_char(p);
      }
      it.hi = hi.lo;
      if (it.hi < it.lo) return fail(p, "a range");
    }
    n->items[n->item_count++] = it;
  }
  if (peek(p) != ']') return fail(p, "a class without ]");
  p->at++;
  return n;
}

static rayo_rx_node *parse_alt(parser *p, rayo_bool icase);

static rayo_rx_node *parse_atom(parser *p, rayo_bool icase) {
  const unsigned char c = peek(p);
  rayo_rx_node *n;
  if (c == '(') {
    rx_kind kind = RX_SEQ;
    rayo_bool neg = RAYO_FALSE, ic = icase;
    p->at++;
    if (peek(p) == '?') {
      p->at++;
      if (peek(p) == ':') {
        p->at++;
      } else if (peek(p) == 'i' && p->at + 1 < p->n && p->s[p->at + 1] == ':') {
        p->at += 2;
        ic = RAYO_TRUE;
      } else if (peek(p) == '!' || peek(p) == '=') {
        kind = RX_LOOK;
        neg = peek(p) == '!';
        p->at++;
      } else {
        return fail(p, "a group kind");
      }
    }
    if (++p->groups > RX_MAX_GROUPS) return fail(p, "groups nested too deeply");
    n = parse_alt(p, ic);
    p->groups--;
    if (!n) return NULL;
    if (peek(p) != ')') return fail(p, "a group without )");
    p->at++;
    if (kind == RX_LOOK) {
      rayo_rx_node *look = node(p, RX_LOOK, ic);
      if (!look) return NULL;
      look->negate = neg;
      look->child = n;
      return look;
    }
    return n;
  }
  if (c == '[') {
    p->at++;
    return parse_class(p, icase);
  }
  if (c == '.') {
    p->at++;
    return node(p, RX_ANY, icase);
  }
  if (c == '\\') {
    rx_item it;
    rayo_bool is_set = RAYO_FALSE;
    p->at++;
    memset(&it, 0, sizeof it);
    if (!read_escape(p, &it, &is_set)) return fail(p, "an escape");
    if (is_set) {
      n = node(p, RX_CLASS, icase);
      if (!n) return NULL;
      n->items = rayo_arena_alloc(p->arena, sizeof *n->items);
      if (!n->items) return fail(p, "out of memory");
      n->items[0] = it;
      n->item_count = 1;
      return n;
    }
    n = node(p, RX_CHAR, icase);
    if (n) n->c = it.lo;
    return n;
  }
  if (c == '^' || c == '$') return fail(p, "anchors");
  n = node(p, RX_CHAR, icase);
  if (n) n->c = next_char(p);
  return n;
}

static rayo_bool read_count(parser *p, long *out) {
  long v = 0;
  if (!(peek(p) >= '0' && peek(p) <= '9')) return RAYO_FALSE;
  while (peek(p) >= '0' && peek(p) <= '9' && v < 100000) v = v * 10 + (p->s[p->at++] - '0');
  *out = v;
  return RAYO_TRUE;
}

static rayo_rx_node *parse_quant(parser *p, rayo_bool icase) {
  rayo_rx_node *atom = parse_atom(p, icase), *r;
  long lo, hi;
  const unsigned char q = peek(p);
  if (!atom) return NULL;
  if (q == '?' || q == '*' || q == '+') {
    p->at++;
    lo = q == '+' ? 1 : 0;
    hi = q == '?' ? 1 : -1;
  } else if (q == '{') {
    p->at++;
    if (!read_count(p, &lo)) return fail(p, "a count");
    hi = lo;
    if (peek(p) == ',') {
      p->at++;
      hi = -1;
      if (peek(p) != '}' && (!read_count(p, &hi) || hi < lo)) return fail(p, "a count");
    }
    if (peek(p) != '}') return fail(p, "a count without }");
    p->at++;
  } else {
    return atom;
  }
  if (peek(p) == '?' || peek(p) == '+') return fail(p, "lazy or possessive quantifiers");
  if (atom->kind != RX_CHAR && atom->kind != RX_CLASS && atom->kind != RX_ANY) {
    return fail(p, "a quantifier on more than one character");
  }
  r = node(p, RX_REPEAT, icase);
  if (!r) return NULL;
  r->child = atom;
  r->min = lo;
  r->max = hi;
  return r;
}

static rayo_rx_node *parse_seq(parser *p, rayo_bool icase) {
  rayo_rx_node *n = node(p, RX_SEQ, icase);
  if (!n) return NULL;
  n->kids = rayo_arena_array(p->arena, RX_MAX_KIDS, sizeof *n->kids);
  if (!n->kids) return fail(p, "out of memory");
  while (more(p) && peek(p) != '|' && peek(p) != ')') {
    rayo_rx_node *k;
    if (n->kid_count == RX_MAX_KIDS) return fail(p, "a sequence too long");
    k = parse_quant(p, icase);
    if (!k) return NULL;
    n->kids[n->kid_count++] = k;
  }
  return n;
}

static rayo_rx_node *parse_alt(parser *p, rayo_bool icase) {
  rayo_rx_node *n = node(p, RX_ALT, icase);
  if (!n) return NULL;
  n->kids = rayo_arena_array(p->arena, RX_MAX_KIDS, sizeof *n->kids);
  if (!n->kids) return fail(p, "out of memory");
  for (;;) {
    rayo_rx_node *k;
    if (n->kid_count == RX_MAX_KIDS) return fail(p, "too many alternatives");
    k = parse_seq(p, icase);
    if (!k) return NULL;
    n->kids[n->kid_count++] = k;
    if (peek(p) != '|') break;
    p->at++;
  }
  return n;
}

rayo_status rayo_regex_compile(const char *pattern, rayo_arena *arena, rayo_regex *out,
                               rayo_message *msg) {
  parser p;
  rayo_rx_node *root;
  p.s = (const unsigned char *)pattern;
  p.at = 0;
  p.n = strlen(pattern);
  p.arena = arena;
  p.msg = msg;
  p.status = RAYO_OK;
  p.groups = 0;
  if (p.n > RAYO_MAX_PATTERN_BYTES) {
    return rayo_fail(msg, RAYO_ERR_LIMIT,
                     "tokenizer pattern: %lu bytes, at most %d (RAYO_MAX_PATTERN_BYTES)",
                     (unsigned long)p.n, RAYO_MAX_PATTERN_BYTES);
  }
  root = parse_alt(&p, RAYO_FALSE);
  if (root && more(&p)) root = fail(&p, "an unmatched )");
  if (!root) return p.status != RAYO_OK ? p.status : RAYO_ERR_UNSUPPORTED;
  out->root = root;
  return RAYO_OK;
}

/* Matching. */

static rayo_u32 fold(rayo_u32 c) {
  return c >= 'A' && c <= 'Z' ? c + 32 : c;
}

static rayo_bool in_set(const rx_item *it, rayo_u32 c, rayo_bool icase) {
  rayo_bool r;
  switch (it->set) {
    case SET_LETTER:
      r = rayo_unicode_is_letter(c);
      break;
    case SET_NUMBER:
      r = rayo_unicode_is_number(c);
      break;
    case SET_SPACE:
      r = rayo_unicode_is_space(c);
      break;
    case SET_DIGIT:
      r = c >= '0' && c <= '9';
      break;
    case SET_WORD:
      r = rayo_unicode_is_letter(c) || rayo_unicode_is_number(c) || c == '_';
      break;
    default:
      r = (c >= it->lo && c <= it->hi) ||
          (icase && fold(c) >= fold(it->lo) && fold(c) <= fold(it->hi));
  }
  return it->negate ? !r : r;
}

static rayo_bool one(const rayo_rx_node *n, rayo_u32 c) {
  int i;
  switch (n->kind) {
    case RX_CHAR:
      return c == n->c || (n->icase && fold(c) == fold(n->c));
    case RX_ANY:
      return c != '\n';
    case RX_CLASS:
      for (i = 0; i < n->item_count; i++) {
        if (in_set(&n->items[i], c, n->icase)) return !n->negate;
      }
      return n->negate;
    default:
      return RAYO_FALSE;
  }
}

/* What is left to match after a node: the rest of a sequence, then what
 * follows that sequence. */
typedef struct rx_cont {
  const rayo_rx_node *seq;
  int index;
  const struct rx_cont *next;
} rx_cont;

typedef struct rx_text {
  const rayo_u32 *s;
  size_t n;
  long *budget; /* steps left */
  int depth;    /* the matcher's recursion now */
} rx_text;

static long match(rx_text *t, const rayo_rx_node *n, size_t at, const rx_cont *k);

static long rest(rx_text *t, const rx_cont *k, size_t at) {
  rx_cont c;
  if (!k) return (long)at;
  if (k->index >= k->seq->kid_count) return rest(t, k->next, at);
  c.seq = k->seq;
  c.index = k->index + 1;
  c.next = k->next;
  return match(t, k->seq->kids[k->index], at, &c);
}

static long match_node(rx_text *t, const rayo_rx_node *n, size_t at, const rx_cont *k) {
  long e;
  int i;
  switch (n->kind) {
    case RX_CHAR:
    case RX_CLASS:
    case RX_ANY:
      return at < t->n && one(n, t->s[at]) ? rest(t, k, at + 1) : -1;
    case RX_SEQ: {
      rx_cont c;
      c.seq = n;
      c.index = 0;
      c.next = k;
      return rest(t, &c, at);
    }
    case RX_ALT:
      for (i = 0; i < n->kid_count; i++) {
        e = match(t, n->kids[i], at, k);
        if (e >= 0 || e == RAYO_REGEX_TOO_COSTLY) return e;
      }
      return -1;
    case RX_REPEAT: {
      size_t run = 0;
      while (at + run < t->n && (n->max < 0 || (long)run < n->max) &&
             one(n->child, t->s[at + run])) {
        run++;
      }
      *t->budget -= (long)run;
      for (;;) {
        if ((long)run < n->min) return -1;
        if (--*t->budget < 0) return RAYO_REGEX_TOO_COSTLY;
        e = rest(t, k, at + run);
        if (e >= 0 || e == RAYO_REGEX_TOO_COSTLY || run == 0) return e;
        run--;
      }
    }
    case RX_LOOK:
      e = match(t, n->child, at, NULL);
      if (e == RAYO_REGEX_TOO_COSTLY) return e;
      return (e >= 0) != n->negate ? rest(t, k, at) : -1;
  }
  return -1;
}

static long match(rx_text *t, const rayo_rx_node *n, size_t at, const rx_cont *k) {
  long e;
  if (--*t->budget < 0 || t->depth >= RX_MAX_DEPTH) return RAYO_REGEX_TOO_COSTLY;
  t->depth++;
  e = match_node(t, n, at, k);
  t->depth--;
  return e;
}

long rayo_regex_match(const rayo_regex *rx, const rayo_u32 *text, size_t n, size_t at,
                      long *budget) {
  rx_text t;
  t.s = text;
  t.n = n;
  t.budget = budget;
  t.depth = 0;
  return match(&t, rx->root, at, NULL);
}

#include "core/small_end.h"
