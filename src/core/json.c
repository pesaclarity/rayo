#include "core/json.h"

#include <float.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/message.h"
#include "rayo/config.h"

#include "core/small_begin.h"

typedef struct parser {
  const char *at;
  const char *end;
  rayo_arena *arena;
  rayo_message *msg;
  size_t values_left; /* the most values still allowed */
} parser;

static rayo_status fail(parser *p, const char *what) {
  return rayo_fail(p->msg, RAYO_ERR_FORMAT, "bad JSON in the header: %s", what);
}

static rayo_status no_memory(parser *p) {
  return rayo_fail(p->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
}

static void skip_ws(parser *p) {
  while (p->at < p->end && (*p->at == ' ' || *p->at == '\t' || *p->at == '\n' || *p->at == '\r')) {
    p->at++;
  }
}

static rayo_status parse_value(parser *p, int depth, rayo_json **out);

/* The length of the UTF-8 sequence at s (before end), or 0 when it is not
 * one: overlong forms, surrogates and values past U+10FFFF are not. */
static size_t utf8_length(const unsigned char *s, const unsigned char *end) {
  const unsigned b = s[0];
  size_t need, i;
  unsigned long c;
  if (b < 0x80) return 1;
  if (b >= 0xC2 && b <= 0xDF) {
    need = 2;
    c = b & 0x1Fu;
  } else if (b >= 0xE0 && b <= 0xEF) {
    need = 3;
    c = b & 0x0Fu;
  } else if (b >= 0xF0 && b <= 0xF4) {
    need = 4;
    c = b & 0x07u;
  } else {
    return 0;
  }
  if ((size_t)(end - s) < need) return 0;
  for (i = 1; i < need; i++) {
    if ((s[i] & 0xC0u) != 0x80u) return 0;
    c = c << 6 | (s[i] & 0x3Fu);
  }
  if ((need == 3 && (c < 0x800 || (c >= 0xD800 && c <= 0xDFFF))) ||
      (need == 4 && (c < 0x10000 || c > 0x10FFFF))) {
    return 0;
  }
  return need;
}

static int hex(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static size_t put_utf8(char *to, rayo_u32 cp) {
  if (cp < 0x80) {
    to[0] = (char)cp;
    return 1;
  }
  if (cp < 0x800) {
    to[0] = (char)(0xC0 | (cp >> 6));
    to[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    to[0] = (char)(0xE0 | (cp >> 12));
    to[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    to[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  to[0] = (char)(0xF0 | (cp >> 18));
  to[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  to[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  to[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

static rayo_bool read_hex4(parser *p, rayo_u32 *out) {
  rayo_u32 v = 0;
  int i;
  if (p->end - p->at < 4) return RAYO_FALSE;
  for (i = 0; i < 4; i++) {
    int h = hex(p->at[i]);
    if (h < 0) return RAYO_FALSE;
    v = v * 16 + (rayo_u32)h;
  }
  p->at += 4;
  *out = v;
  return RAYO_TRUE;
}

/* A \u escape (after the "\u"), with its low surrogate when it has one. */
static rayo_status read_escape_u(parser *p, const char *end, rayo_u32 *cp) {
  rayo_u32 lo;
  if (!read_hex4(p, cp)) return fail(p, "bad \\u escape");
  if (*cp >= 0xD800 && *cp <= 0xDBFF) {
    if (end - p->at < 6 || p->at[0] != '\\' || p->at[1] != 'u') return fail(p, "lone surrogate");
    p->at += 2;
    if (!read_hex4(p, &lo) || lo < 0xDC00 || lo > 0xDFFF) return fail(p, "lone surrogate");
    *cp = 0x10000 + ((*cp - 0xD800) << 10) + (lo - 0xDC00);
  } else if (*cp >= 0xDC00 && *cp <= 0xDFFF) {
    return fail(p, "lone surrogate");
  }
  if (*cp == 0) return fail(p, "NUL in a string");
  return RAYO_OK;
}

static int simple_escape(char e) {
  switch (e) {
    case '"':
      return '"';
    case '\\':
      return '\\';
    case '/':
      return '/';
    case 'b':
      return '\b';
    case 'f':
      return '\f';
    case 'n':
      return '\n';
    case 'r':
      return '\r';
    case 't':
      return '\t';
  }
  return -1;
}

/* After the opening quote. The unescaped text is never longer than the
 * escaped one, so one allocation of the escaped length suffices. */
static rayo_status parse_string(parser *p, const char **out, size_t *len) {
  const char *q = p->at;
  char *s;
  size_t n = 0;
  while (q < p->end && *q != '"') q += (*q == '\\' && q + 1 < p->end) ? 2 : 1;
  if (q >= p->end) return fail(p, "unterminated string");
  s = (char *)rayo_arena_alloc(p->arena, (size_t)(q - p->at) + 1);
  if (!s) return no_memory(p);
  while (p->at < q) {
    unsigned char c = (unsigned char)*p->at++;
    char e;
    int simple;
    if (c < 0x20) return fail(p, "control character in a string");
    if (c >= 0x80) {
      /* Text is UTF-8 (RFC 8259): names and metadata reach callers that
       * decode it, such as the Dart package. */
      const size_t seq = utf8_length((const unsigned char *)p->at - 1, (const unsigned char *)q);
      if (seq == 0) return fail(p, "a string that is not UTF-8");
      memcpy(s + n, p->at - 1, seq);
      n += seq;
      p->at += seq - 1;
      continue;
    }
    if (c != '\\') {
      s[n++] = (char)c;
      continue;
    }
    e = *p->at++;
    simple = simple_escape(e);
    if (simple >= 0) {
      s[n++] = (char)simple;
    } else if (e == 'u') {
      rayo_u32 cp = 0;
      rayo_status st = read_escape_u(p, q, &cp);
      if (st != RAYO_OK) return st;
      n += put_utf8(s + n, cp);
    } else {
      return fail(p, "bad escape");
    }
  }
  p->at = q + 1;
  s[n] = '\0';
  *out = s;
  *len = n;
  return RAYO_OK;
}

static const char *skip_digits(const char *s, const char *end) {
  while (s < end && *s >= '0' && *s <= '9') s++;
  return s;
}

static rayo_status parse_number(parser *p, double *out) {
  const char *s = p->at;
  char buf[64];
  size_t n;
  if (s < p->end && *s == '-') s++;
  if (s >= p->end) return fail(p, "bad number");
  if (*s == '0') {
    s++;
  } else if (*s >= '1' && *s <= '9') {
    s = skip_digits(s, p->end);
  } else {
    return fail(p, "bad number");
  }
  if (s < p->end && *s == '.') {
    s++;
    if (s >= p->end || *s < '0' || *s > '9') return fail(p, "bad number");
    s = skip_digits(s, p->end);
  }
  if (s < p->end && (*s == 'e' || *s == 'E')) {
    s++;
    if (s < p->end && (*s == '+' || *s == '-')) s++;
    if (s >= p->end || *s < '0' || *s > '9') return fail(p, "bad number");
    s = skip_digits(s, p->end);
  }
  n = (size_t)(s - p->at);
  if (n >= sizeof buf) return fail(p, "number too long");
  memcpy(buf, p->at, n);
  buf[n] = '\0';
  {
    /* strtod reads the C library's locale: JSON's '.' as its decimal
     * point, so a program that sets a locale reads the same numbers. */
    const char point = localeconv()->decimal_point[0];
    char *dot = strchr(buf, '.');
    if (dot && point && point != '.') *dot = point;
  }
  *out = strtod(buf, NULL);
  if (!(*out <= DBL_MAX && *out >= -DBL_MAX)) return fail(p, "number out of range");
  p->at = s;
  return RAYO_OK;
}

static rayo_bool literal(parser *p, const char *word) {
  size_t n = strlen(word);
  if ((size_t)(p->end - p->at) < n || memcmp(p->at, word, n) != 0) return RAYO_FALSE;
  p->at += n;
  return RAYO_TRUE;
}

/* A growing array of items (and keys) in the arena. */
typedef struct items {
  rayo_json **values;
  const char **keys;
  size_t n, cap;
} items;

static rayo_status push(parser *p, items *it, const char *key, rayo_json *v) {
  if (it->n == it->cap) {
    size_t cap = it->cap ? it->cap * 2 : 8;
    rayo_json **values = (rayo_json **)rayo_arena_array(p->arena, cap, sizeof *values);
    const char **keys = (const char **)rayo_arena_array(p->arena, cap, sizeof *keys);
    if (!values || !keys) return no_memory(p);
    if (it->n) {
      memcpy(values, it->values, it->n * sizeof *values);
      memcpy(keys, it->keys, it->n * sizeof *keys);
    }
    it->values = values;
    it->keys = keys;
    it->cap = cap;
  }
  it->values[it->n] = v;
  it->keys[it->n] = key;
  it->n++;
  return RAYO_OK;
}

static rayo_status parse_key(parser *p, const char **key) {
  size_t key_len;
  rayo_status s;
  skip_ws(p);
  if (p->at >= p->end || *p->at != '"') return fail(p, "expected a key");
  p->at++;
  s = parse_string(p, key, &key_len);
  if (s != RAYO_OK) return s;
  skip_ws(p);
  if (p->at >= p->end || *p->at != ':') return fail(p, "expected ':'");
  p->at++;
  return RAYO_OK;
}

static rayo_status parse_container(parser *p, int depth, rayo_bool object, rayo_json *v) {
  items it;
  char close = object ? '}' : ']';
  it.values = NULL;
  it.keys = NULL;
  it.n = it.cap = 0;
  skip_ws(p);
  if (p->at < p->end && *p->at == close) {
    p->at++;
  } else {
    for (;;) {
      const char *key = NULL;
      rayo_json *item = NULL;
      rayo_status s;
      if (object) {
        s = parse_key(p, &key);
        if (s != RAYO_OK) return s;
      }
      s = parse_value(p, depth + 1, &item);
      if (s != RAYO_OK) return s;
      s = push(p, &it, key, item);
      if (s != RAYO_OK) return s;
      skip_ws(p);
      if (p->at < p->end && *p->at == ',') {
        p->at++;
        continue;
      }
      if (p->at < p->end && *p->at == close) {
        p->at++;
        break;
      }
      return fail(p, object ? "expected ',' or '}'" : "expected ',' or ']'");
    }
  }
  v->items = it.values;
  v->keys = object ? it.keys : NULL;
  v->length = it.n;
  return RAYO_OK;
}

static rayo_status parse_value(parser *p, int depth, rayo_json **out) {
  rayo_json *v;
  rayo_status s = RAYO_OK;
  char c;
  if (depth > RAYO_JSON_MAX_DEPTH) return fail(p, "nested too deep");
  if (p->values_left == 0) {
    return rayo_fail(p->msg, RAYO_ERR_LIMIT, "JSON with too many values for rayo/config.h's limit");
  }
  p->values_left--;
  skip_ws(p);
  if (p->at >= p->end) return fail(p, "unexpected end");
  v = (rayo_json *)rayo_arena_alloc(p->arena, sizeof *v);
  if (!v) return no_memory(p);
  v->raw = p->at;
  c = *p->at;
  if (c == '{' || c == '[') {
    p->at++;
    v->kind = c == '{' ? RAYO_JSON_OBJECT : RAYO_JSON_ARRAY;
    s = parse_container(p, depth, c == '{', v);
  } else if (c == '"') {
    p->at++;
    v->kind = RAYO_JSON_STRING;
    s = parse_string(p, &v->string, &v->length);
  } else if (c == '-' || (c >= '0' && c <= '9')) {
    v->kind = RAYO_JSON_NUMBER;
    s = parse_number(p, &v->number);
  } else if (literal(p, "true")) {
    v->kind = RAYO_JSON_BOOL;
    v->boolean = RAYO_TRUE;
  } else if (literal(p, "false")) {
    v->kind = RAYO_JSON_BOOL;
  } else if (literal(p, "null")) {
    v->kind = RAYO_JSON_NULL;
  } else {
    s = fail(p, "unexpected character");
  }
  if (s != RAYO_OK) return s;
  v->raw_length = (size_t)(p->at - v->raw);
  *out = v;
  return RAYO_OK;
}

rayo_status rayo_json_parse(const char *text, size_t size, size_t max_values, rayo_arena *arena,
                            rayo_json **out, rayo_message *msg) {
  parser p;
  rayo_status s;
  p.at = text;
  p.end = text + size;
  p.arena = arena;
  p.msg = msg;
  p.values_left = max_values;
  s = parse_value(&p, 0, out);
  if (s != RAYO_OK) return s;
  skip_ws(&p);
  if (p.at != p.end) return fail(&p, "text after the value");
  return RAYO_OK;
}

const rayo_json *rayo_json_get(const rayo_json *obj, const char *key) {
  size_t i;
  if (!obj || obj->kind != RAYO_JSON_OBJECT) return NULL;
  for (i = 0; i < obj->length; i++) {
    if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
  }
  return NULL;
}

rayo_bool rayo_json_as_int(const rayo_json *v, rayo_i32 *out) {
  double d;
  if (!v || v->kind != RAYO_JSON_NUMBER) return RAYO_FALSE;
  d = v->number;
  if (d != floor(d) || d < -2147483647.0 || d > 2147483647.0) return RAYO_FALSE;
  *out = (rayo_i32)d;
  return RAYO_TRUE;
}

const char *rayo_json_as_string(const rayo_json *v) {
  return v && v->kind == RAYO_JSON_STRING ? v->string : NULL;
}

#include "core/small_end.h"
