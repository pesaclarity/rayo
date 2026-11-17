#include "core/json.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/message.h"

#define RAYO_JSON_DEPTH 32

typedef struct parser {
  const char *at;
  const char *end;
  rayo_arena *arena;
  rayo_message *msg;
} parser;

static rayo_status fail(parser *p, const char *what) {
  return rayo_fail(p->msg, RAYO_ERR_FORMAT, "bad JSON in the header: %s", what);
}

static void skip_ws(parser *p) {
  while (p->at < p->end && (*p->at == ' ' || *p->at == '\t' || *p->at == '\n' || *p->at == '\r')) {
    p->at++;
  }
}

static rayo_status parse_value(parser *p, int depth, rayo_json **out);

static int hex(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static size_t put_utf8(char *to, uint32_t cp) {
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

static bool read_hex4(parser *p, uint32_t *out) {
  if (p->end - p->at < 4) return false;
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) {
    int h = hex(p->at[i]);
    if (h < 0) return false;
    v = v * 16 + (uint32_t)h;
  }
  p->at += 4;
  *out = v;
  return true;
}

/* After the opening quote. The unescaped text is never longer than the
 * escaped one, so one allocation of the escaped length suffices. */
static rayo_status parse_string(parser *p, const char **out, size_t *len) {
  const char *start = p->at;
  const char *q = start;
  while (q < p->end && *q != '"') q += (*q == '\\' && q + 1 < p->end) ? 2 : 1;
  if (q >= p->end) return fail(p, "unterminated string");
  char *s = rayo_arena_alloc(p->arena, (size_t)(q - start) + 1);
  if (!s) return rayo_fail(p->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  size_t n = 0;
  while (p->at < q) {
    unsigned char c = (unsigned char)*p->at++;
    if (c < 0x20) return fail(p, "control character in a string");
    if (c != '\\') {
      s[n++] = (char)c;
      continue;
    }
    char e = *p->at++;
    switch (e) {
      case '"':
        s[n++] = '"';
        break;
      case '\\':
        s[n++] = '\\';
        break;
      case '/':
        s[n++] = '/';
        break;
      case 'b':
        s[n++] = '\b';
        break;
      case 'f':
        s[n++] = '\f';
        break;
      case 'n':
        s[n++] = '\n';
        break;
      case 'r':
        s[n++] = '\r';
        break;
      case 't':
        s[n++] = '\t';
        break;
      case 'u': {
        uint32_t cp;
        if (!read_hex4(p, &cp)) return fail(p, "bad \\u escape");
        if (cp >= 0xD800 && cp <= 0xDBFF) {
          uint32_t lo;
          if (q - p->at < 6 || p->at[0] != '\\' || p->at[1] != 'u')
            return fail(p, "lone surrogate");
          p->at += 2;
          if (!read_hex4(p, &lo) || lo < 0xDC00 || lo > 0xDFFF) return fail(p, "lone surrogate");
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          return fail(p, "lone surrogate");
        }
        if (cp == 0) return fail(p, "NUL in a string");
        n += put_utf8(s + n, cp);
        break;
      }
      default:
        return fail(p, "bad escape");
    }
  }
  p->at = q + 1;
  s[n] = '\0';
  *out = s;
  *len = n;
  return RAYO_OK;
}

static rayo_status parse_number(parser *p, double *out) {
  const char *s = p->at;
  if (s < p->end && *s == '-') s++;
  if (s >= p->end) return fail(p, "bad number");
  if (*s == '0') {
    s++;
  } else if (*s >= '1' && *s <= '9') {
    while (s < p->end && *s >= '0' && *s <= '9') s++;
  } else {
    return fail(p, "bad number");
  }
  if (s < p->end && *s == '.') {
    s++;
    if (s >= p->end || *s < '0' || *s > '9') return fail(p, "bad number");
    while (s < p->end && *s >= '0' && *s <= '9') s++;
  }
  if (s < p->end && (*s == 'e' || *s == 'E')) {
    s++;
    if (s < p->end && (*s == '+' || *s == '-')) s++;
    if (s >= p->end || *s < '0' || *s > '9') return fail(p, "bad number");
    while (s < p->end && *s >= '0' && *s <= '9') s++;
  }
  char buf[64];
  size_t n = (size_t)(s - p->at);
  if (n >= sizeof buf) return fail(p, "number too long");
  memcpy(buf, p->at, n);
  buf[n] = '\0';
  *out = strtod(buf, NULL);
  if (!isfinite(*out)) return fail(p, "number out of range");
  p->at = s;
  return RAYO_OK;
}

static bool literal(parser *p, const char *word) {
  size_t n = strlen(word);
  if ((size_t)(p->end - p->at) < n || memcmp(p->at, word, n) != 0) return false;
  p->at += n;
  return true;
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
    rayo_json **values = rayo_arena_array(p->arena, cap, sizeof *values);
    const char **keys = rayo_arena_array(p->arena, cap, sizeof *keys);
    if (!values || !keys) return rayo_fail(p->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
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

static rayo_status parse_container(parser *p, int depth, bool object, rayo_json *v) {
  items it = {0};
  char close = object ? '}' : ']';
  skip_ws(p);
  if (p->at < p->end && *p->at == close) {
    p->at++;
  } else {
    for (;;) {
      const char *key = NULL;
      if (object) {
        size_t key_len;
        skip_ws(p);
        if (p->at >= p->end || *p->at != '"') return fail(p, "expected a key");
        p->at++;
        rayo_status s = parse_string(p, &key, &key_len);
        if (s != RAYO_OK) return s;
        skip_ws(p);
        if (p->at >= p->end || *p->at != ':') return fail(p, "expected ':'");
        p->at++;
      }
      rayo_json *item;
      rayo_status s = parse_value(p, depth + 1, &item);
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
  if (depth > RAYO_JSON_DEPTH) return fail(p, "nested too deep");
  skip_ws(p);
  if (p->at >= p->end) return fail(p, "unexpected end");
  rayo_json *v = rayo_arena_alloc(p->arena, sizeof *v);
  if (!v) return rayo_fail(p->msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  v->raw = p->at;
  rayo_status s = RAYO_OK;
  char c = *p->at;
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
    v->boolean = true;
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

rayo_status rayo_json_parse(const char *text, size_t size, rayo_arena *arena, rayo_json **out,
                            rayo_message *msg) {
  parser p = {text, text + size, arena, msg};
  rayo_status s = parse_value(&p, 0, out);
  if (s != RAYO_OK) return s;
  skip_ws(&p);
  if (p.at != p.end) return fail(&p, "text after the value");
  return RAYO_OK;
}

const rayo_json *rayo_json_get(const rayo_json *obj, const char *key) {
  if (!obj || obj->kind != RAYO_JSON_OBJECT) return NULL;
  for (size_t i = 0; i < obj->length; i++) {
    if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
  }
  return NULL;
}

bool rayo_json_as_int(const rayo_json *v, int64_t *out) {
  if (!v || v->kind != RAYO_JSON_NUMBER) return false;
  double d = v->number;
  if (d != floor(d) || d < -9.0e15 || d > 9.0e15) return false;
  *out = (int64_t)d;
  return true;
}

const char *rayo_json_as_string(const rayo_json *v) {
  return v && v->kind == RAYO_JSON_STRING ? v->string : NULL;
}
