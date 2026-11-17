/* A strict JSON parser (RFC 8259) for model headers, which are untrusted.
 * The tree lives in an arena. */
#ifndef RAYO_CORE_JSON_H
#define RAYO_CORE_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "core/arena.h"
#include "rayo/error.h"

typedef enum rayo_json_kind {
  RAYO_JSON_NULL,
  RAYO_JSON_BOOL,
  RAYO_JSON_NUMBER,
  RAYO_JSON_STRING,
  RAYO_JSON_ARRAY,
  RAYO_JSON_OBJECT,
} rayo_json_kind;

typedef struct rayo_json rayo_json;

struct rayo_json {
  rayo_json_kind kind;
  bool boolean;
  double number;
  const char *string; /* NUL-terminated, unescaped; a STRING's value */
  size_t length;      /* of `string`, or the count of `items` */
  rayo_json **items;  /* ARRAY or OBJECT values */
  const char **keys;  /* OBJECT keys, parallel to `items` */
  const char *raw;    /* this value's text in the input */
  size_t raw_length;
};

/* Parses the `size` bytes at `text` (one JSON value, then only whitespace).
 * RAYO_ERR_FORMAT with a message for anything else; nesting deeper than 32
 * is refused. */
rayo_status rayo_json_parse(const char *text, size_t size, rayo_arena *arena, rayo_json **out,
                            rayo_message *msg);

/* Member `key` of an OBJECT, or NULL (also when `obj` is NULL or not an
 * object). */
const rayo_json *rayo_json_get(const rayo_json *obj, const char *key);

/* A NUMBER that is a whole number in the int64 range. */
bool rayo_json_as_int(const rayo_json *v, int64_t *out);

/* A STRING's text, or NULL. */
const char *rayo_json_as_string(const rayo_json *v);

#endif
