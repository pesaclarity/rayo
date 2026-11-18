/* A strict JSON parser (RFC 8259) for model headers, which are untrusted.
 * The tree lives in an arena. */
#ifndef RAYO_CORE_JSON_H
#define RAYO_CORE_JSON_H

#include <stddef.h>

#include "core/arena.h"
#include "rayo/error.h"
#include "rayo/types.h"

typedef enum rayo_json_kind {
  RAYO_JSON_NULL,
  RAYO_JSON_BOOL,
  RAYO_JSON_NUMBER,
  RAYO_JSON_STRING,
  RAYO_JSON_ARRAY,
  RAYO_JSON_OBJECT
} rayo_json_kind;

typedef struct rayo_json rayo_json;

struct rayo_json {
  rayo_json_kind kind;
  rayo_bool boolean;
  double number;
  const char *string; /* NUL-terminated, unescaped; a STRING's value */
  size_t length;      /* of `string`, or the count of `items` */
  rayo_json **items;  /* ARRAY or OBJECT values */
  const char **keys;  /* OBJECT keys, parallel to `items` */
  const char *raw;    /* this value's text in the input */
  size_t raw_length;
};

/* Parses the `size` bytes at `text` (one JSON value, then only whitespace).
 * RAYO_ERR_FORMAT with a message for anything else, strings that are not
 * UTF-8 included; nesting deeper than RAYO_JSON_MAX_DEPTH is refused, and
 * more than `max_values` values (numbers, strings, lists, objects) is
 * RAYO_ERR_LIMIT, which bounds the arena the tree takes (about 100 bytes a
 * value). */
rayo_status rayo_json_parse(const char *text, size_t size, size_t max_values, rayo_arena *arena,
                            rayo_json **out, rayo_message *msg);

/* Member `key` of an OBJECT, or NULL (also when `obj` is NULL or not an
 * object). */
const rayo_json *rayo_json_get(const rayo_json *obj, const char *key);

/* A NUMBER that is a whole number from -(2^31 - 1) to 2^31 - 1. */
rayo_bool rayo_json_as_int(const rayo_json *v, rayo_i32 *out);

/* A STRING's text, or NULL. */
const char *rayo_json_as_string(const rayo_json *v);

#endif
