#include "core/names.h"

#include <string.h>

#include "core/compat.h"

#include "core/small_begin.h"

/* FNV-1a of the name. */
static size_t hash(const char *s) {
  rayo_u32 h = 2166136261u;
  while (*s) h = ((h ^ (rayo_u8)*s++) * 16777619u) & 0xFFFFFFFFu;
  return (size_t)h;
}

rayo_bool rayo_names_init(rayo_names *t, size_t count, rayo_arena *arena) {
  size_t slots = 16;
  while (slots / 2 < count) {
    if (slots > RAYO_SIZE_MAX / 4) return RAYO_FALSE;
    slots *= 2;
  }
  t->keys = (const char **)rayo_arena_array(arena, slots, sizeof *t->keys);
  t->values = (rayo_i32 *)rayo_arena_array(arena, slots, sizeof *t->values);
  t->mask = slots - 1;
  return t->keys && t->values;
}

static size_t find(const rayo_names *t, const char *name) {
  size_t i = hash(name) & t->mask;
  while (t->keys[i] && strcmp(t->keys[i], name) != 0) i = (i + 1) & t->mask;
  return i;
}

rayo_i32 *rayo_names_slot(rayo_names *t, const char *name) {
  const size_t i = find(t, name);
  if (!t->keys[i]) {
    t->keys[i] = name;
    t->values[i] = -1;
  }
  return &t->values[i];
}

rayo_i32 rayo_names_get(const rayo_names *t, const char *name) {
  const size_t i = find(t, name);
  return t->keys[i] ? t->values[i] : -1;
}

#include "core/small_end.h"
