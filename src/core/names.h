/* A table of names to indexes, for reading a model's tensors and graph in
 * time linear in their count: an untrusted file may hold many. Open
 * addressing in an arena; the names are the caller's. */
#ifndef RAYO_CORE_NAMES_H
#define RAYO_CORE_NAMES_H

#include <stddef.h>

#include "core/arena.h"
#include "rayo/types.h"

typedef struct rayo_names {
  const char **keys; /* NULL: an empty slot */
  rayo_i32 *values;
  size_t mask; /* slots - 1 */
} rayo_names;

/* A table for up to `count` names. RAYO_FALSE when out of memory. */
rayo_bool rayo_names_init(rayo_names *t, size_t count, rayo_arena *arena);

/* The slot of `name`: its value, or -1 (and the name added) if it had none.
 * The caller sets the value. At most `count` names may be added. */
rayo_i32 *rayo_names_slot(rayo_names *t, const char *name);

/* The value of `name`, or -1. */
rayo_i32 rayo_names_get(const rayo_names *t, const char *name);

#endif
