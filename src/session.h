/* struct rayo_session: a model's planned arena and current shapes. */
#ifndef RAYO_SESSION_INTERNAL_H
#define RAYO_SESSION_INTERNAL_H

#include "core/arena.h"
#include "core/tensor.h"
#include "graph/plan.h"
#include "kernels/kernels.h"
#include "model.h"
#include "platform/platform.h"
#include "rayo/session.h"

struct rayo_session {
  const rayo_model *model;
  rayo_kernels kernels;  /* chosen for this CPU at creation */
  rayo_pool *pool;       /* owned; NULL for one thread */
  unsigned char *memory; /* owned: the planned arena, one allocation */
  rayo_arena book;       /* owns the plan and the tensor tables */
  rayo_plan plan;
  rayo_i32 *symbols;     /* the size symbols' current values */
  rayo_i32 *symbol_max;  /* the largest each may be: the graph's, or a limit's */
  rayo_i32 *state_rows;  /* per state: rows at most (0: as declared) */
  rayo_i32 *state_group; /* per state: 0 float32, else int8 with groups of this */
  size_t memory_bytes;   /* of `memory` */
  rayo_values values;    /* current shapes; data in `memory` */
  size_t *planned;       /* each node's bytes as planned: no bind may exceed them */
};

#endif
