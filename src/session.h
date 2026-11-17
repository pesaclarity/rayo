/* struct rayo_session: a model's planned arena and current shapes. */
#ifndef RAYO_SESSION_INTERNAL_H
#define RAYO_SESSION_INTERNAL_H

#include "core/arena.h"
#include "core/tensor.h"
#include "graph/plan.h"
#include "kernels/kernels.h"
#include "model.h"
#include "rayo/session.h"

struct rayo_session {
  const rayo_model *model;
  rayo_kernels kernels;  /* chosen for this CPU at creation */
  unsigned char *memory; /* owned: the planned arena, one allocation */
  rayo_arena book;       /* owns the plan and the tensor tables */
  rayo_plan plan;
  rayo_i32 *symbols;  /* the size symbols' current values */
  rayo_values values; /* current shapes; data in `memory` */
  size_t *planned;    /* each node's bytes as planned: no bind may exceed them */
};

#endif
