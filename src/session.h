/* struct rayo_session: a model's planned arena and current shapes. */
#ifndef RAYO_SESSION_INTERNAL_H
#define RAYO_SESSION_INTERNAL_H

#include <stdint.h>

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
  int64_t symbols[RAYO_MAX_SYMBOLS];
  rayo_tensor *inputs; /* current shapes; data in `memory` */
  rayo_tensor *nodes;
};

#endif
