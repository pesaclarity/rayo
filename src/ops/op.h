/* An op: what a graph node computes. Each op family lives in its own file
 * and is listed in ops/registry.c. */
#ifndef RAYO_OPS_OP_H
#define RAYO_OPS_OP_H

#include <stddef.h>

#include "core/arena.h"
#include "core/compat.h"
#include "core/json.h"
#include "core/tensor.h"
#include "kernels/kernels.h"
#include "rayo/error.h"
#include "rayo/op.h"

typedef struct rayo_op rayo_op;

/* What a run gives an op besides its tensors. */
typedef struct rayo_exec {
  const rayo_kernels *kernels; /* chosen for this CPU when the session was made */
  void *scratch;               /* at least the bytes `infer` asked for, RAYO_ALIGN-aligned */
  const rayo_i32 *symbols;     /* the size symbols' current values */
} rayo_exec;

/* Looks a size symbol up by name: its index, or -1. */
typedef rayo_i32 rayo_symbol_fn(const void *graph, const char *name);

/* What an op is given about its node when the model loads. */
typedef struct rayo_op_spec {
  const rayo_json *json; /* the node's object, for its attributes */
  const char *output;    /* the node's output name, for messages */
  int input_count;
  const rayo_tensor *inputs; /* weights have `weight` set; activations only a dtype */
  rayo_symbol_fn *symbol;    /* resolves symbols named in attributes */
  const void *graph;
  const rayo_custom_op *custom; /* for rayo_op_custom: the program's op */
} rayo_op_spec;

struct rayo_op {
  const char *name;
  int min_inputs;
  int max_inputs;
  /* At load: reads the attributes, checks the weights and packs them into
   * `arena` for the kernels, and sets `*params` (owned by `arena`). */
  rayo_status (*prepare)(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                         rayo_message *msg);
  /* The output's dtype and shape for these inputs, and the scratch bytes the
   * run needs. Called when a session is planned and at every bind. */
  rayo_status (*infer)(const void *params, const rayo_tensor *inputs, const rayo_i32 *symbols,
                       rayo_tensor *output, size_t *scratch, rayo_message *msg);
  /* Computes `output` (its memory set) from `inputs`. Allocates nothing. */
  rayo_status (*run)(const void *params, const rayo_tensor *inputs, rayo_tensor *output,
                     const rayo_exec *exec);
  /* The output is a view of input 0's memory (a state), not planned memory. */
  rayo_bool aliases_input;
};

/* The built-in op called `name`, or NULL. */
const rayo_op *rayo_op_find(const char *name);

/* RAYO_OK when `op`, a program's own, can be used. */
rayo_status rayo_custom_op_check(const rayo_custom_op *op, rayo_message *msg);

#endif
