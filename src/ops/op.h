/* An op: what a graph node computes. Each op family lives in its own file
 * and is listed in ops/registry.c. */
#ifndef RAYO_OPS_OP_H
#define RAYO_OPS_OP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "core/arena.h"
#include "core/json.h"
#include "core/tensor.h"
#include "kernels/kernels.h"
#include "rayo/error.h"

#define RAYO_MAX_INPUTS 4

typedef struct rayo_op rayo_op;

/* What a run gives an op besides its tensors. */
typedef struct rayo_exec {
  const rayo_kernels *kernels; /* chosen for this CPU when the session was made */
  void *scratch;               /* at least the bytes `infer` asked for, RAYO_ALIGN-aligned */
  const int64_t *symbols;      /* the size symbols' current values */
} rayo_exec;

/* Looks a size symbol up by name: its index, or -1. */
typedef int32_t rayo_symbol_fn(const void *graph, const char *name);

/* What an op is given about its node when the model loads. */
typedef struct rayo_op_spec {
  const rayo_json *json; /* the node's object, for its attributes */
  const char *output;    /* the node's output name, for messages */
  int input_count;
  const rayo_tensor *inputs; /* weights have `weight` set; activations only a dtype */
  rayo_symbol_fn *symbol;    /* resolves symbols named in attributes */
  const void *graph;
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
  rayo_status (*infer)(const void *params, const rayo_tensor *inputs, const int64_t *symbols,
                       rayo_tensor *output, size_t *scratch, rayo_message *msg);
  /* Computes `output` (its memory set) from `inputs`. Allocates nothing. */
  rayo_status (*run)(const void *params, const rayo_tensor *inputs, rayo_tensor *output,
                     const rayo_exec *exec);
  /* The output is a view of input 0's memory (a state), not planned memory. */
  bool aliases_input;
};

/* The op called `name`, or NULL. */
const rayo_op *rayo_op_find(const char *name);

#endif
