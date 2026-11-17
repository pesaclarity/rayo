/* Shapes and the memory plan. A session plans once, for the largest sizes
 * the graph allows: states first (kept across runs), then the inputs, then
 * every activation at an offset in one arena, two activations sharing
 * memory when their lifetimes do not overlap, then one scratch region. */
#ifndef RAYO_GRAPH_PLAN_H
#define RAYO_GRAPH_PLAN_H

#include <stddef.h>

#include "core/arena.h"
#include "core/tensor.h"
#include "graph/graph.h"

/* The tensors of one session: its inputs, states and node outputs. */
typedef struct rayo_values {
  rayo_tensor *inputs;
  rayo_tensor *states;
  rayo_tensor *nodes;
} rayo_values;

typedef struct rayo_plan {
  size_t *state_offset; /* per state */
  size_t *input_offset; /* per graph input */
  size_t *node_offset;  /* per node output; unused for nodes that view a state */
  size_t scratch_offset;
  size_t scratch_bytes; /* the largest any op needs */
  size_t arena_bytes;   /* in all */
} rayo_plan;

/* The tensors node `index` reads, into `out` (RAYO_MAX_INPUTS of them). */
void rayo_graph_node_inputs(const rayo_graph *graph, const rayo_pcwt *weights, rayo_i32 index,
                            const rayo_values *values, rayo_tensor *out);

/* Every input's, state's and node's dtype and shape with the symbols at
 * `symbols`, and the most scratch one op needs. A node that views a state
 * gets the state's memory. Allocates nothing. */
rayo_status rayo_graph_shapes(const rayo_graph *graph, const rayo_pcwt *weights,
                              const rayo_i32 *symbols, rayo_values *values, size_t *scratch,
                              rayo_message *msg);

/* Plans the arena for the shapes in `values` (the largest), with the
 * offsets in `arena`. */
rayo_status rayo_plan_make(const rayo_graph *graph, const rayo_values *values, size_t scratch,
                           rayo_arena *arena, rayo_plan *out, rayo_message *msg);

#endif
