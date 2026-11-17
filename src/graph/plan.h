/* Shapes and the memory plan. A session plans once, for the largest sizes
 * the graph allows: every activation gets an offset in one arena, and two
 * activations share memory when their lifetimes do not overlap. */
#ifndef RAYO_GRAPH_PLAN_H
#define RAYO_GRAPH_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "core/arena.h"
#include "core/tensor.h"
#include "graph/graph.h"

typedef struct rayo_plan {
  size_t *input_offset; /* per graph input */
  size_t *node_offset;  /* per node output */
  size_t scratch_offset;
  size_t scratch_bytes; /* the largest any op needs */
  size_t arena_bytes;   /* in all */
} rayo_plan;

/* The tensors node `index` reads, from the graph inputs' and the nodes'
 * current tensors, into `out` (RAYO_MAX_INPUTS of them). */
void rayo_graph_node_inputs(const rayo_graph *graph, const rayo_pcwt *weights, int32_t index,
                            const rayo_tensor *inputs, const rayo_tensor *nodes, rayo_tensor *out);

/* Every input's and node's dtype and shape with the symbols at `symbols`,
 * and the most scratch one op needs. Allocates nothing. */
rayo_status rayo_graph_shapes(const rayo_graph *graph, const rayo_pcwt *weights,
                              const int64_t *symbols, rayo_tensor *inputs, rayo_tensor *nodes,
                              size_t *scratch, rayo_message *msg);

/* Plans the arena for the shapes in `inputs` and `nodes` (the largest),
 * with the offsets in `arena`. */
rayo_status rayo_plan_make(const rayo_graph *graph, const rayo_tensor *inputs,
                           const rayo_tensor *nodes, size_t scratch, rayo_arena *arena,
                           rayo_plan *out, rayo_message *msg);

#endif
