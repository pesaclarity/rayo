/* A model's graph: its inputs, nodes and outputs, read from the header's
 * "graph" object (docs/FORMAT.md) and checked against the weights. */
#ifndef RAYO_GRAPH_GRAPH_H
#define RAYO_GRAPH_GRAPH_H

#include "core/arena.h"
#include "core/json.h"
#include "format/pcwt.h"
#include "ops/op.h"
#include "rayo/model.h"

#define RAYO_GRAPH_VERSION 1

typedef enum rayo_ref_kind {
  RAYO_REF_INPUT,
  RAYO_REF_STATE,
  RAYO_REF_WEIGHT,
  RAYO_REF_NODE
} rayo_ref_kind;

/* What a node reads: graph input, state, weight or earlier node `index`. */
typedef struct rayo_ref {
  rayo_ref_kind kind;
  rayo_i32 index;
} rayo_ref;

/* A dimension: `value`, or size symbol `symbol` when that is >= 0. */
typedef struct rayo_dim {
  rayo_i32 value;
  rayo_i32 symbol;
} rayo_dim;

/* A graph input, or a state (a tensor kept across runs, such as a key/value
 * cache: float32, whole-number dimensions, zeroed by rayo_session_reset). */
typedef struct rayo_graph_input {
  const char *name;
  rayo_dtype dtype;
  rayo_i32 rank;
  rayo_dim dims[RAYO_MAX_RANK];
} rayo_graph_input;

typedef struct rayo_node {
  const rayo_op *op;
  const char *output;
  rayo_i32 input_count;
  rayo_ref *inputs;   /* input_count of them, in the graph's arena */
  const void *params; /* the op's, from its prepare */
} rayo_node;

typedef struct rayo_graph_output {
  const char *name;
  rayo_i32 node;
} rayo_graph_output;

typedef struct rayo_graph {
  rayo_i32 symbol_count;
  const char **symbols;
  rayo_i32 *symbol_max;
  /* The values a session plans its memory for: each symbol's largest, unless
   * the graph's "plan" says otherwise (when symbols trade off, as a cache's
   * past P and new T with P + T at most its length). */
  rayo_i32 *symbol_plan;
  rayo_i32 input_count;
  rayo_graph_input *inputs;
  rayo_i32 state_count;
  rayo_graph_input *states;
  rayo_i32 node_count;
  rayo_node *nodes;
  rayo_i32 max_node_inputs; /* the most inputs any node reads */
  rayo_i32 output_count;
  rayo_graph_output *outputs;
} rayo_graph;

/* Reads `json` (the header's "graph") into `out`, preparing every node's op
 * with its weights from `weights`. Everything lives in `arena`. */
rayo_status rayo_graph_read(const rayo_json *json, const rayo_pcwt *weights,
                            const rayo_model_options *options, rayo_arena *arena, rayo_graph *out,
                            rayo_message *msg);

/* The index of symbol `name`, or -1. */
rayo_i32 rayo_graph_symbol(const rayo_graph *graph, const char *name);

#endif
