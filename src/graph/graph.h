/* A model's graph: its inputs, nodes and outputs, read from the header's
 * "graph" object (docs/FORMAT.md) and checked against the weights. */
#ifndef RAYO_GRAPH_GRAPH_H
#define RAYO_GRAPH_GRAPH_H

#include <stdint.h>

#include "core/arena.h"
#include "core/json.h"
#include "format/pcwt.h"
#include "ops/op.h"

#define RAYO_GRAPH_VERSION 1
#define RAYO_MAX_SYMBOLS 8

typedef enum rayo_ref_kind {
  RAYO_REF_INPUT,
  RAYO_REF_STATE,
  RAYO_REF_WEIGHT,
  RAYO_REF_NODE,
} rayo_ref_kind;

/* What a node reads: graph input, state, weight or earlier node `index`. */
typedef struct rayo_ref {
  rayo_ref_kind kind;
  int32_t index;
} rayo_ref;

/* A dimension: `value`, or size symbol `symbol` when that is >= 0. */
typedef struct rayo_dim {
  int64_t value;
  int32_t symbol;
} rayo_dim;

/* A graph input, or a state (a tensor kept across runs, such as a key/value
 * cache: float32, whole-number dimensions, zeroed by rayo_session_reset). */
typedef struct rayo_graph_input {
  const char *name;
  rayo_dtype dtype;
  int32_t rank;
  rayo_dim dims[RAYO_MAX_RANK];
} rayo_graph_input;

typedef struct rayo_node {
  const rayo_op *op;
  const char *output;
  int32_t input_count;
  rayo_ref inputs[RAYO_MAX_INPUTS];
  const void *params; /* the op's, from its prepare */
} rayo_node;

typedef struct rayo_graph_output {
  const char *name;
  int32_t node;
} rayo_graph_output;

typedef struct rayo_graph {
  int32_t symbol_count;
  const char *symbols[RAYO_MAX_SYMBOLS];
  int64_t symbol_max[RAYO_MAX_SYMBOLS];
  /* The values a session plans its memory for: each symbol's largest, unless
   * the graph's "plan" says otherwise (when symbols trade off, as a cache's
   * past P and new T with P + T at most its length). */
  int64_t symbol_plan[RAYO_MAX_SYMBOLS];
  int32_t input_count;
  rayo_graph_input *inputs;
  int32_t state_count;
  rayo_graph_input *states;
  int32_t node_count;
  rayo_node *nodes;
  int32_t output_count;
  rayo_graph_output *outputs;
} rayo_graph;

/* Reads `json` (the header's "graph") into `out`, preparing every node's op
 * with its weights from `weights`. Everything lives in `arena`. */
rayo_status rayo_graph_read(const rayo_json *json, const rayo_pcwt *weights, rayo_arena *arena,
                            rayo_graph *out, rayo_message *msg);

/* The index of symbol `name`, or -1. */
int32_t rayo_graph_symbol(const rayo_graph *graph, const char *name);

#endif
