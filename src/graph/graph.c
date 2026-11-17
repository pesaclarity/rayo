#include "graph/graph.h"

#include <stdlib.h>
#include <string.h>

#include "core/message.h"
#include "ops/ops.h"
#include "rayo/config.h"

rayo_i32 rayo_graph_symbol(const rayo_graph *graph, const char *name) {
  rayo_i32 i;
  for (i = 0; i < graph->symbol_count; i++) {
    if (strcmp(graph->symbols[i], name) == 0) return i;
  }
  return -1;
}

static rayo_status read_plan(const rayo_json *json, rayo_graph *g, rayo_message *msg) {
  const rayo_json *plan = rayo_json_get(json, "plan");
  size_t i;
  if (!plan) return RAYO_OK;
  if (plan->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph plan: an object of symbol values");
  }
  for (i = 0; i < plan->length; i++) {
    rayo_i32 k = rayo_graph_symbol(g, plan->keys[i]);
    rayo_i32 v;
    if (k < 0 || !rayo_json_as_int(plan->items[i], &v) || v < 0 || v > g->symbol_max[k]) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "graph plan: %s", plan->keys[i]);
    }
    g->symbol_plan[k] = v;
  }
  return RAYO_OK;
}

static rayo_status read_symbols(const rayo_json *json, rayo_graph *g, rayo_arena *arena,
                                rayo_message *msg) {
  const rayo_json *s = rayo_json_get(json, "symbols");
  const size_t n = s && s->kind == RAYO_JSON_OBJECT ? s->length : 0;
  size_t i;
  if (s && s->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph symbols: an object of sizes");
  }
  g->symbols = rayo_arena_array(arena, n ? n : 1, sizeof *g->symbols);
  g->symbol_max = rayo_arena_array(arena, n ? n : 1, sizeof *g->symbol_max);
  g->symbol_plan = rayo_arena_array(arena, n ? n : 1, sizeof *g->symbol_plan);
  if (!g->symbols || !g->symbol_max || !g->symbol_plan) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  if (!s) return RAYO_OK;
  for (i = 0; i < s->length; i++) {
    rayo_i32 max;
    if (!rayo_json_as_int(s->items[i], &max) || max < 1 || max > RAYO_MAX_DIM) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "symbol %s: its largest size", s->keys[i]);
    }
    g->symbols[i] = s->keys[i];
    g->symbol_max[i] = max;
    g->symbol_plan[i] = max;
  }
  g->symbol_count = (rayo_i32)s->length;
  return read_plan(json, g, msg);
}

static rayo_status read_dtype(const char *s, rayo_dtype *out) {
  if (s && strcmp(s, "float32") == 0) {
    *out = RAYO_DTYPE_F32;
  } else if (s && strcmp(s, "int32") == 0) {
    *out = RAYO_DTYPE_I32;
  } else {
    return RAYO_ERR_UNSUPPORTED;
  }
  return RAYO_OK;
}

static rayo_status read_dim(const rayo_json *j, const rayo_graph *g, rayo_bool state,
                            const char *name, rayo_dim *d, rayo_message *msg) {
  const char *sym = rayo_json_as_string(j);
  if (sym && state) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "state %s: sizes must be whole numbers", name);
  }
  if (sym) {
    d->symbol = rayo_graph_symbol(g, sym);
    if (d->symbol < 0) return rayo_fail(msg, RAYO_ERR_FORMAT, "input %s: no symbol %s", name, sym);
    return RAYO_OK;
  }
  if (!rayo_json_as_int(j, &d->value) || d->value < 1 || d->value > RAYO_MAX_DIM) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "input %s: bad dimension", name);
  }
  d->symbol = -1;
  return RAYO_OK;
}

static rayo_status read_input(const rayo_json *j, rayo_graph *g, rayo_graph_input *in,
                              rayo_bool state, rayo_message *msg) {
  const rayo_json *shape = rayo_json_get(j, "shape");
  size_t i;
  in->name = rayo_json_as_string(rayo_json_get(j, "name"));
  if (!in->name ||
      read_dtype(rayo_json_as_string(rayo_json_get(j, "dtype")), &in->dtype) != RAYO_OK || !shape ||
      shape->kind != RAYO_JSON_ARRAY || shape->length > RAYO_MAX_RANK) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "a graph input or state needs a name, a dtype and a shape");
  }
  if (state && in->dtype != RAYO_DTYPE_F32) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "state %s: only float32", in->name);
  }
  in->rank = (rayo_i32)shape->length;
  for (i = 0; i < shape->length; i++) {
    rayo_status s = read_dim(shape->items[i], g, state, in->name, &in->dims[i], msg);
    if (s != RAYO_OK) return s;
  }
  return RAYO_OK;
}

static rayo_ref ref_of(rayo_ref_kind kind, rayo_i32 index) {
  rayo_ref r;
  r.kind = kind;
  r.index = index;
  return r;
}

/* A name in scope before node `upto`: an earlier node, an input, a state,
 * else a weight. Later definitions shadow earlier ones. */
static rayo_bool resolve(const rayo_graph *g, rayo_i32 upto, const rayo_pcwt *w, const char *name,
                         rayo_ref *ref) {
  rayo_i32 i;
  size_t k;
  for (i = upto - 1; i >= 0; i--) {
    if (strcmp(g->nodes[i].output, name) == 0) {
      *ref = ref_of(RAYO_REF_NODE, i);
      return RAYO_TRUE;
    }
  }
  for (i = 0; i < g->input_count; i++) {
    if (strcmp(g->inputs[i].name, name) == 0) {
      *ref = ref_of(RAYO_REF_INPUT, i);
      return RAYO_TRUE;
    }
  }
  for (i = 0; i < g->state_count; i++) {
    if (strcmp(g->states[i].name, name) == 0) {
      *ref = ref_of(RAYO_REF_STATE, i);
      return RAYO_TRUE;
    }
  }
  for (k = 0; w && k < w->weight_count; k++) {
    if (strcmp(w->weights[k].name, name) == 0) {
      *ref = ref_of(RAYO_REF_WEIGHT, (rayo_i32)k);
      return RAYO_TRUE;
    }
  }
  return RAYO_FALSE;
}

static rayo_i32 symbol_of(const void *graph, const char *name) {
  return rayo_graph_symbol((const rayo_graph *)graph, name);
}

/* What prepare sees of input `ref`: a weight in full; a state's dtype and
 * shape; an activation's nothing yet. */
static rayo_tensor seen_input(const rayo_graph *g, const rayo_pcwt *w, rayo_ref ref) {
  rayo_tensor t;
  rayo_i32 d;
  memset(&t, 0, sizeof t);
  if (ref.kind == RAYO_REF_WEIGHT) return rayo_tensor_of_weight(&w->weights[ref.index]);
  if (ref.kind == RAYO_REF_STATE) {
    const rayo_graph_input *st = &g->states[ref.index];
    t.dtype = st->dtype;
    t.shape.rank = st->rank;
    for (d = 0; d < st->rank; d++) t.shape.dims[d] = st->dims[d].value;
  }
  return t;
}

/* A program's op called `name`, or NULL. */
static const rayo_custom_op *custom_op(const rayo_model_options *o, const char *name) {
  int i;
  for (i = 0; o && i < o->op_count; i++) {
    if (strcmp(o->ops[i]->name, name) == 0) return o->ops[i];
  }
  return NULL;
}

static rayo_status read_node(const rayo_json *j, rayo_graph *g, rayo_i32 index, const rayo_pcwt *w,
                             const rayo_model_options *options, rayo_arena *arena,
                             rayo_message *msg) {
  rayo_node *n = &g->nodes[index];
  const char *op = rayo_json_as_string(rayo_json_get(j, "op"));
  const rayo_json *inputs = rayo_json_get(j, "inputs");
  rayo_tensor *seen;
  rayo_op_spec spec;
  rayo_status s;
  rayo_i32 i;
  int lo, hi;
  n->output = rayo_json_as_string(rayo_json_get(j, "output"));
  if (!op || !n->output || !inputs || inputs->kind != RAYO_JSON_ARRAY) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph node %ld: needs an op, inputs and an output",
                     (long)index);
  }
  n->op = rayo_op_find(op);
  spec.custom = n->op ? NULL : custom_op(options, op);
  if (spec.custom) n->op = &rayo_op_custom;
  if (!n->op) return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: unknown op %s", n->output, op);
  lo = spec.custom ? spec.custom->min_inputs : n->op->min_inputs;
  hi = spec.custom ? spec.custom->max_inputs : n->op->max_inputs;
  if ((int)inputs->length < lo || (int)inputs->length > hi) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s takes %d to %d inputs", n->output, op, lo,
                     hi);
  }
  n->input_count = (rayo_i32)inputs->length;
  if (n->input_count > g->max_node_inputs) g->max_node_inputs = n->input_count;
  n->inputs = rayo_arena_array(arena, inputs->length ? inputs->length : 1, sizeof *n->inputs);
  /* What prepare sees, for this call only. */
  seen = calloc(inputs->length ? inputs->length : 1, sizeof *seen);
  if (!n->inputs || !seen) {
    free(seen);
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (i = 0; i < n->input_count; i++) {
    const char *name = rayo_json_as_string(inputs->items[i]);
    if (!name || !resolve(g, index, w, name, &n->inputs[i])) {
      free(seen);
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: no input, node or weight %s", n->output,
                       name ? name : "(not a string)");
    }
    seen[i] = seen_input(g, w, n->inputs[i]);
  }
  spec.json = j;
  spec.output = n->output;
  spec.input_count = n->input_count;
  spec.inputs = seen;
  spec.symbol = symbol_of;
  spec.graph = g;
  s = n->op->prepare(&spec, arena, &n->params, msg);
  free(seen);
  return s;
}

static rayo_status read_outputs(const rayo_json *list, rayo_graph *g, rayo_arena *arena,
                                rayo_message *msg) {
  size_t i;
  if (!list || list->kind != RAYO_JSON_ARRAY || list->length == 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "the graph needs outputs");
  }
  g->outputs = rayo_arena_array(arena, list->length, sizeof *g->outputs);
  if (!g->outputs) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  g->output_count = (rayo_i32)list->length;
  for (i = 0; i < list->length; i++) {
    const char *name = rayo_json_as_string(rayo_json_get(list->items[i], "name"));
    const char *node = rayo_json_as_string(rayo_json_get(list->items[i], "node"));
    rayo_ref ref;
    if (!name || !node || !resolve(g, g->node_count, NULL, node, &ref) ||
        ref.kind != RAYO_REF_NODE) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "graph output %lu: needs a name and a node",
                       (unsigned long)i);
    }
    g->outputs[i].name = name;
    g->outputs[i].node = ref.index;
  }
  return RAYO_OK;
}

static rayo_status read_decls(const rayo_json *list, rayo_graph *g, rayo_arena *arena,
                              rayo_bool state, rayo_graph_input **out, rayo_i32 *count,
                              rayo_message *msg) {
  size_t i;
  *out = rayo_arena_array(arena, list->length ? list->length : 1, sizeof **out);
  if (!*out) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  *count = (rayo_i32)list->length;
  for (i = 0; i < list->length; i++) {
    rayo_status s = read_input(list->items[i], g, &(*out)[i], state, msg);
    if (s != RAYO_OK) return s;
  }
  return RAYO_OK;
}

rayo_status rayo_graph_read(const rayo_json *json, const rayo_pcwt *weights,
                            const rayo_model_options *options, rayo_arena *arena, rayo_graph *out,
                            rayo_message *msg) {
  rayo_i32 version;
  const rayo_json *inputs, *states, *nodes;
  rayo_status s;
  size_t i;
  memset(out, 0, sizeof *out);
  for (i = 0; options && i < (size_t)(options->op_count > 0 ? options->op_count : 0); i++) {
    s = rayo_custom_op_check(options->ops[i], msg);
    if (s != RAYO_OK) return s;
  }
  if (!json || json->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "no graph in the header: not a rayo model");
  }
  if (!rayo_json_as_int(rayo_json_get(json, "version"), &version) ||
      version != RAYO_GRAPH_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "graph version: rayo reads %d", RAYO_GRAPH_VERSION);
  }
  s = read_symbols(json, out, arena, msg);
  if (s != RAYO_OK) return s;
  inputs = rayo_json_get(json, "inputs");
  states = rayo_json_get(json, "states");
  nodes = rayo_json_get(json, "nodes");
  if (!inputs || inputs->kind != RAYO_JSON_ARRAY || inputs->length == 0 || !nodes ||
      nodes->kind != RAYO_JSON_ARRAY || nodes->length == 0) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "the graph needs inputs and nodes");
  }
  if (states && states->kind != RAYO_JSON_ARRAY) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph states: a list");
  }
  s = read_decls(inputs, out, arena, RAYO_FALSE, &out->inputs, &out->input_count, msg);
  if (s == RAYO_OK && states) {
    s = read_decls(states, out, arena, RAYO_TRUE, &out->states, &out->state_count, msg);
  }
  if (s != RAYO_OK) return s;
  out->nodes = rayo_arena_array(arena, nodes->length, sizeof *out->nodes);
  if (!out->nodes) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (i = 0; i < nodes->length; i++) {
    out->node_count = (rayo_i32)i;
    s = read_node(nodes->items[i], out, (rayo_i32)i, weights, options, arena, msg);
    if (s != RAYO_OK) return s;
  }
  out->node_count = (rayo_i32)nodes->length;
  return read_outputs(rayo_json_get(json, "outputs"), out, arena, msg);
}
