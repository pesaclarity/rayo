#include "graph/graph.h"

#include <string.h>

#include "core/message.h"

#define RAYO_MAX_NODES 4096

int32_t rayo_graph_symbol(const rayo_graph *graph, const char *name) {
  for (int32_t i = 0; i < graph->symbol_count; i++) {
    if (strcmp(graph->symbols[i], name) == 0) return i;
  }
  return -1;
}

static rayo_status read_symbols(const rayo_json *json, rayo_graph *g, rayo_message *msg) {
  const rayo_json *s = rayo_json_get(json, "symbols");
  if (!s) return RAYO_OK;
  if (s->kind != RAYO_JSON_OBJECT || s->length > RAYO_MAX_SYMBOLS) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph symbols: an object of at most %d sizes",
                     RAYO_MAX_SYMBOLS);
  }
  for (size_t i = 0; i < s->length; i++) {
    int64_t max;
    if (!rayo_json_as_int(s->items[i], &max) || max < 1 || max > (1 << 24)) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "symbol %s: its largest size", s->keys[i]);
    }
    g->symbols[i] = s->keys[i];
    g->symbol_max[i] = max;
    g->symbol_plan[i] = max;
  }
  g->symbol_count = (int32_t)s->length;
  const rayo_json *plan = rayo_json_get(json, "plan");
  if (!plan) return RAYO_OK;
  if (plan->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph plan: an object of symbol values");
  }
  for (size_t i = 0; i < plan->length; i++) {
    int32_t k = rayo_graph_symbol(g, plan->keys[i]);
    int64_t v;
    if (k < 0 || !rayo_json_as_int(plan->items[i], &v) || v < 0 || v > g->symbol_max[k]) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "graph plan: %s", plan->keys[i]);
    }
    g->symbol_plan[k] = v;
  }
  return RAYO_OK;
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

static rayo_status read_input(const rayo_json *j, rayo_graph *g, rayo_graph_input *in, bool state,
                              rayo_message *msg) {
  in->name = rayo_json_as_string(rayo_json_get(j, "name"));
  const rayo_json *shape = rayo_json_get(j, "shape");
  if (!in->name ||
      read_dtype(rayo_json_as_string(rayo_json_get(j, "dtype")), &in->dtype) != RAYO_OK || !shape ||
      shape->kind != RAYO_JSON_ARRAY || shape->length > RAYO_MAX_RANK) {
    return rayo_fail(msg, RAYO_ERR_FORMAT,
                     "a graph input or state needs a name, a dtype and a shape");
  }
  if (state && in->dtype != RAYO_DTYPE_F32) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "state %s: only float32", in->name);
  }
  in->rank = (int32_t)shape->length;
  for (size_t i = 0; i < shape->length; i++) {
    const char *sym = rayo_json_as_string(shape->items[i]);
    rayo_dim *d = &in->dims[i];
    if (sym && state) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "state %s: sizes must be whole numbers", in->name);
    } else if (sym) {
      d->symbol = rayo_graph_symbol(g, sym);
      if (d->symbol < 0)
        return rayo_fail(msg, RAYO_ERR_FORMAT, "input %s: no symbol %s", in->name, sym);
    } else if (!rayo_json_as_int(shape->items[i], &d->value) || d->value < 1 ||
               d->value > (1 << 24)) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "input %s: bad dimension", in->name);
    } else {
      d->symbol = -1;
    }
  }
  return RAYO_OK;
}

/* A name in scope before node `upto`: an earlier node, an input, a state,
 * else a weight. Later definitions shadow earlier ones. */
static bool resolve(const rayo_graph *g, int32_t upto, const rayo_pcwt *w, const char *name,
                    rayo_ref *ref) {
  for (int32_t i = upto - 1; i >= 0; i--) {
    if (strcmp(g->nodes[i].output, name) == 0) {
      *ref = (rayo_ref){RAYO_REF_NODE, i};
      return true;
    }
  }
  for (int32_t i = 0; i < g->input_count; i++) {
    if (strcmp(g->inputs[i].name, name) == 0) {
      *ref = (rayo_ref){RAYO_REF_INPUT, i};
      return true;
    }
  }
  for (int32_t i = 0; i < g->state_count; i++) {
    if (strcmp(g->states[i].name, name) == 0) {
      *ref = (rayo_ref){RAYO_REF_STATE, i};
      return true;
    }
  }
  for (size_t i = 0; i < w->weight_count; i++) {
    if (strcmp(w->weights[i].name, name) == 0) {
      *ref = (rayo_ref){RAYO_REF_WEIGHT, (int32_t)i};
      return true;
    }
  }
  return false;
}

static int32_t symbol_of(const void *graph, const char *name) {
  return rayo_graph_symbol(graph, name);
}

static rayo_status read_node(const rayo_json *j, rayo_graph *g, int32_t index, const rayo_pcwt *w,
                             rayo_arena *arena, rayo_message *msg) {
  rayo_node *n = &g->nodes[index];
  const char *op = rayo_json_as_string(rayo_json_get(j, "op"));
  n->output = rayo_json_as_string(rayo_json_get(j, "output"));
  const rayo_json *inputs = rayo_json_get(j, "inputs");
  if (!op || !n->output || !inputs || inputs->kind != RAYO_JSON_ARRAY) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "graph node %d: needs an op, inputs and an output",
                     (int)index);
  }
  n->op = rayo_op_find(op);
  if (!n->op) return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "node %s: unknown op %s", n->output, op);
  if ((int)inputs->length < n->op->min_inputs || (int)inputs->length > n->op->max_inputs) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: %s takes %d to %d inputs", n->output, op,
                     n->op->min_inputs, n->op->max_inputs);
  }
  n->input_count = (int32_t)inputs->length;
  rayo_tensor seen[RAYO_MAX_INPUTS];
  memset(seen, 0, sizeof seen);
  for (int32_t i = 0; i < n->input_count; i++) {
    const char *name = rayo_json_as_string(inputs->items[i]);
    if (!name || !resolve(g, index, w, name, &n->inputs[i])) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "node %s: no input, node or weight %s", n->output,
                       name ? name : "(not a string)");
    }
    if (n->inputs[i].kind == RAYO_REF_WEIGHT) {
      const rayo_weight *wt = &w->weights[n->inputs[i].index];
      seen[i] = (rayo_tensor){wt->dtype, wt->shape, (void *)(uintptr_t)wt->data, wt};
    } else if (n->inputs[i].kind == RAYO_REF_STATE) {
      const rayo_graph_input *st = &g->states[n->inputs[i].index];
      seen[i].dtype = st->dtype;
      seen[i].shape.rank = st->rank;
      for (int32_t d = 0; d < st->rank; d++) seen[i].shape.dims[d] = st->dims[d].value;
    }
  }
  rayo_op_spec spec = {j, n->output, n->input_count, seen, symbol_of, g};
  return n->op->prepare(&spec, arena, &n->params, msg);
}

static rayo_status read_outputs(const rayo_json *list, rayo_graph *g, rayo_arena *arena,
                                rayo_message *msg) {
  if (!list || list->kind != RAYO_JSON_ARRAY || list->length == 0 || list->length > 64) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "the graph needs 1 to 64 outputs");
  }
  g->outputs = rayo_arena_array(arena, list->length, sizeof *g->outputs);
  if (!g->outputs) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  g->output_count = (int32_t)list->length;
  for (size_t i = 0; i < list->length; i++) {
    const char *name = rayo_json_as_string(rayo_json_get(list->items[i], "name"));
    const char *node = rayo_json_as_string(rayo_json_get(list->items[i], "node"));
    rayo_ref ref;
    if (!name || !node || !resolve(g, g->node_count, &(rayo_pcwt){0}, node, &ref) ||
        ref.kind != RAYO_REF_NODE) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "graph output %d: needs a name and a node", (int)i);
    }
    g->outputs[i] = (rayo_graph_output){name, ref.index};
  }
  return RAYO_OK;
}

rayo_status rayo_graph_read(const rayo_json *json, const rayo_pcwt *weights, rayo_arena *arena,
                            rayo_graph *out, rayo_message *msg) {
  memset(out, 0, sizeof *out);
  int64_t version;
  if (!json || json->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "no graph in the header: not a rayo model");
  }
  if (!rayo_json_as_int(rayo_json_get(json, "version"), &version) ||
      version != RAYO_GRAPH_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "graph version: rayo reads %d", RAYO_GRAPH_VERSION);
  }
  rayo_status s = read_symbols(json, out, msg);
  if (s != RAYO_OK) return s;
  const rayo_json *inputs = rayo_json_get(json, "inputs");
  const rayo_json *nodes = rayo_json_get(json, "nodes");
  if (!inputs || inputs->kind != RAYO_JSON_ARRAY || inputs->length == 0 || inputs->length > 64 ||
      !nodes || nodes->kind != RAYO_JSON_ARRAY || nodes->length == 0 ||
      nodes->length > RAYO_MAX_NODES) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "the graph needs inputs and nodes");
  }
  out->inputs = rayo_arena_array(arena, inputs->length, sizeof *out->inputs);
  out->nodes = rayo_arena_array(arena, nodes->length, sizeof *out->nodes);
  if (!out->inputs || !out->nodes) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  out->input_count = (int32_t)inputs->length;
  for (size_t i = 0; i < inputs->length; i++) {
    s = read_input(inputs->items[i], out, &out->inputs[i], false, msg);
    if (s != RAYO_OK) return s;
  }
  const rayo_json *states = rayo_json_get(json, "states");
  if (states) {
    if (states->kind != RAYO_JSON_ARRAY || states->length > 256) {
      return rayo_fail(msg, RAYO_ERR_FORMAT, "graph states: a list of at most 256");
    }
    out->states = rayo_arena_array(arena, states->length ? states->length : 1, sizeof *out->states);
    if (!out->states) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    out->state_count = (int32_t)states->length;
    for (size_t i = 0; i < states->length; i++) {
      s = read_input(states->items[i], out, &out->states[i], true, msg);
      if (s != RAYO_OK) return s;
    }
  }
  for (size_t i = 0; i < nodes->length; i++) {
    out->node_count = (int32_t)i;
    s = read_node(nodes->items[i], out, (int32_t)i, weights, arena, msg);
    if (s != RAYO_OK) return s;
  }
  out->node_count = (int32_t)nodes->length;
  return read_outputs(rayo_json_get(json, "outputs"), out, arena, msg);
}
