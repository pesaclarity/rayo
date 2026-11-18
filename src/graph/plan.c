#include "graph/plan.h"

#include <string.h>

#include "core/alloc.h"
#include "core/compat.h"
#include "core/message.h"

void rayo_graph_node_inputs(const rayo_graph *graph, const rayo_pcwt *weights, rayo_i32 index,
                            const rayo_values *values, rayo_tensor *out) {
  const rayo_node *n = &graph->nodes[index];
  rayo_i32 i;
  for (i = 0; i < n->input_count; i++) {
    const rayo_ref r = n->inputs[i];
    if (r.kind == RAYO_REF_INPUT) {
      out[i] = values->inputs[r.index];
    } else if (r.kind == RAYO_REF_STATE) {
      out[i] = values->states[r.index];
    } else if (r.kind == RAYO_REF_NODE) {
      out[i] = values->nodes[r.index];
    } else {
      out[i] = rayo_tensor_of_weight(&weights->weights[r.index]);
    }
  }
}

static void decl_shape(const rayo_graph_input *in, const rayo_i32 *symbols, rayo_tensor *t) {
  rayo_i32 d;
  t->dtype = in->dtype;
  t->shape.rank = in->rank;
  for (d = 0; d < in->rank; d++) {
    t->shape.dims[d] = in->dims[d].symbol >= 0 ? symbols[in->dims[d].symbol] : in->dims[d].value;
  }
}

/* Every tensor's bytes stay below 2^31, so they fit size_t everywhere. */
static rayo_bool fits(const rayo_shape *shape) {
  size_t n = 1;
  rayo_i32 i;
  for (i = 0; i < shape->rank; i++) {
    if (shape->dims[i] < 0) return RAYO_FALSE;
    if (shape->dims[i] != 0 && n > (size_t)(RAYO_I32_MAX / 4) / (size_t)shape->dims[i]) {
      return RAYO_FALSE;
    }
    n *= (size_t)shape->dims[i];
  }
  return RAYO_TRUE;
}

static rayo_status node_shape(const rayo_graph *graph, const rayo_pcwt *weights,
                              const rayo_i32 *symbols, rayo_values *values, rayo_i32 i,
                              size_t *need, rayo_message *msg) {
  const rayo_node *n = &graph->nodes[i];
  rayo_tensor *args = values->args;
  rayo_tensor *t = &values->nodes[i];
  void *data;
  rayo_status s;
  rayo_graph_node_inputs(graph, weights, i, values, args);
  data = n->op->aliases_input ? args[0].data : t->data;
  memset(t, 0, sizeof *t);
  *need = 0;
  s = n->op->infer(n->params, args, symbols, t, need, msg);
  if (s == RAYO_OK && !fits(&t->shape)) {
    s = rayo_fail(msg, RAYO_ERR_SHAPE, "too large");
  }
  if (s != RAYO_OK) {
    rayo_message why;
    why.text[0] = '\0';
    if (msg) why = *msg;
    return rayo_fail(msg, s, "node %s: %s", n->output, why.text);
  }
  t->data = data;
  return RAYO_OK;
}

rayo_status rayo_graph_shapes(const rayo_graph *graph, const rayo_pcwt *weights,
                              const rayo_i32 *symbols, rayo_values *values, size_t *scratch,
                              rayo_message *msg) {
  rayo_i32 i;
  for (i = 0; i < graph->input_count; i++) {
    decl_shape(&graph->inputs[i], symbols, &values->inputs[i]);
    if (!fits(&values->inputs[i].shape) || rayo_shape_count(&values->inputs[i].shape) < 1) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "input %s would be empty or too large",
                       graph->inputs[i].name);
    }
  }
  for (i = 0; i < graph->state_count; i++) {
    rayo_tensor *st = &values->states[i];
    decl_shape(&graph->states[i], symbols, st);
    if (values->state_rows && values->state_rows[i] > 0 && st->shape.rank >= 1 &&
        st->shape.dims[0] > values->state_rows[i]) {
      st->shape.dims[0] = values->state_rows[i];
    }
    if (values->state_group && values->state_group[i] > 0) {
      st->dtype = RAYO_DTYPE_I8;
      st->group = values->state_group[i];
    }
    if (!fits(&values->states[i].shape)) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "state %s is too large", graph->states[i].name);
    }
  }
  *scratch = 0;
  for (i = 0; i < graph->node_count; i++) {
    size_t need;
    rayo_status s = node_shape(graph, weights, symbols, values, i, &need, msg);
    if (s != RAYO_OK) return s;
    if (need > *scratch) *scratch = need;
  }
  return RAYO_OK;
}

typedef struct block {
  size_t offset;
  size_t size;
  rayo_i32 owner; /* the node */
} block;

/* The lowest offset where `size` bytes fit between the live blocks (sorted
 * by offset), best fit first. */
static size_t place(const block *live, rayo_i32 count, size_t size) {
  size_t best = RAYO_SIZE_MAX, best_gap = RAYO_SIZE_MAX, at = 0;
  rayo_i32 i;
  for (i = 0; i < count; i++) {
    if (live[i].offset >= at && live[i].offset - at >= size && live[i].offset - at < best_gap) {
      best = at;
      best_gap = live[i].offset - at;
    }
    if (live[i].offset + live[i].size > at) at = live[i].offset + live[i].size;
  }
  return best != RAYO_SIZE_MAX ? best : at;
}

static void insert(block *live, rayo_i32 *count, size_t offset, size_t size, rayo_i32 owner) {
  rayo_i32 i = *count;
  while (i > 0 && live[i - 1].offset > offset) {
    live[i] = live[i - 1];
    i--;
  }
  live[i].offset = offset;
  live[i].size = size;
  live[i].owner = owner;
  (*count)++;
}

static void release(block *live, rayo_i32 *count, rayo_i32 owner) {
  rayo_i32 i;
  for (i = 0; i < *count; i++) {
    if (live[i].owner == owner) {
      memmove(&live[i], &live[i + 1], (size_t)(*count - i - 1) * sizeof *live);
      (*count)--;
      return;
    }
  }
}

/* For each node, the last node that reads it (the node count for outputs). */
static void last_uses(const rayo_graph *graph, rayo_i32 *last) {
  rayo_i32 i, a, o;
  for (i = 0; i < graph->node_count; i++) {
    const rayo_node *node = &graph->nodes[i];
    last[i] = i;
    for (a = 0; a < node->input_count; a++) {
      if (node->inputs[a].kind == RAYO_REF_NODE) last[node->inputs[a].index] = i;
    }
  }
  for (o = 0; o < graph->output_count; o++) last[graph->outputs[o].node] = graph->node_count;
}

static size_t sum_fixed(const rayo_tensor *tensors, rayo_i32 count, size_t *offsets, size_t base) {
  rayo_i32 i;
  for (i = 0; i < count; i++) {
    offsets[i] = base;
    base += rayo_align_up(rayo_tensor_storage(&tensors[i]));
  }
  return base;
}

rayo_status rayo_plan_make(const rayo_graph *graph, const rayo_values *values, size_t scratch,
                           rayo_arena *arena, rayo_plan *out, rayo_message *msg) {
  const rayo_i32 n = graph->node_count;
  rayo_i32 *last;
  block *live;
  rayo_i32 count = 0, i, j;
  size_t base, top = 0;
  out->state_offset = rayo_arena_array(arena, (size_t)graph->state_count + 1, sizeof(size_t));
  out->input_offset = rayo_arena_array(arena, (size_t)graph->input_count, sizeof(size_t));
  out->node_offset = rayo_arena_array(arena, (size_t)n, sizeof(size_t));
  last = rayo_arena_array(arena, (size_t)n, sizeof *last);
  live = rayo_arena_array(arena, (size_t)n + 1, sizeof *live);
  if (!out->state_offset || !out->input_offset || !out->node_offset || !last || !live) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  last_uses(graph, last);
  /* States and inputs live for the whole run, below everything else. */
  base = sum_fixed(values->states, graph->state_count, out->state_offset, 0);
  base = sum_fixed(values->inputs, graph->input_count, out->input_offset, base);
  for (i = 0; i < n; i++) {
    if (!graph->nodes[i].op->aliases_input) {
      size_t size = rayo_align_up(rayo_tensor_bytes(&values->nodes[i]));
      size_t at = place(live, count, size);
      out->node_offset[i] = base + at;
      insert(live, &count, at, size, i);
      if (at + size > top) top = at + size;
    }
    /* Nodes not read after this one are free from now on. */
    for (j = 0; j <= i; j++) {
      if (last[j] == i) release(live, &count, j);
    }
  }
  out->scratch_offset = base + top;
  out->scratch_bytes = rayo_align_up(scratch);
  out->arena_bytes = out->scratch_offset + out->scratch_bytes;
  return RAYO_OK;
}
