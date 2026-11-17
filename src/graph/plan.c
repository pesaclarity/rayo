#include "graph/plan.h"

#include <string.h>

#include "core/alloc.h"
#include "core/message.h"

void rayo_graph_node_inputs(const rayo_graph *graph, const rayo_pcwt *weights, int32_t index,
                            const rayo_values *values, rayo_tensor *out) {
  const rayo_node *n = &graph->nodes[index];
  for (int32_t i = 0; i < n->input_count; i++) {
    const rayo_ref r = n->inputs[i];
    if (r.kind == RAYO_REF_INPUT) {
      out[i] = values->inputs[r.index];
    } else if (r.kind == RAYO_REF_STATE) {
      out[i] = values->states[r.index];
    } else if (r.kind == RAYO_REF_NODE) {
      out[i] = values->nodes[r.index];
    } else {
      const rayo_weight *w = &weights->weights[r.index];
      out[i] = (rayo_tensor){w->dtype, w->shape, (void *)(uintptr_t)w->data, w};
    }
  }
}

static void decl_shape(const rayo_graph_input *in, const int64_t *symbols, rayo_tensor *t) {
  t->dtype = in->dtype;
  t->shape.rank = in->rank;
  for (int32_t d = 0; d < in->rank; d++) {
    t->shape.dims[d] = in->dims[d].symbol >= 0 ? symbols[in->dims[d].symbol] : in->dims[d].value;
  }
}

rayo_status rayo_graph_shapes(const rayo_graph *graph, const rayo_pcwt *weights,
                              const int64_t *symbols, rayo_values *values, size_t *scratch,
                              rayo_message *msg) {
  for (int32_t i = 0; i < graph->input_count; i++) {
    decl_shape(&graph->inputs[i], symbols, &values->inputs[i]);
    if (rayo_shape_count(&values->inputs[i].shape) < 1) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "input %s would be empty", graph->inputs[i].name);
    }
  }
  for (int32_t i = 0; i < graph->state_count; i++) {
    decl_shape(&graph->states[i], symbols, &values->states[i]);
  }
  *scratch = 0;
  for (int32_t i = 0; i < graph->node_count; i++) {
    const rayo_node *n = &graph->nodes[i];
    rayo_tensor args[RAYO_MAX_INPUTS];
    rayo_graph_node_inputs(graph, weights, i, values, args);
    rayo_tensor *t = &values->nodes[i];
    void *data = n->op->aliases_input ? args[0].data : t->data;
    size_t need = 0;
    memset(t, 0, sizeof *t);
    rayo_status s = n->op->infer(n->params, args, symbols, t, &need, msg);
    if (s != RAYO_OK) {
      rayo_message why = {{0}};
      if (msg) why = *msg;
      return rayo_fail(msg, s, "node %s: %s", n->output, why.text);
    }
    t->data = data;
    if (need > *scratch) *scratch = need;
  }
  return RAYO_OK;
}

typedef struct block {
  size_t offset;
  size_t size;
  int32_t owner; /* node, or -1 for an input */
} block;

/* The lowest offset where `size` bytes fit between the live blocks (sorted
 * by offset), best fit first. */
static size_t place(const block *live, int32_t count, size_t size) {
  size_t best = SIZE_MAX, best_gap = SIZE_MAX, at = 0;
  for (int32_t i = 0; i < count; i++) {
    if (live[i].offset >= at && live[i].offset - at >= size && live[i].offset - at < best_gap) {
      best = at;
      best_gap = live[i].offset - at;
    }
    if (live[i].offset + live[i].size > at) at = live[i].offset + live[i].size;
  }
  return best != SIZE_MAX ? best : at;
}

static void insert(block *live, int32_t *count, block b) {
  int32_t i = *count;
  while (i > 0 && live[i - 1].offset > b.offset) {
    live[i] = live[i - 1];
    i--;
  }
  live[i] = b;
  (*count)++;
}

static void release(block *live, int32_t *count, int32_t owner) {
  for (int32_t i = 0; i < *count; i++) {
    if (live[i].owner == owner) {
      memmove(&live[i], &live[i + 1], (size_t)(*count - i - 1) * sizeof *live);
      (*count)--;
      return;
    }
  }
}

rayo_status rayo_plan_make(const rayo_graph *graph, const rayo_values *values, size_t scratch,
                           rayo_arena *arena, rayo_plan *out, rayo_message *msg) {
  const int32_t n = graph->node_count;
  out->state_offset = rayo_arena_array(arena, (size_t)graph->state_count + 1, sizeof(size_t));
  out->input_offset = rayo_arena_array(arena, (size_t)graph->input_count, sizeof(size_t));
  out->node_offset = rayo_arena_array(arena, (size_t)n, sizeof(size_t));
  int32_t *last = rayo_arena_array(arena, (size_t)n, sizeof *last);
  block *live = rayo_arena_array(arena, (size_t)(n + 1), sizeof *live);
  if (!out->state_offset || !out->input_offset || !out->node_offset || !last || !live) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (int32_t i = 0; i < n; i++) {
    last[i] = i;
    const rayo_node *node = &graph->nodes[i];
    for (int32_t a = 0; a < node->input_count; a++) {
      if (node->inputs[a].kind == RAYO_REF_NODE) last[node->inputs[a].index] = i;
    }
  }
  for (int32_t o = 0; o < graph->output_count; o++) last[graph->outputs[o].node] = n;
  /* States and inputs live for the whole run, below everything else. */
  size_t base = 0;
  for (int32_t i = 0; i < graph->state_count; i++) {
    out->state_offset[i] = base;
    base += rayo_align_up(rayo_tensor_bytes(&values->states[i]));
  }
  for (int32_t i = 0; i < graph->input_count; i++) {
    out->input_offset[i] = base;
    base += rayo_align_up(rayo_tensor_bytes(&values->inputs[i]));
  }
  int32_t count = 0;
  size_t top = 0;
  for (int32_t i = 0; i < n; i++) {
    if (!graph->nodes[i].op->aliases_input) {
      size_t size = rayo_align_up(rayo_tensor_bytes(&values->nodes[i]));
      if (size == 0) {
        return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: too large", graph->nodes[i].output);
      }
      size_t at = place(live, count, size);
      out->node_offset[i] = base + at;
      insert(live, &count, (block){at, size, i});
      if (at + size > top) top = at + size;
    }
    /* Nodes not read after this one are free from now on. */
    for (int32_t j = 0; j <= i; j++) {
      if (last[j] == i) release(live, &count, j);
    }
  }
  out->scratch_offset = base + top;
  out->scratch_bytes = rayo_align_up(scratch);
  out->arena_bytes = out->scratch_offset + out->scratch_bytes;
  return RAYO_OK;
}
