#include "graph/plan.h"

#include <string.h>

#include "core/alloc.h"
#include "core/compat.h"
#include "core/message.h"
#include "rayo/config.h"

#include "core/small_begin.h"

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
  /* The memory stays the node's whatever infer says: a bind that fails
   * puts the shapes back, and must find the planned memory there. */
  t->data = data;
  if (s == RAYO_OK && !fits(&t->shape)) {
    s = rayo_fail(msg, RAYO_ERR_SHAPE, "too large");
  }
  if (s != RAYO_OK) {
    rayo_message why;
    why.text[0] = '\0';
    if (msg) why = *msg;
    return rayo_fail(msg, s, "node %s: %s", n->output, why.text);
  }
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

/* a + b, or RAYO_FALSE when that is more than RAYO_MAX_ARENA_BYTES: every
 * offset of the plan stays below it, on 32-bit systems too. */
static rayo_bool add_bytes(size_t a, size_t b, size_t *out) {
  if (a > (size_t)RAYO_MAX_ARENA_BYTES || b > (size_t)RAYO_MAX_ARENA_BYTES - a) return RAYO_FALSE;
  *out = a + b;
  return RAYO_TRUE;
}

/* `size` rounded up to RAYO_ALIGN, within RAYO_MAX_ARENA_BYTES. */
static rayo_bool aligned_bytes(size_t size, size_t *out) {
  if (!add_bytes(size, RAYO_ALIGN - 1, out)) return RAYO_FALSE;
  *out &= ~(size_t)(RAYO_ALIGN - 1);
  return RAYO_TRUE;
}

static rayo_bool sum_fixed(const rayo_tensor *tensors, rayo_i32 count, size_t *offsets,
                           size_t *base) {
  rayo_i32 i;
  for (i = 0; i < count; i++) {
    size_t size;
    offsets[i] = *base;
    if (!aligned_bytes(rayo_tensor_storage(&tensors[i]), &size) || !add_bytes(*base, size, base)) {
      return RAYO_FALSE;
    }
  }
  return RAYO_TRUE;
}

static rayo_status too_large(rayo_message *msg) {
  return rayo_fail(msg, RAYO_ERR_LIMIT,
                   "the session would need more than %ld bytes (RAYO_MAX_ARENA_BYTES)",
                   (long)RAYO_MAX_ARENA_BYTES);
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
  base = 0;
  if (!sum_fixed(values->states, graph->state_count, out->state_offset, &base) ||
      !sum_fixed(values->inputs, graph->input_count, out->input_offset, &base)) {
    return too_large(msg);
  }
  for (i = 0; i < n; i++) {
    if (!graph->nodes[i].op->aliases_input) {
      size_t size, at, end;
      if (!aligned_bytes(rayo_tensor_bytes(&values->nodes[i]), &size)) return too_large(msg);
      /* Every live block ends within `top`, so `at` is at most `top`. */
      at = place(live, count, size);
      if (!add_bytes(at, size, &end) || !add_bytes(base, end, &out->node_offset[i])) {
        return too_large(msg);
      }
      out->node_offset[i] = base + at;
      insert(live, &count, at, size, i);
      if (end > top) top = end;
    }
    /* Nodes not read after this one are free from now on. */
    for (j = 0; j <= i; j++) {
      if (last[j] == i) release(live, &count, j);
    }
  }
  if (!add_bytes(base, top, &out->scratch_offset) || !aligned_bytes(scratch, &out->scratch_bytes) ||
      !add_bytes(out->scratch_offset, out->scratch_bytes, &out->arena_bytes)) {
    return too_large(msg);
  }
  return RAYO_OK;
}

#include "core/small_end.h"
