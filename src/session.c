#include "session.h"

#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"
#include "ops/ops.h"

/* Whether node `n` reads state `index` (as `kind`) or node output `index`. */
static rayo_bool reads(const rayo_node *n, rayo_ref_kind kind, rayo_i32 index, rayo_i32 *at) {
  rayo_i32 a;
  for (a = 0; a < n->input_count; a++) {
    if (n->inputs[a].kind == kind && n->inputs[a].index == index) {
      *at = a;
      return RAYO_TRUE;
    }
  }
  return RAYO_FALSE;
}

/* The int8 cache's group for state `index`: its head size, when only
 * cache_write nodes write it and only attention nodes read what they
 * write; else 0 (float32). */
static rayo_i32 cache_group(const rayo_graph *g, rayo_i32 index) {
  const rayo_graph_input *st = &g->states[index];
  rayo_i32 i, j, o, at, group = 0;
  rayo_i32 width;
  if (st->rank != 2) return 0;
  width = st->dims[1].value;
  for (i = 0; i < g->node_count; i++) {
    if (!reads(&g->nodes[i], RAYO_REF_STATE, index, &at)) continue;
    if (g->nodes[i].op != &rayo_op_cache_write || at != 0) return 0;
    for (o = 0; o < g->output_count; o++) {
      if (g->outputs[o].node == i) return 0;
    }
    for (j = i + 1; j < g->node_count; j++) {
      rayo_i32 kv, hd;
      if (!reads(&g->nodes[j], RAYO_REF_NODE, i, &at)) continue;
      if (g->nodes[j].op != &rayo_op_attention || at == 0) return 0;
      kv = rayo_attention_kv_heads(g->nodes[j].params);
      if (kv < 1 || width % kv != 0) return 0;
      hd = width / kv;
      if (group != 0 && group != hd) return 0;
      group = hd;
    }
  }
  return group;
}

static rayo_status apply_options(rayo_session *s, const rayo_session_options *o,
                                 rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  rayo_i32 i;
  int l;
  s->symbol_max = rayo_arena_array(&s->book, (size_t)g->symbol_count + 1, sizeof *s->symbol_max);
  s->symbols = rayo_arena_array(&s->book, (size_t)g->symbol_count + 1, sizeof *s->symbols);
  s->state_rows = rayo_arena_array(&s->book, (size_t)g->state_count + 1, sizeof *s->state_rows);
  s->state_group = rayo_arena_array(&s->book, (size_t)g->state_count + 1, sizeof *s->state_group);
  if (!s->symbol_max || !s->symbols || !s->state_rows || !s->state_group) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (i = 0; i < g->symbol_count; i++) {
    s->symbol_max[i] = g->symbol_max[i];
    s->symbols[i] = g->symbol_plan[i];
  }
  if (!o) return RAYO_OK;
  if (o->state_rows < 0 || o->limit_count < 0 || (o->limit_count > 0 && !o->limits) ||
      (o->cache != RAYO_CACHE_FLOAT32 && o->cache != RAYO_CACHE_INT8)) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "bad session options");
  }
  for (l = 0; l < o->limit_count; l++) {
    const rayo_symbol_limit *lim = &o->limits[l];
    i = lim->name ? rayo_graph_symbol(g, lim->name) : -1;
    if (i < 0) {
      return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "no size symbol %s",
                       lim->name ? lim->name : "(null)");
    }
    if (lim->max < 0 || lim->max > g->symbol_max[i] || lim->plan < 0 || lim->plan > lim->max) {
      return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT,
                       "%s: a limit of 0 to %ld, planned at most at the limit", lim->name,
                       (long)g->symbol_max[i]);
    }
    s->symbol_max[i] = lim->max;
    s->symbols[i] = lim->plan;
  }
  for (i = 0; i < g->state_count; i++) {
    s->state_rows[i] = o->state_rows;
    s->state_group[i] = o->cache == RAYO_CACHE_INT8 ? cache_group(g, i) : 0;
  }
  return RAYO_OK;
}

static rayo_status plan(rayo_session *s, const rayo_session_options *o, rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  rayo_values *v = &s->values;
  size_t scratch;
  rayo_status st;
  rayo_i32 i;
  v->inputs = rayo_arena_array(&s->book, (size_t)g->input_count, sizeof *v->inputs);
  v->states = rayo_arena_array(&s->book, (size_t)g->state_count + 1, sizeof *v->states);
  v->nodes = rayo_arena_array(&s->book, (size_t)g->node_count, sizeof *v->nodes);
  v->args = rayo_arena_array(&s->book, (size_t)g->max_node_inputs + 1, sizeof *v->args);
  if (!v->inputs || !v->states || !v->nodes || !v->args) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  st = apply_options(s, o, msg);
  if (st != RAYO_OK) return st;
  v->state_rows = s->state_rows;
  v->state_group = s->state_group;
  st = rayo_graph_shapes(g, &s->model->pcwt, s->symbols, v, &scratch, msg);
  if (st != RAYO_OK) return st;
  st = rayo_plan_make(g, v, scratch, &s->book, &s->plan, msg);
  if (st != RAYO_OK) return st;
  s->planned = rayo_arena_array(&s->book, (size_t)g->node_count, sizeof *s->planned);
  if (!s->planned) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (i = 0; i < g->node_count; i++) s->planned[i] = rayo_tensor_bytes(&v->nodes[i]);
  /* Whole pages from the system: a cache's rows cost memory only once
   * written, and a reset gives them back. */
  s->memory = rayo_pages_alloc(s->plan.arena_bytes);
  if (!s->memory) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s->memory_bytes = s->plan.arena_bytes;
  for (i = 0; i < g->state_count; i++) {
    v->states[i].data = s->memory + s->plan.state_offset[i];
    if (v->states[i].group > 0) {
      v->states[i].scales = rayo_tensor_scales_at(v->states[i].data, &v->states[i]);
    }
  }
  for (i = 0; i < g->input_count; i++) v->inputs[i].data = s->memory + s->plan.input_offset[i];
  for (i = 0; i < g->node_count; i++) v->nodes[i].data = s->memory + s->plan.node_offset[i];
  /* Again, so the nodes that view a state get its memory. */
  return rayo_graph_shapes(g, &s->model->pcwt, s->symbols, v, &scratch, msg);
}

rayo_status rayo_session_create(const rayo_model *model, rayo_session **out, rayo_message *msg) {
  return rayo_session_create_with(model, NULL, out, msg);
}

rayo_status rayo_session_create_with(const rayo_model *model, const rayo_session_options *options,
                                     rayo_session **out, rayo_message *msg) {
  rayo_session *s;
  int threads = options ? options->threads : 1;
  rayo_status st;
  if (!model || !out) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *out = NULL;
  s = calloc(1, sizeof *s);
  if (!s) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s->model = model;
  rayo_kernels_best(&s->kernels);
  if (threads < 0) threads = rayo_cpu_fast_cores();
  s->pool = rayo_pool_create(threads);
  st = plan(s, options, msg);
  if (st != RAYO_OK) {
    rayo_session_destroy(s);
    return st;
  }
  *out = s;
  return RAYO_OK;
}

void rayo_session_destroy(rayo_session *session) {
  if (!session) return;
  rayo_pool_destroy(session->pool);
  rayo_pages_free(session->memory, session->memory_bytes);
  rayo_arena_free(&session->book);
  free(session);
}

/* Sizes bound together must fit the memory planned for them. */
static rayo_status within_plan(const rayo_session *s, size_t scratch, rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  rayo_i32 i;
  if (rayo_align_up(scratch) > s->plan.scratch_bytes) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "these sizes need more scratch than planned");
  }
  for (i = 0; i < g->node_count; i++) {
    if (!g->nodes[i].op->aliases_input && rayo_tensor_bytes(&s->values.nodes[i]) > s->planned[i]) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: these sizes need more than planned",
                       g->nodes[i].output);
    }
  }
  return RAYO_OK;
}

rayo_status rayo_session_bind(rayo_session *session, const char *name, rayo_i32 value,
                              rayo_message *msg) {
  const rayo_graph *g;
  rayo_i32 i, before;
  size_t scratch;
  rayo_status st;
  if (!session || !name) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  g = &session->model->graph;
  i = rayo_graph_symbol(g, name);
  if (i < 0) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "no size symbol %s", name);
  if (value < 0 || value > session->symbol_max[i]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "%s = %ld, it must be 0 to %ld", name, (long)value,
                     (long)session->symbol_max[i]);
  }
  before = session->symbols[i];
  session->symbols[i] = value;
  st = rayo_graph_shapes(g, &session->model->pcwt, session->symbols, &session->values, &scratch,
                         msg);
  if (st == RAYO_OK) st = within_plan(session, scratch, msg);
  if (st != RAYO_OK) {
    session->symbols[i] = before;
    rayo_graph_shapes(g, &session->model->pcwt, session->symbols, &session->values, &scratch, NULL);
  }
  return st;
}

static void view(const rayo_tensor *t, const char *name, rayo_tensor_view *out) {
  out->name = name;
  out->dtype = t->dtype;
  out->shape = t->shape;
  out->data = t->data;
  out->bytes = rayo_tensor_bytes(t);
}

rayo_status rayo_session_input(rayo_session *session, int index, rayo_tensor_view *out) {
  if (!session || !out || index < 0 || index >= session->model->graph.input_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  view(&session->values.inputs[index], session->model->graph.inputs[index].name, out);
  return RAYO_OK;
}

rayo_status rayo_session_run(rayo_session *session, rayo_message *msg) {
  const rayo_graph *g;
  rayo_exec exec;
  rayo_i32 i;
  if (!session) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  g = &session->model->graph;
  exec.kernels = &session->kernels;
  exec.scratch = session->memory + session->plan.scratch_offset;
  exec.symbols = session->symbols;
  exec.pool = session->pool;
  for (i = 0; i < g->node_count; i++) {
    const rayo_node *n = &g->nodes[i];
    rayo_status st;
    rayo_graph_node_inputs(g, &session->model->pcwt, i, &session->values, session->values.args);
    st = n->op->run(n->params, session->values.args, &session->values.nodes[i], &exec);
    if (st != RAYO_OK) {
      return rayo_fail(msg, st, "node %s: %s", n->output,
                       st == RAYO_ERR_SHAPE ? "an id out of range" : rayo_status_str(st));
    }
  }
  return RAYO_OK;
}

rayo_status rayo_session_output(const rayo_session *session, int index, rayo_tensor_view *out) {
  const rayo_graph_output *o;
  if (!session || !out || index < 0 || index >= session->model->graph.output_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  o = &session->model->graph.outputs[index];
  view(&session->values.nodes[o->node], o->name, out);
  return RAYO_OK;
}

void rayo_session_memory(const rayo_session *session, rayo_memory_info *out) {
  if (!out) return;
  memset(out, 0, sizeof *out);
  if (!session) return;
  rayo_model_memory(session->model, out);
  out->arena = session->plan.arena_bytes;
  out->overhead += session->book.bytes;
  out->states = session->plan.input_offset && session->model->graph.input_count > 0
                    ? session->plan.input_offset[0]
                    : 0;
}

const char *rayo_session_kernels(const rayo_session *session) {
  return session ? session->kernels.name : "";
}

void rayo_session_reset(rayo_session *session) {
  rayo_i32 i;
  if (!session) return;
  for (i = 0; i < session->model->graph.state_count; i++) {
    rayo_pages_zero(session->values.states[i].data,
                    rayo_tensor_storage(&session->values.states[i]));
  }
}

int rayo_session_threads(const rayo_session *session) {
  return session ? rayo_pool_threads(session->pool) : 1;
}
