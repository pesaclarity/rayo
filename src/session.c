#include "session.h"

#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"

static rayo_status plan(rayo_session *s, rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  rayo_values *v = &s->values;
  v->inputs = rayo_arena_array(&s->book, (size_t)g->input_count, sizeof *v->inputs);
  v->states = rayo_arena_array(&s->book, (size_t)g->state_count + 1, sizeof *v->states);
  v->nodes = rayo_arena_array(&s->book, (size_t)g->node_count, sizeof *v->nodes);
  if (!v->inputs || !v->states || !v->nodes) {
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (int32_t i = 0; i < g->symbol_count; i++) s->symbols[i] = g->symbol_plan[i];
  size_t scratch;
  rayo_status st = rayo_graph_shapes(g, &s->model->pcwt, s->symbols, v, &scratch, msg);
  if (st != RAYO_OK) return st;
  st = rayo_plan_make(g, v, scratch, &s->book, &s->plan, msg);
  if (st != RAYO_OK) return st;
  s->planned = rayo_arena_array(&s->book, (size_t)g->node_count, sizeof *s->planned);
  if (!s->planned) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (int32_t i = 0; i < g->node_count; i++) s->planned[i] = rayo_tensor_bytes(&v->nodes[i]);
  s->memory = rayo_aligned_alloc(s->plan.arena_bytes);
  if (!s->memory) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (int32_t i = 0; i < g->state_count; i++)
    v->states[i].data = s->memory + s->plan.state_offset[i];
  for (int32_t i = 0; i < g->input_count; i++)
    v->inputs[i].data = s->memory + s->plan.input_offset[i];
  for (int32_t i = 0; i < g->node_count; i++) v->nodes[i].data = s->memory + s->plan.node_offset[i];
  /* Again, so the nodes that view a state get its memory. */
  return rayo_graph_shapes(g, &s->model->pcwt, s->symbols, v, &scratch, msg);
}

rayo_status rayo_session_create(const rayo_model *model, rayo_session **out, rayo_message *msg) {
  if (!model || !out) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *out = NULL;
  rayo_session *s = calloc(1, sizeof *s);
  if (!s) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  s->model = model;
  rayo_kernels_best(&s->kernels);
  rayo_status st = plan(s, msg);
  if (st != RAYO_OK) {
    rayo_session_destroy(s);
    return st;
  }
  *out = s;
  return RAYO_OK;
}

void rayo_session_destroy(rayo_session *session) {
  if (!session) return;
  rayo_aligned_free(session->memory);
  rayo_arena_free(&session->book);
  free(session);
}

/* Sizes bound together must fit the memory planned for them. */
static rayo_status within_plan(const rayo_session *s, size_t scratch, rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  if (rayo_align_up(scratch) > s->plan.scratch_bytes) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "these sizes need more scratch than planned");
  }
  for (int32_t i = 0; i < g->node_count; i++) {
    if (!g->nodes[i].op->aliases_input && rayo_tensor_bytes(&s->values.nodes[i]) > s->planned[i]) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "node %s: these sizes need more than planned",
                       g->nodes[i].output);
    }
  }
  return RAYO_OK;
}

rayo_status rayo_session_bind(rayo_session *session, const char *name, int64_t value,
                              rayo_message *msg) {
  if (!session || !name) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  const rayo_graph *g = &session->model->graph;
  int32_t i = rayo_graph_symbol(g, name);
  if (i < 0) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "no size symbol %s", name);
  if (value < 0 || value > g->symbol_max[i]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "%s = %lld, it must be 0 to %lld", name, (long long)value,
                     (long long)g->symbol_max[i]);
  }
  int64_t before = session->symbols[i];
  session->symbols[i] = value;
  size_t scratch;
  rayo_status st = rayo_graph_shapes(g, &session->model->pcwt, session->symbols, &session->values,
                                     &scratch, msg);
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
  if (!session) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  const rayo_graph *g = &session->model->graph;
  const rayo_exec exec = {&session->kernels, session->memory + session->plan.scratch_offset,
                          session->symbols};
  for (int32_t i = 0; i < g->node_count; i++) {
    const rayo_node *n = &g->nodes[i];
    rayo_tensor args[RAYO_MAX_INPUTS];
    rayo_graph_node_inputs(g, &session->model->pcwt, i, &session->values, args);
    rayo_status st = n->op->run(n->params, args, &session->values.nodes[i], &exec);
    if (st != RAYO_OK) {
      return rayo_fail(msg, st, "node %s: %s", n->output,
                       st == RAYO_ERR_SHAPE ? "an id out of range" : rayo_status_str(st));
    }
  }
  return RAYO_OK;
}

rayo_status rayo_session_output(const rayo_session *session, int index, rayo_tensor_view *out) {
  if (!session || !out || index < 0 || index >= session->model->graph.output_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  const rayo_graph_output *o = &session->model->graph.outputs[index];
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
}

const char *rayo_session_kernels(const rayo_session *session) {
  return session ? session->kernels.name : "";
}

void rayo_session_reset(rayo_session *session) {
  if (!session) return;
  const rayo_graph *g = &session->model->graph;
  for (int32_t i = 0; i < g->state_count; i++) {
    memset(session->values.states[i].data, 0, rayo_tensor_bytes(&session->values.states[i]));
  }
}
