#include "session.h"

#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"

static rayo_status plan(rayo_session *s, rayo_message *msg) {
  const rayo_graph *g = &s->model->graph;
  s->inputs = rayo_arena_array(&s->book, (size_t)g->input_count, sizeof *s->inputs);
  s->nodes = rayo_arena_array(&s->book, (size_t)g->node_count, sizeof *s->nodes);
  if (!s->inputs || !s->nodes) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (int32_t i = 0; i < g->symbol_count; i++) s->symbols[i] = g->symbol_max[i];
  size_t scratch;
  rayo_status st =
      rayo_graph_shapes(g, &s->model->pcwt, s->symbols, s->inputs, s->nodes, &scratch, msg);
  if (st != RAYO_OK) return st;
  st = rayo_plan_make(g, s->inputs, s->nodes, scratch, &s->book, &s->plan, msg);
  if (st != RAYO_OK) return st;
  s->memory = rayo_aligned_alloc(s->plan.arena_bytes);
  if (!s->memory) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  for (int32_t i = 0; i < g->input_count; i++)
    s->inputs[i].data = s->memory + s->plan.input_offset[i];
  for (int32_t i = 0; i < g->node_count; i++) s->nodes[i].data = s->memory + s->plan.node_offset[i];
  return RAYO_OK;
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

rayo_status rayo_session_bind(rayo_session *session, const char *name, int64_t value,
                              rayo_message *msg) {
  if (!session || !name) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  const rayo_graph *g = &session->model->graph;
  int32_t i = rayo_graph_symbol(g, name);
  if (i < 0) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "no size symbol %s", name);
  if (value < 1 || value > g->symbol_max[i]) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "%s = %lld, it must be 1 to %lld", name, (long long)value,
                     (long long)g->symbol_max[i]);
  }
  int64_t before = session->symbols[i];
  session->symbols[i] = value;
  size_t scratch;
  rayo_status st = rayo_graph_shapes(g, &session->model->pcwt, session->symbols, session->inputs,
                                     session->nodes, &scratch, msg);
  if (st != RAYO_OK) {
    session->symbols[i] = before;
    rayo_graph_shapes(g, &session->model->pcwt, session->symbols, session->inputs, session->nodes,
                      &scratch, NULL);
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
  view(&session->inputs[index], session->model->graph.inputs[index].name, out);
  return RAYO_OK;
}

rayo_status rayo_session_run(rayo_session *session, rayo_message *msg) {
  if (!session) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  const rayo_graph *g = &session->model->graph;
  const rayo_exec exec = {&session->kernels, session->memory + session->plan.scratch_offset};
  for (int32_t i = 0; i < g->node_count; i++) {
    const rayo_node *n = &g->nodes[i];
    rayo_tensor args[RAYO_MAX_INPUTS];
    rayo_graph_node_inputs(g, &session->model->pcwt, i, session->inputs, session->nodes, args);
    rayo_status st = n->op->run(n->params, args, &session->nodes[i], &exec);
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
  view(&session->nodes[o->node], o->name, out);
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
