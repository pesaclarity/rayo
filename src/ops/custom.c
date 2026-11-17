/* Ops a program gives rayo (rayo/op.h): one adapter between rayo's own op
 * interface and the public one, which sees tensors as rayo_tensor_view. */
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"
#include "ops/attrs.h"
#include "ops/ops.h"

struct rayo_op_node {
  const rayo_op_spec *spec;
  rayo_arena *arena;
};

typedef struct custom_params {
  const rayo_custom_op *op;
  const void *params; /* the op's own, from its prepare */
  int input_count;    /* the node's */
} custom_params;

rayo_status rayo_custom_op_check(const rayo_custom_op *op, rayo_message *msg) {
  if (!op || !op->name || !op->prepare || !op->infer || !op->run) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "an op needs a name and three functions");
  }
  if (op->api_version != RAYO_OP_API_VERSION) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "op %s: API version %d, rayo has %d", op->name,
                     op->api_version, RAYO_OP_API_VERSION);
  }
  if (rayo_op_find(op->name)) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "op %s: rayo has an op of that name",
                     op->name);
  }
  if (op->min_inputs < 0 || op->max_inputs < op->min_inputs ||
      op->max_inputs > RAYO_OP_MAX_INPUTS) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "op %s: takes 0 to %d inputs", op->name,
                     RAYO_OP_MAX_INPUTS);
  }
  return RAYO_OK;
}

static void view_of(const rayo_tensor *t, rayo_tensor_view *v) {
  memset(v, 0, sizeof *v);
  v->name = t->weight ? t->weight->name : NULL;
  v->dtype = t->dtype;
  v->shape = t->shape;
  v->data = t->data;
  v->bytes = t->data ? rayo_tensor_bytes(t) : 0;
}

static rayo_status prepare(const rayo_op_spec *spec, rayo_arena *arena, const void **params,
                           rayo_message *msg) {
  custom_params *p = rayo_arena_alloc(arena, sizeof *p);
  rayo_op_node node;
  rayo_status s;
  if (!p) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  node.spec = spec;
  node.arena = arena;
  p->op = spec->custom;
  p->params = NULL;
  p->input_count = spec->input_count;
  s = p->op->prepare(p->op->user, &node, &p->params, msg);
  *params = p;
  return s;
}

static rayo_status infer(const void *params, const rayo_tensor *in, const rayo_i32 *symbols,
                         rayo_tensor *out, size_t *scratch, rayo_message *msg) {
  const custom_params *p = params;
  rayo_tensor_view views[RAYO_OP_MAX_INPUTS], y;
  int i;
  rayo_status s;
  (void)symbols;
  for (i = 0; i < p->input_count; i++) view_of(&in[i], &views[i]);
  memset(&y, 0, sizeof y);
  *scratch = 0;
  s = p->op->infer(p->params, views, p->input_count, &y, scratch, msg);
  if (s != RAYO_OK) return s;
  if ((y.dtype != RAYO_DTYPE_F32 && y.dtype != RAYO_DTYPE_I32 && y.dtype != RAYO_DTYPE_I8) ||
      y.shape.rank < 0 || y.shape.rank > RAYO_MAX_RANK) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "op %s: an output of no dtype or rank", p->op->name);
  }
  for (i = 0; i < y.shape.rank; i++) {
    if (y.shape.dims[i] < 0) {
      return rayo_fail(msg, RAYO_ERR_SHAPE, "op %s: a negative size", p->op->name);
    }
  }
  out->dtype = y.dtype;
  out->shape = y.shape;
  return RAYO_OK;
}

static rayo_status run(const void *params, const rayo_tensor *in, rayo_tensor *out,
                       const rayo_exec *exec) {
  const custom_params *p = params;
  rayo_tensor_view views[RAYO_OP_MAX_INPUTS], y;
  int i;
  for (i = 0; i < p->input_count; i++) view_of(&in[i], &views[i]);
  view_of(out, &y);
  return p->op->run(p->params, views, p->input_count, &y, exec->scratch);
}

const rayo_op rayo_op_custom = {"(custom)", 0, RAYO_OP_MAX_INPUTS, prepare, infer, run, RAYO_FALSE};

const char *rayo_op_node_output(const rayo_op_node *node) {
  return node ? node->spec->output : "";
}

int rayo_op_node_input_count(const rayo_op_node *node) {
  return node ? node->spec->input_count : 0;
}

rayo_status rayo_op_node_input(const rayo_op_node *node, int index, rayo_tensor_view *out) {
  if (!node || !out || index < 0 || index >= node->spec->input_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  view_of(&node->spec->inputs[index], out);
  return RAYO_OK;
}

rayo_status rayo_op_node_int(const rayo_op_node *node, const char *key, long fallback, long *out) {
  rayo_i32 v;
  rayo_status s;
  if (!node || !key || !out) return RAYO_ERR_INVALID_ARGUMENT;
  s = rayo_attr_int(node->spec, key, 0, -RAYO_I32_MAX, RAYO_I32_MAX, &v, NULL);
  if (s != RAYO_OK) return s;
  *out = rayo_attr_has(node->spec, key) ? (long)v : fallback;
  return RAYO_OK;
}

rayo_status rayo_op_node_number(const rayo_op_node *node, const char *key, double fallback,
                                double *out) {
  if (!node || !key || !out) return RAYO_ERR_INVALID_ARGUMENT;
  return rayo_attr_double(node->spec, key, fallback, out, NULL);
}

rayo_status rayo_op_node_string(const rayo_op_node *node, const char *key, const char *fallback,
                                const char **out) {
  if (!node || !key || !out) return RAYO_ERR_INVALID_ARGUMENT;
  return rayo_attr_string(node->spec, key, fallback, out, NULL);
}

void *rayo_op_node_alloc(const rayo_op_node *node, size_t bytes) {
  return node ? rayo_arena_alloc_aligned(node->arena, bytes ? bytes : 1) : NULL;
}
