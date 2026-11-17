/* rayo/op.h: ops of your own. rayo runs a model's graph with its built-in
 * ops (docs/OPS.md); a model may also name ops a program gives rayo when it
 * loads the model, with rayo_model_create_with.
 *
 * An op is three functions, as rayo's own are:
 * - prepare, once when the model loads: read the node's attributes and its
 *   weights, check them, and set the op's parameters (memory from
 *   rayo_op_node_alloc lives as long as the model);
 * - infer, when a session is planned and at every bind: the output's dtype
 *   and shape for the inputs' current shapes, and the scratch bytes a run
 *   needs;
 * - run, for every run: compute the output from the inputs. It must not
 *   allocate, and may run on several sessions, on several threads, at once:
 *   keep anything that changes in `scratch`, never in the parameters. */
#ifndef RAYO_OP_H
#define RAYO_OP_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this interface. An op says which it was written for, and
 * rayo refuses ops for a version it does not know. */
#define RAYO_OP_API_VERSION 1

/* A node of the graph being loaded, for prepare. */
typedef struct rayo_op_node rayo_op_node;

typedef struct rayo_custom_op {
  int api_version; /* RAYO_OP_API_VERSION */
  const char *name;
  int min_inputs;
  int max_inputs; /* at most RAYO_OP_MAX_INPUTS (rayo/config.h) */
  void *user;     /* given back to prepare */
  rayo_status (*prepare)(void *user, const rayo_op_node *node, const void **params,
                         rayo_message *msg);
  /* `output`: set its dtype and shape. `*scratch` starts at 0. */
  rayo_status (*infer)(const void *params, const rayo_tensor_view *inputs, int input_count,
                       rayo_tensor_view *output, size_t *scratch, rayo_message *msg);
  /* `scratch`: the bytes infer asked for, RAYO_ALIGN-aligned. */
  rayo_status (*run)(const void *params, const rayo_tensor_view *inputs, int input_count,
                     rayo_tensor_view *output, void *scratch);
} rayo_custom_op;

/* What prepare can read of its node. */
RAYO_API const char *rayo_op_node_output(const rayo_op_node *node);
RAYO_API int rayo_op_node_input_count(const rayo_op_node *node);

/* Input `index`: a weight's name, dtype, shape and data; for an activation
 * or a graph input, `data` is NULL (its shape comes to infer). */
RAYO_API rayo_status rayo_op_node_input(const rayo_op_node *node, int index, rayo_tensor_view *out);

/* Attribute `key` of the node: a whole number, a number or a string (owned
 * by the model); `fallback` when it is absent. RAYO_ERR_FORMAT when it is
 * there but of another kind. */
RAYO_API rayo_status rayo_op_node_int(const rayo_op_node *node, const char *key, long fallback,
                                      long *out);
RAYO_API rayo_status rayo_op_node_number(const rayo_op_node *node, const char *key, double fallback,
                                         double *out);
RAYO_API rayo_status rayo_op_node_string(const rayo_op_node *node, const char *key,
                                         const char *fallback, const char **out);

/* `bytes` of zeroed memory, RAYO_ALIGN-aligned, freed with the model; NULL
 * when out of memory. */
RAYO_API void *rayo_op_node_alloc(const rayo_op_node *node, size_t bytes);

/* The ops rayo has built in, by index from 0, for listing them; NULL past
 * the last. */
RAYO_API const char *rayo_op_builtin(int index);

#ifdef __cplusplus
}
#endif

#endif
