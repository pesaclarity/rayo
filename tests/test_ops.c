/* The general ops (compare, masked_reduce) on small graphs, and ops of a
 * program's own (rayo/op.h). Each model is a PCWT file made here, with no
 * weights unless a test adds one. */
#include <stdlib.h>
#include <string.h>

#include "test.h"

static void put_u32(unsigned char *p, rayo_u32 v) {
  p[0] = (unsigned char)(v & 0xffu);
  p[1] = (unsigned char)(v >> 8 & 0xffu);
  p[2] = (unsigned char)(v >> 16 & 0xffu);
  p[3] = (unsigned char)(v >> 24 & 0xffu);
}

/* A PCWT file of `graph` (JSON) and, if `weight` is set, one float32 tensor
 * "w" of `count` values. free() it. */
static unsigned char *model_file(const char *graph, const float *weight, size_t count,
                                 size_t *size) {
  char header[4096];
  size_t h, pad, data = weight ? count * sizeof(float) : 0;
  unsigned char *file;
  if (weight) {
    sprintf(header,
            "{\"format_version\":1,\"model\":\"test-ops\",\"tensors\":[{\"name\":\"w\","
            "\"dtype\":\"float32\",\"shape\":[%lu],\"offset\":0,\"bytes\":%lu}],\"graph\":%s}",
            (unsigned long)count, (unsigned long)data, graph);
  } else {
    sprintf(header, "{\"format_version\":1,\"model\":\"test-ops\",\"tensors\":[],\"graph\":%s}",
            graph);
  }
  h = strlen(header);
  pad = (4 - (12 + h) % 4) % 4;
  *size = 12 + h + pad + data;
  file = malloc(*size);
  if (!file) return NULL;
  memcpy(file, "PCWT", 4);
  put_u32(file + 4, 1);
  put_u32(file + 8, (rayo_u32)(h + pad));
  memcpy(file + 12, header, h);
  memset(file + 12 + h, ' ', pad);
  if (weight) memcpy(file + 12 + h + pad, weight, data);
  return file;
}

static rayo_model *load(const char *graph, const rayo_model_options *options) {
  size_t size;
  unsigned char *file = model_file(graph, NULL, 0, &size);
  rayo_model *m = NULL;
  rayo_message msg;
  if (!file) return NULL;
  if (rayo_model_create_with(file, size, options, &m, &msg) != RAYO_OK) {
    fprintf(stderr, "  %s\n", msg.text);
  }
  free(file);
  return m;
}

/* Runs `m` with float input 0 (and 1 when `b`), and checks output 0. */
static void expect_run(rayo_model *m, const float *a, size_t na, const float *b, size_t nb,
                       const float *want, size_t nw) {
  rayo_session *s = NULL;
  rayo_tensor_view v;
  size_t i;
  EXPECT(m != NULL);
  if (!m) return;
  EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
  if (!s) return;
  rayo_session_input(s, 0, &v);
  EXPECT(v.bytes == na * sizeof(float));
  if (v.dtype == RAYO_DTYPE_F32) {
    memcpy(v.data, a, na * sizeof(float));
  } else {
    for (i = 0; i < na; i++) ((rayo_i32 *)v.data)[i] = (rayo_i32)a[i];
  }
  if (b) {
    rayo_session_input(s, 1, &v);
    EXPECT(v.bytes == nb * sizeof(float));
    memcpy(v.data, b, nb * sizeof(float));
  }
  EXPECT(rayo_session_run(s, NULL) == RAYO_OK);
  rayo_session_output(s, 0, &v);
  EXPECT(v.bytes == nw * sizeof(float));
  for (i = 0; i < nw && v.bytes == nw * sizeof(float); i++) {
    const float got =
        v.dtype == RAYO_DTYPE_F32 ? ((float *)v.data)[i] : (float)((rayo_i32 *)v.data)[i];
    EXPECT(got == want[i]);
  }
  rayo_session_destroy(s);
}

void test_compare(void) {
  static const char *const graph =
      "{\"version\":1,\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\",\"shape\":[5]}],"
      "\"nodes\":[{\"op\":\"compare\",\"inputs\":[\"x\"],\"output\":\"y\",\"comparison\":\"%s\","
      "\"value\":%s%s}],\"outputs\":[{\"name\":\"y\",\"node\":\"y\"}]}";
  static const float x[] = {-2.0f, 0.0f, 0.5f, 1.0f, 3.0f};
  static const struct {
    const char *how, *value;
    float want[5];
  } cases[] = {{"greater", "0", {0, 0, 1, 1, 1}}, {"greater_equal", "1", {0, 0, 0, 1, 1}},
               {"less", "0.5", {1, 1, 0, 0, 0}},  {"less_equal", "0", {1, 1, 0, 0, 0}},
               {"equal", "1", {0, 0, 0, 1, 0}},   {"not_equal", "0", {1, 0, 1, 1, 1}}};
  static const float ids[] = {3, 0, 7, 0, 1}, mask[] = {1, 0, 1, 0, 1};
  char g[1024];
  size_t i;
  rayo_model *m;
  for (i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    sprintf(g, graph, "float32", cases[i].how, cases[i].value, "");
    m = load(g, NULL);
    expect_run(m, x, 5, NULL, 0, cases[i].want, 5);
    rayo_model_destroy(m);
  }
  /* A padding mask over int32 ids, as int32. */
  sprintf(g, graph, "int32", "greater", "0", ",\"dtype\":\"int32\"");
  m = load(g, NULL);
  expect_run(m, ids, 5, NULL, 0, mask, 5);
  rayo_model_destroy(m);
  /* No comparison, or one rayo does not know: refused at load. */
  sprintf(g, graph, "float32", "sideways", "0", "");
  EXPECT(load(g, NULL) == NULL);
}

void test_masked_reduce(void) {
  static const char *const graph =
      "{\"version\":1,\"inputs\":[{\"name\":\"x\",\"dtype\":\"float32\",\"shape\":[3,2]},"
      "{\"name\":\"m\",\"dtype\":\"float32\",\"shape\":[%d]}],\"nodes\":[{\"op\":"
      "\"masked_reduce\",\"inputs\":[\"x\",\"m\"],\"output\":\"y\",%s}],"
      "\"outputs\":[{\"name\":\"y\",\"node\":\"y\"}]}";
  /* x is [[1, -4], [2, 5], [-3, 6]]. */
  static const float x[] = {1, -4, 2, 5, -3, 6};
  static const float keep02[] = {1, 0, 1}, none[] = {0, 0, 0}, keep1[] = {0, 1};
  static const float max02[] = {1, 6}, sum02[] = {-2, 2}, mean02[] = {-1, 1};
  static const float filled[] = {1, 6}, fill_all[] = {-9, -9}, empty[] = {7, 7};
  static const float axis1_max[] = {-4, 5, 6};
  char g[1024];
  rayo_model *m;
  sprintf(g, graph, 3, "\"reduction\":\"max\"");
  m = load(g, NULL);
  expect_run(m, x, 6, keep02, 3, max02, 2);
  rayo_model_destroy(m);
  sprintf(g, graph, 3, "\"reduction\":\"sum\"");
  m = load(g, NULL);
  expect_run(m, x, 6, keep02, 3, sum02, 2);
  rayo_model_destroy(m);
  sprintf(g, graph, 3, "\"reduction\":\"mean\"");
  m = load(g, NULL);
  expect_run(m, x, 6, keep02, 3, mean02, 2);
  rayo_model_destroy(m);
  /* "fill": dropped positions count as -9 (here they lose to the kept). */
  sprintf(g, graph, 3, "\"reduction\":\"max\",\"fill\":-9");
  m = load(g, NULL);
  expect_run(m, x, 6, keep02, 3, filled, 2);
  expect_run(m, x, 6, none, 3, fill_all, 2);
  rayo_model_destroy(m);
  /* Nothing kept and no fill: "empty". */
  sprintf(g, graph, 3, "\"reduction\":\"mean\",\"empty\":7");
  m = load(g, NULL);
  expect_run(m, x, 6, none, 3, empty, 2);
  rayo_model_destroy(m);
  /* Over the last axis, with a mask of its length. */
  sprintf(g, graph, 2, "\"reduction\":\"max\",\"axis\":-1");
  m = load(g, NULL);
  expect_run(m, x, 6, keep1, 2, axis1_max, 3);
  rayo_model_destroy(m);
  sprintf(g, graph, 3, "\"reduction\":\"median\"");
  EXPECT(load(g, NULL) == NULL);
}

/* A program's own op: y = x * scale + w, with "scale" an attribute and w
 * a weight the size of x. */
typedef struct axpy_params {
  float scale;
  const float *w;
  size_t n;
} axpy_params;

static int axpy_prepared;

static rayo_status axpy_prepare(void *user, const rayo_op_node *node, const void **params,
                                rayo_message *msg) {
  axpy_params *p = rayo_op_node_alloc(node, sizeof *p);
  rayo_tensor_view w;
  double scale;
  (void)msg;
  axpy_prepared += *(int *)user;
  if (!p || rayo_op_node_number(node, "scale", 1.0, &scale) != RAYO_OK ||
      rayo_op_node_input(node, 1, &w) != RAYO_OK || !w.data || w.dtype != RAYO_DTYPE_F32) {
    return RAYO_ERR_FORMAT;
  }
  p->scale = (float)scale;
  p->w = w.data;
  p->n = w.bytes / sizeof(float);
  *params = p;
  return RAYO_OK;
}

static rayo_status axpy_infer(const void *params, const rayo_tensor_view *in, int count,
                              rayo_tensor_view *out, size_t *scratch, rayo_message *msg) {
  const axpy_params *p = params;
  (void)scratch;
  (void)msg;
  if (count != 2 || in[0].dtype != RAYO_DTYPE_F32 || in[0].shape.rank != 1 ||
      (size_t)in[0].shape.dims[0] != p->n) {
    return RAYO_ERR_SHAPE;
  }
  out->dtype = RAYO_DTYPE_F32;
  out->shape = in[0].shape;
  return RAYO_OK;
}

static rayo_status axpy_run(const void *params, const rayo_tensor_view *in, int count,
                            rayo_tensor_view *out, void *scratch) {
  const axpy_params *p = params;
  const float *x = in[0].data;
  float *y = out->data;
  size_t i;
  (void)count;
  (void)scratch;
  for (i = 0; i < p->n; i++) y[i] = x[i] * p->scale + p->w[i];
  return RAYO_OK;
}

void test_custom_op(void) {
  static const char *const graph =
      "{\"version\":1,\"inputs\":[{\"name\":\"x\",\"dtype\":\"float32\",\"shape\":[3]}],"
      "\"nodes\":[{\"op\":\"axpy\",\"inputs\":[\"x\",\"w\"],\"output\":\"y\",\"scale\":2}],"
      "\"outputs\":[{\"name\":\"y\",\"node\":\"y\"}]}";
  static const float w[] = {0.5f, -1.0f, 4.0f}, x[] = {1.0f, 2.0f, -3.0f};
  static const float want[] = {2.5f, 3.0f, -2.0f};
  static int one = 1;
  rayo_custom_op axpy;
  const rayo_custom_op *ops[1];
  rayo_model_options o;
  rayo_model *m = NULL;
  rayo_message msg;
  unsigned char *file;
  size_t size;
  int i, found = 0;
  memset(&axpy, 0, sizeof axpy);
  axpy.api_version = RAYO_OP_API_VERSION;
  axpy.name = "axpy";
  axpy.min_inputs = 2;
  axpy.max_inputs = 2;
  axpy.user = &one;
  axpy.prepare = axpy_prepare;
  axpy.infer = axpy_infer;
  axpy.run = axpy_run;
  ops[0] = &axpy;
  memset(&o, 0, sizeof o);
  o.ops = ops;
  o.op_count = 1;
  file = model_file(graph, w, 3, &size);
  EXPECT(file != NULL);
  if (!file) return;
  /* Without the op the graph is refused, by name. */
  EXPECT(rayo_model_create(file, size, &m, &msg) == RAYO_ERR_UNSUPPORTED);
  EXPECT(strstr(msg.text, "axpy") != NULL);
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_OK);
  EXPECT(axpy_prepared == 1);
  expect_run(m, x, 3, NULL, 0, want, 3);
  rayo_model_destroy(m);
  /* An op for another API version, or named like a built-in one. */
  axpy.api_version = RAYO_OP_API_VERSION + 1;
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_ERR_UNSUPPORTED);
  axpy.api_version = RAYO_OP_API_VERSION;
  axpy.name = "dense";
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_ERR_INVALID_ARGUMENT);
  free(file);
  /* The built-in ops, listed. */
  for (i = 0; rayo_op_builtin(i); i++) found += strcmp(rayo_op_builtin(i), "masked_reduce") == 0;
  EXPECT(found == 1 && i > 10 && rayo_op_builtin(-1) == NULL);
}
