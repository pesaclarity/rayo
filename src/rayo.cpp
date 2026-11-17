// rayo: the graph interpreter on ggml. See rayo.h and docs/FORMAT.md.

#include "rayo.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#define RAYO_VERSION "0.1.0"
#define RAYO_FORMAT_VERSION 1
#define RAYO_MAX_NODES 4096

namespace {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  throw Error(buf);
}

void put_error(char *err, size_t err_len, const char *msg) {
  if (err && err_len) {
    snprintf(err, err_len, "%s", msg);
  }
}

// ggml writes its own messages to stderr; keep them out of the app's log.
void quiet_log(enum ggml_log_level, const char *, void *) {}

void init_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    ggml_log_set(quiet_log, nullptr);
    ggml_cpu_init();
  });
}

// ---- The graph text (docs/FORMAT.md) --------------------------------------

struct Decl {  // input or state
  std::string name;
  ggml_type type;
  std::vector<std::string> dims;  // innermost first; ints or size expressions
};

struct Node {
  std::string out;
  std::string op;
  std::vector<std::string> args;
  std::map<std::string, std::string> attrs;
  int line;
};

struct Graph {
  std::vector<Decl> inputs;
  std::vector<Decl> states;
  std::vector<std::pair<std::string, std::string>> outputs;  // name, node
  std::vector<Node> nodes;
};

std::vector<std::string> split_ws(const std::string &s) {
  std::istringstream in(s);
  std::vector<std::string> out;
  std::string w;
  while (in >> w) out.push_back(w);
  return out;
}

ggml_type parse_type(const std::string &s, int line) {
  if (s == "f32") return GGML_TYPE_F32;
  if (s == "i32") return GGML_TYPE_I32;
  fail("line %d: unknown type %s", line, s.c_str());
}

Graph parse_graph(const std::string &text) {
  Graph g;
  std::istringstream in(text);
  std::string raw;
  int line = 0;
  while (std::getline(in, raw)) {
    line++;
    auto hash = raw.find('#');
    if (hash != std::string::npos) raw.resize(hash);
    auto w = split_ws(raw);
    if (w.empty()) continue;
    if (w[0] == "input" || w[0] == "state") {
      if (w.size() < 4) fail("line %d: %s needs a name, a type and dims", line, w[0].c_str());
      Decl d{w[1], parse_type(w[2], line), {w.begin() + 3, w.end()}};
      if (d.dims.size() > 4) fail("line %d: more than 4 dims", line);
      (w[0] == "input" ? g.inputs : g.states).push_back(d);
    } else if (w[0] == "output") {
      if (w.size() != 3) fail("line %d: output needs a name and a node", line);
      g.outputs.emplace_back(w[1], w[2]);
    } else {
      if (w.size() < 3 || w[1] != "=") fail("line %d: expected `name = op args...`", line);
      Node n{w[0], w[2], {}, {}, line};
      for (size_t i = 3; i < w.size(); i++) {
        auto eq = w[i].find('=');
        if (eq == std::string::npos) {
          n.args.push_back(w[i]);
        } else {
          n.attrs[w[i].substr(0, eq)] = w[i].substr(eq + 1);
        }
      }
      static const char *known[] = {
          "get_rows", "range", "mask_pos", "concat", "dense", "conv1d", "add", "mul",
          "scale", "relu", "gelu", "gelu_erf", "softmax", "layer_norm", "masked_max",
          "masked_mean", "cols", "write", "attention"};
      if (std::none_of(std::begin(known), std::end(known),
                       [&](const char *k) { return n.op == k; })) {
        fail("line %d: unknown op %s (a newer rayo may know it)", line, n.op.c_str());
      }
      g.nodes.push_back(n);
    }
  }
  if (g.outputs.empty()) fail("the graph has no outputs");
  return g;
}

// ---- JSON for rayo_model_info --------------------------------------------

void json_string(std::string &out, const char *s) {
  out += '"';
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += (char)c;
        }
    }
  }
  out += '"';
}

void json_number(std::string &out, double v) {
  if (!std::isfinite(v)) {
    out += "null";
    return;
  }
  char buf[64];
  snprintf(buf, sizeof buf, "%.17g", v);
  out += buf;
}

void json_value(std::string &out, const gguf_context *g, int64_t i, gguf_type t,
                const void *data, size_t j) {
  switch (t) {
    case GGUF_TYPE_UINT8: json_number(out, ((const uint8_t *)data)[j]); break;
    case GGUF_TYPE_INT8: json_number(out, ((const int8_t *)data)[j]); break;
    case GGUF_TYPE_UINT16: json_number(out, ((const uint16_t *)data)[j]); break;
    case GGUF_TYPE_INT16: json_number(out, ((const int16_t *)data)[j]); break;
    case GGUF_TYPE_UINT32: json_number(out, ((const uint32_t *)data)[j]); break;
    case GGUF_TYPE_INT32: json_number(out, ((const int32_t *)data)[j]); break;
    case GGUF_TYPE_UINT64: json_number(out, (double)((const uint64_t *)data)[j]); break;
    case GGUF_TYPE_INT64: json_number(out, (double)((const int64_t *)data)[j]); break;
    case GGUF_TYPE_FLOAT32: json_number(out, ((const float *)data)[j]); break;
    case GGUF_TYPE_FLOAT64: json_number(out, ((const double *)data)[j]); break;
    case GGUF_TYPE_BOOL: out += ((const int8_t *)data)[j] ? "true" : "false"; break;
    case GGUF_TYPE_STRING: json_string(out, gguf_get_arr_str(g, i, j)); break;
    default: out += "null";
  }
}

std::string info_json(const gguf_context *g, const Graph &graph) {
  std::string out = "{\"metadata\":{";
  int64_t n = gguf_get_n_kv(g);
  bool first = true;
  for (int64_t i = 0; i < n; i++) {
    const char *key = gguf_get_key(g, i);
    if (strcmp(key, "rayo.graph") == 0) continue;
    if (!first) out += ',';
    first = false;
    json_string(out, key);
    out += ':';
    gguf_type t = gguf_get_kv_type(g, i);
    if (t == GGUF_TYPE_STRING) {
      json_string(out, gguf_get_val_str(g, i));
    } else if (t == GGUF_TYPE_ARRAY) {
      gguf_type at = gguf_get_arr_type(g, i);
      size_t len = gguf_get_arr_n(g, i);
      const void *data = at == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(g, i);
      out += '[';
      for (size_t j = 0; j < len; j++) {
        if (j) out += ',';
        json_value(out, g, i, at, data, j);
      }
      out += ']';
    } else {
      json_value(out, g, i, t, gguf_get_val_data(g, i), 0);
    }
  }
  out += "},";
  auto decls = [&](const char *name, const std::vector<Decl> &ds) {
    json_string(out, name);
    out += ":[";
    for (size_t i = 0; i < ds.size(); i++) {
      if (i) out += ',';
      out += "{\"name\":";
      json_string(out, ds[i].name.c_str());
      out += ",\"dtype\":";
      json_string(out, ds[i].type == GGML_TYPE_I32 ? "i32" : "f32");
      out += ",\"shape\":[";
      for (size_t j = 0; j < ds[i].dims.size(); j++) {
        if (j) out += ',';
        json_string(out, ds[i].dims[j].c_str());
      }
      out += "]}";
    }
    out += "],";
  };
  decls("inputs", graph.inputs);
  decls("states", graph.states);
  out += "\"outputs\":[";
  for (size_t i = 0; i < graph.outputs.size(); i++) {
    if (i) out += ',';
    json_string(out, graph.outputs[i].first.c_str());
  }
  out += "]}";
  return out;
}

// ---- Custom ops -------------------------------------------------------------

// 1 where the id is above 0 (a real token), else 0: i32 [T] -> f32 [T].
void op_mask_pos(ggml_tensor *dst, int ith, int, void *) {
  if (ith != 0) return;
  const ggml_tensor *ids = dst->src[0];
  const int32_t *x = (const int32_t *)ids->data;
  float *y = (float *)dst->data;
  for (int64_t i = 0; i < ggml_nelements(ids); i++) y[i] = x[i] > 0 ? 1.0f : 0.0f;
}

// Over the positions where the mask is 1: max or mean of each channel.
// x [C, T], mask [T] -> [C]. userdata: PoolParams.
struct PoolParams {
  bool mean;
  float empty;  // the value when no position is real
};

void op_masked_pool(ggml_tensor *dst, int ith, int nth, void *userdata) {
  const PoolParams *pp = (const PoolParams *)userdata;
  const bool mean = pp->mean;
  const ggml_tensor *x = dst->src[0];
  const ggml_tensor *m = dst->src[1];
  const int64_t c = x->ne[0], t = x->ne[1];
  const float *mask = (const float *)m->data;
  float *y = (float *)dst->data;
  int64_t n = 0;
  for (int64_t i = 0; i < t; i++) n += mask[i] > 0;
  for (int64_t j = ith; j < c; j += nth) {
    float best = -INFINITY;
    double sum = 0;
    for (int64_t i = 0; i < t; i++) {
      if (!(mask[i] > 0)) continue;
      float v = *(const float *)((const char *)x->data + i * x->nb[1] + j * x->nb[0]);
      best = std::max(best, v);
      sum += v;
    }
    y[j] = n == 0 ? pp->empty : mean ? (float)(sum / (double)n) : best;
  }
}

// 0 where key j may be seen by query i (j <= S - T + i), else -inf: [S, T].
void op_causal_mask(ggml_tensor *dst, int ith, int, void *) {
  if (ith != 0) return;
  const int64_t sl = dst->ne[0], t = dst->ne[1];
  float *y = (float *)dst->data;
  for (int64_t i = 0; i < t; i++) {
    for (int64_t j = 0; j < sl; j++) y[i * sl + j] = j <= sl - t + i ? 0.0f : -INFINITY;
  }
}

// The k taps around each position, side by side: x [C, T] -> [k * C, T],
// tap j of position i being position i + (j - (k - 1) / 2) * d, or zeros
// outside the sequence. userdata: ConvParams.
struct ConvParams {
  int k;
  int d;
};

void op_unfold(ggml_tensor *dst, int ith, int nth, void *userdata) {
  const ConvParams *p = (const ConvParams *)userdata;
  const ggml_tensor *x = dst->src[0];
  const int64_t c = x->ne[0], t = x->ne[1];
  const int64_t half = (int64_t)(p->k - 1) / 2 * p->d;
  for (int64_t i = ith; i < t; i += nth) {
    float *row = (float *)((char *)dst->data + i * dst->nb[1]);
    for (int j = 0; j < p->k; j++) {
      const int64_t at = i + (int64_t)j * p->d - half;
      float *to = row + j * c;
      if (at < 0 || at >= t) {
        memset(to, 0, c * sizeof(float));
      } else {
        memcpy(to, (const char *)x->data + at * x->nb[1], c * sizeof(float));
      }
    }
  }
}

// start, start + 1, ...: i32 [n]. userdata points at the start.
void op_range(ggml_tensor *dst, int ith, int, void *userdata) {
  if (ith != 0) return;
  const int64_t start = *(const int64_t *)userdata;
  int32_t *y = (int32_t *)dst->data;
  for (int64_t i = 0; i < dst->ne[0]; i++) y[i] = (int32_t)(start + i);
}

}  // namespace

// ---- Model -------------------------------------------------------------------

struct rayo_model {
  gguf_context *gguf = nullptr;
  ggml_context *weights = nullptr;
  Graph graph;
  std::string info;

  ~rayo_model() {
    if (gguf) gguf_free(gguf);
    if (weights) ggml_free(weights);
  }
};

// ---- Session -------------------------------------------------------------------

struct rayo_session {
  rayo_model *model = nullptr;
  int n_threads = 1;
  ggml_threadpool *pool = nullptr;
  ggml_context *state_ctx = nullptr;
  std::map<std::string, ggml_tensor *> states;
  std::map<std::string, int64_t> symbols;
  std::map<std::string, std::vector<uint8_t>> inputs;
  ggml_gallocr_t galloc = nullptr;
  ggml_context *ctx = nullptr;  // the last run's graph
  std::map<std::string, ggml_tensor *> outputs;
  std::vector<uint8_t> work;
  std::deque<int64_t> scalars;  // custom ops' parameters for the current run
  std::deque<PoolParams> pools;
  std::deque<ConvParams> convs;

  ~rayo_session() {
    if (ctx) ggml_free(ctx);
    if (galloc) ggml_gallocr_free(galloc);
    if (state_ctx) ggml_free(state_ctx);
    if (pool) ggml_threadpool_free(pool);
  }

  // A size expression: ints and symbols joined by + and -.
  int64_t eval(const std::string &expr, int line) const {
    int64_t total = 0;
    int sign = 1;
    size_t i = 0;
    if (expr.empty()) fail("line %d: empty size", line);
    while (i < expr.size()) {
      size_t j = i;
      while (j < expr.size() && expr[j] != '+' && expr[j] != '-') j++;
      std::string term = expr.substr(i, j - i);
      if (term.empty()) fail("line %d: bad size %s", line, expr.c_str());
      int64_t v;
      if (isdigit((unsigned char)term[0])) {
        v = std::stoll(term);
      } else {
        auto it = symbols.find(term);
        if (it == symbols.end()) fail("size symbol %s is not bound", term.c_str());
        v = it->second;
      }
      total += sign * v;
      if (j < expr.size()) sign = expr[j] == '+' ? 1 : -1;
      i = j + 1;
      if (j == expr.size()) break;
      if (i == expr.size()) fail("line %d: bad size %s", line, expr.c_str());
    }
    return total;
  }

  void build_and_run();
};

namespace {

struct Builder {
  rayo_session *s;
  ggml_context *ctx;
  ggml_cgraph *gf;
  std::map<std::string, ggml_tensor *> env;
  const Node *node = nullptr;

  [[noreturn]] void bad(const char *what) {
    fail("line %d (%s): %s", node->line, node->op.c_str(), what);
  }

  ggml_tensor *arg(size_t i) {
    if (i >= node->args.size()) bad("missing argument");
    const std::string &name = node->args[i];
    auto it = env.find(name);
    if (it != env.end()) return it->second;
    ggml_tensor *w = ggml_get_tensor(s->model->weights, name.c_str());
    if (!w) fail("line %d: unknown tensor %s", node->line, name.c_str());
    env[name] = w;
    return w;
  }

  bool has(const char *key) const { return node->attrs.count(key) > 0; }

  int64_t iattr(const char *key, int64_t def) {
    auto it = node->attrs.find(key);
    return it == node->attrs.end() ? def : s->eval(it->second, node->line);
  }

  float fattr(const char *key, float def) {
    auto it = node->attrs.find(key);
    if (it == node->attrs.end()) return def;
    char *end = nullptr;
    float v = strtof(it->second.c_str(), &end);
    if (!end || *end) bad("bad number");
    return v;
  }

  std::string sattr(const char *key, const char *def) {
    auto it = node->attrs.find(key);
    return it == node->attrs.end() ? def : it->second;
  }

  void need_f32(ggml_tensor *t) {
    if (t->type != GGML_TYPE_F32) bad("expected a float32 tensor");
  }

  ggml_tensor *act(ggml_tensor *x) {
    std::string a = sattr("act", "none");
    if (a == "none") return x;
    if (a == "relu") return ggml_relu(ctx, x);
    if (a == "gelu") return ggml_gelu(ctx, x);
    if (a == "gelu_erf") return ggml_gelu_erf(ctx, x);
    bad("unknown act");
  }

  // y = w x (+ b): w [in, out], x [in, ...] -> [out, ...].
  ggml_tensor *matmul_bias(ggml_tensor *w, ggml_tensor *x, ggml_tensor *b) {
    if (w->ne[0] != x->ne[0]) bad("weight and input sizes differ");
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    if (b) {
      if (b->ne[0] != w->ne[1]) bad("bias size");
      y = ggml_add(ctx, y, b);
    }
    return y;
  }

  ggml_tensor *same_shape_or_broadcast(ggml_tensor *a, ggml_tensor *b) {
    if (!ggml_can_repeat(b, a)) bad("shapes do not broadcast");
    return b;
  }

  ggml_tensor *run(const Node &n) {
    node = &n;
    const std::string &op = n.op;
    if (op == "get_rows") {
      ggml_tensor *table = arg(0), *ids = arg(1);
      if (ids->type != GGML_TYPE_I32) bad("ids must be int32");
      ggml_tensor *y = ggml_get_rows(ctx, table, ids);
      return y->type == GGML_TYPE_F32 ? y : ggml_cast(ctx, y, GGML_TYPE_F32);
    }
    if (op == "mask_pos") {
      ggml_tensor *ids = arg(0);
      if (ids->type != GGML_TYPE_I32) bad("ids must be int32");
      ggml_tensor *args[] = {ids};
      return ggml_custom_4d(ctx, GGML_TYPE_F32, ids->ne[0], 1, 1, 1, args, 1,
                            op_mask_pos, 1, nullptr);
    }
    if (op == "masked_max" || op == "masked_mean") {
      ggml_tensor *x = arg(0), *m = arg(1);
      need_f32(x);
      need_f32(m);
      if (ggml_nelements(m) != x->ne[1] || x->ne[2] != 1 || x->ne[3] != 1) bad("mask size");
      ggml_tensor *args[] = {x, m};
      s->pools.push_back({op == "masked_mean", fattr("empty", 0.0f)});
      return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], 1, 1, 1, args, 2,
                            op_masked_pool, GGML_N_TASKS_MAX, &s->pools.back());
    }
    if (op == "range") {
      s->scalars.push_back(iattr("start", 0));
      int64_t len = iattr("len", -1);
      if (len < 1) bad("len must be at least 1");
      return ggml_custom_4d(ctx, GGML_TYPE_I32, len, 1, 1, 1, nullptr, 0, op_range, 1,
                            &s->scalars.back());
    }
    if (op == "concat") {
      ggml_tensor *a = arg(0), *b = arg(1);
      int dim = (int)iattr("dim", 0);
      if (dim < 0 || dim > 3) bad("dim");
      for (int d = 0; d < 4; d++) {
        if (d != dim && a->ne[d] != b->ne[d]) bad("shapes differ outside dim");
      }
      return ggml_concat(ctx, a, b, dim);
    }
    if (op == "dense") {
      ggml_tensor *x = arg(0), *w = arg(1);
      ggml_tensor *b = n.args.size() > 2 ? arg(2) : nullptr;
      return act(matmul_bias(w, x, b));
    }
    if (op == "conv1d") {
      // Same-padded 1-D convolution over positions. x [C, T]; w [k * C, out]
      // with element (j, c) at j * C + c (tap-major, as Keras and TFLite keep
      // it); b [out] -> [out, T].
      ggml_tensor *x = arg(0), *w = arg(1);
      ggml_tensor *b = n.args.size() > 2 ? arg(2) : nullptr;
      int k = (int)iattr("k", 0), d = (int)iattr("d", 1);
      if (k < 1 || d < 1 || (k - 1) * d % 2) bad("k and d must give a centred window");
      need_f32(x);
      if (x->ne[2] != 1 || x->ne[3] != 1) bad("conv1d takes [C, T]");
      const int64_t c = x->ne[0];
      if (w->ne[0] != c * k) bad("kernel size");
      if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
      s->convs.push_back({k, d});
      ggml_tensor *args[] = {x};
      ggml_tensor *cols = ggml_custom_4d(ctx, GGML_TYPE_F32, c * k, x->ne[1], 1, 1, args, 1,
                                         op_unfold, GGML_N_TASKS_MAX, &s->convs.back());
      return act(matmul_bias(w, cols, b));
    }
    if (op == "add") {
      ggml_tensor *a = arg(0), *b = arg(1);
      return act(ggml_add(ctx, a, same_shape_or_broadcast(a, b)));
    }
    if (op == "mul") {
      ggml_tensor *a = arg(0), *b = arg(1);
      return ggml_mul(ctx, a, same_shape_or_broadcast(a, b));
    }
    if (op == "relu") return ggml_relu(ctx, arg(0));
    if (op == "gelu") return ggml_gelu(ctx, arg(0));
    if (op == "gelu_erf") return ggml_gelu_erf(ctx, arg(0));
    if (op == "scale") return ggml_scale(ctx, arg(0), fattr("s", 1.0f));
    if (op == "softmax") return ggml_soft_max(ctx, arg(0));
    if (op == "layer_norm") {
      // Over the innermost dimension, then * g + b.
      ggml_tensor *x = arg(0);
      ggml_tensor *y = ggml_norm(ctx, x, fattr("eps", 1e-5f));
      if (n.args.size() > 1) y = ggml_mul(ctx, y, same_shape_or_broadcast(y, arg(1)));
      if (n.args.size() > 2) y = ggml_add(ctx, y, same_shape_or_broadcast(y, arg(2)));
      return y;
    }
    if (op == "cols") {
      // Columns [start, start + len) of a [C, T] tensor (positions).
      ggml_tensor *x = arg(0);
      int64_t start = iattr("start", 0), len = iattr("len", x->ne[1] - start);
      if (start < 0 || len < 1 || start + len > x->ne[1] || x->ne[2] != 1) bad("columns out of range");
      return ggml_view_2d(ctx, x, x->ne[0], len, x->nb[1], start * x->nb[1]);
    }
    if (op == "write") {
      // Copies x [C, T] into state columns [at, at + T) and gives the state's
      // columns [0, at + T). Runs before every later node.
      ggml_tensor *st = arg(0), *x = arg(1);
      int64_t at = iattr("at", 0);
      if (st->ne[0] != x->ne[0] || x->ne[2] != 1 || at < 0 || at + x->ne[1] > st->ne[1])
        bad("does not fit the state");
      ggml_tensor *dst = ggml_view_2d(ctx, st, st->ne[0], x->ne[1], st->nb[1], at * st->nb[1]);
      ggml_build_forward_expand(gf, ggml_cpy(ctx, x, dst));
      return ggml_view_2d(ctx, st, st->ne[0], at + x->ne[1], st->nb[1], 0);
    }
    if (op == "attention") {
      // Multi-head scaled dot-product attention. q [D, T]; k, v [D, S] with
      // the T queries at positions S - T .. S - 1; causal=1 hides later keys.
      ggml_tensor *q = arg(0), *k = arg(1), *v = arg(2);
      int64_t heads = iattr("heads", 1);
      const int64_t dm = q->ne[0], t = q->ne[1], sl = k->ne[1];
      if (heads < 1 || dm % heads || k->ne[0] != dm || v->ne[0] != dm || v->ne[1] != sl || sl < t)
        bad("q, k and v sizes");
      const int64_t hd = dm / heads;
      // [hd, heads, T] -> [hd, T, heads]
      ggml_tensor *qh = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q), hd, heads, t), 0, 2, 1, 3);
      ggml_tensor *kh = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, k), hd, heads, sl), 0, 2, 1, 3);
      ggml_tensor *kq = ggml_mul_mat(ctx, kh, qh);  // [S, T, heads]
      float scale = fattr("scale", 1.0f / sqrtf((float)hd));
      ggml_tensor *mask = nullptr;
      if (iattr("causal", 0)) {
        mask = ggml_custom_4d(ctx, GGML_TYPE_F32, sl, t, 1, 1, nullptr, 0, op_causal_mask, 1,
                              nullptr);
      }
      kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
      // v as [S, hd, heads] so the product sums over S.
      ggml_tensor *vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, v), hd, heads, sl), 1, 2, 0, 3));
      ggml_tensor *o = ggml_mul_mat(ctx, vh, kq);           // [hd, T, heads]
      o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [hd, heads, T]
      return ggml_reshape_2d(ctx, o, dm, t);
    }
    fail("line %d: unknown op %s", n.line, op.c_str());
  }
};

}  // namespace

void rayo_session::build_and_run() {
  const Graph &g = model->graph;
  if (ctx) {
    ggml_free(ctx);
    ctx = nullptr;
  }
  outputs.clear();
  scalars.clear();
  pools.clear();
  convs.clear();
  ggml_init_params params = {
      ggml_tensor_overhead() * RAYO_MAX_NODES + ggml_graph_overhead_custom(RAYO_MAX_NODES, false),
      nullptr, true};
  ctx = ggml_init(params);
  if (!ctx) fail("out of memory");
  Builder b{this, ctx, ggml_new_graph_custom(ctx, RAYO_MAX_NODES, false), {}};
  for (auto &st : states) b.env[st.first] = st.second;
  std::vector<std::pair<ggml_tensor *, const std::vector<uint8_t> *>> feeds;
  for (const Decl &d : g.inputs) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < d.dims.size(); i++) {
      ne[i] = eval(d.dims[i], 0);
      if (ne[i] < 1) fail("input %s has size %lld", d.name.c_str(), (long long)ne[i]);
    }
    ggml_tensor *t = ggml_new_tensor(ctx, d.type, (int)d.dims.size(), ne);
    ggml_set_name(t, d.name.c_str());
    ggml_set_input(t);
    auto it = inputs.find(d.name);
    if (it == inputs.end()) fail("input %s is not set", d.name.c_str());
    if (it->second.size() != ggml_nbytes(t)) {
      fail("input %s has %zu bytes, the graph wants %zu", d.name.c_str(), it->second.size(),
           ggml_nbytes(t));
    }
    b.env[d.name] = t;
    feeds.emplace_back(t, &it->second);
  }
  for (const Node &n : g.nodes) {
    ggml_tensor *t = b.run(n);
    b.env[n.out] = t;
  }
  for (auto &o : g.outputs) {
    auto it = b.env.find(o.second);
    if (it == b.env.end()) fail("output %s: no node %s", o.first.c_str(), o.second.c_str());
    ggml_tensor *t = it->second;
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
    ggml_set_output(t);
    ggml_build_forward_expand(b.gf, t);
    outputs[o.first] = t;
  }
  if (!ggml_gallocr_alloc_graph(galloc, b.gf)) fail("could not allocate the graph");
  for (auto &f : feeds) memcpy(f.first->data, f.second->data(), f.second->size());
  ggml_cplan plan = ggml_graph_plan(b.gf, n_threads, pool);
  if (plan.work_size > work.size()) work.resize(plan.work_size);
  plan.work_data = work.data();
  if (ggml_graph_compute(b.gf, &plan) != GGML_STATUS_SUCCESS) fail("compute failed");
}

// ---- SHA-256 (FIPS 180-4) -----------------------------------------------------

namespace {

const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_block(uint32_t h[8], const uint8_t *p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++) {
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 |
           p[4 * i + 3];
  }
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
    uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

}  // namespace

// ---- C API ----------------------------------------------------------------------

extern "C" {

const char *rayo_version(void) {
  static std::string v = std::string(RAYO_VERSION " (ggml ") + ggml_version() + " " + ggml_commit() + ")";
  return v.c_str();
}

void rayo_sha256(const void *data, size_t size, uint8_t out[32]) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint8_t *p = (const uint8_t *)data;
  size_t left = size;
  while (left >= 64) {
    sha256_block(h, p);
    p += 64;
    left -= 64;
  }
  uint8_t tail[128] = {0};
  memcpy(tail, p, left);
  tail[left] = 0x80;
  size_t n = left < 56 ? 64 : 128;
  uint64_t bits = (uint64_t)size * 8;
  for (int i = 0; i < 8; i++) tail[n - 1 - i] = (uint8_t)(bits >> (8 * i));
  sha256_block(h, tail);
  if (n == 128) sha256_block(h, tail + 64);
  for (int i = 0; i < 8; i++) {
    out[4 * i] = (uint8_t)(h[i] >> 24);
    out[4 * i + 1] = (uint8_t)(h[i] >> 16);
    out[4 * i + 2] = (uint8_t)(h[i] >> 8);
    out[4 * i + 3] = (uint8_t)h[i];
  }
}

rayo_model *rayo_model_load(const void *data, size_t size, char *err, size_t err_len) {
  init_once();
  auto m = std::make_unique<rayo_model>();
  try {
    gguf_init_params p = {false, &m->weights};
    m->gguf = gguf_init_from_buffer(data, size, p);
    if (!m->gguf) fail("not a GGUF file, or a damaged one");
    int64_t ver = gguf_find_key(m->gguf, "rayo.format_version");
    if (ver < 0 || gguf_get_kv_type(m->gguf, ver) != GGUF_TYPE_UINT32) {
      fail("no rayo.format_version: not a rayo model");
    }
    if (gguf_get_val_u32(m->gguf, ver) != RAYO_FORMAT_VERSION) {
      fail("format version %u, this runtime reads %d", gguf_get_val_u32(m->gguf, ver),
           RAYO_FORMAT_VERSION);
    }
    int64_t gk = gguf_find_key(m->gguf, "rayo.graph");
    if (gk < 0 || gguf_get_kv_type(m->gguf, gk) != GGUF_TYPE_STRING) fail("no rayo.graph");
    m->graph = parse_graph(gguf_get_val_str(m->gguf, gk));
    m->info = info_json(m->gguf, m->graph);
    return m.release();
  } catch (const std::exception &e) {
    put_error(err, err_len, e.what());
    return nullptr;
  }
}

void rayo_model_free(rayo_model *model) { delete model; }

const char *rayo_model_info(const rayo_model *model) { return model->info.c_str(); }

rayo_session *rayo_session_new(rayo_model *model, int n_threads, char *err, size_t err_len) {
  auto s = std::make_unique<rayo_session>();
  try {
    s->model = model;
    s->n_threads = std::max(1, std::min(n_threads, 64));
    if (s->n_threads > 1) {
      ggml_threadpool_params tp = ggml_threadpool_params_default(s->n_threads);
      s->pool = ggml_threadpool_new(&tp);
      if (!s->pool) fail("could not start %d threads", s->n_threads);
    }
    s->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    const auto &states = model->graph.states;
    size_t bytes = ggml_tensor_overhead() * (states.size() + 1);
    for (const Decl &d : states) {
      size_t n = ggml_type_size(d.type);
      for (auto &dim : d.dims) n *= (size_t)s->eval(dim, 0);
      bytes += n + 64;
    }
    ggml_init_params p = {bytes, nullptr, false};
    s->state_ctx = ggml_init(p);
    if (!s->state_ctx) fail("out of memory");
    for (const Decl &d : states) {
      int64_t ne[4] = {1, 1, 1, 1};
      for (size_t i = 0; i < d.dims.size(); i++) ne[i] = s->eval(d.dims[i], 0);
      ggml_tensor *t = ggml_new_tensor(s->state_ctx, d.type, (int)d.dims.size(), ne);
      s->states[d.name] = t;
    }
    rayo_session_reset(s.get());
    return s.release();
  } catch (const std::exception &e) {
    put_error(err, err_len, e.what());
    return nullptr;
  }
}

void rayo_session_free(rayo_session *session) { delete session; }

int rayo_session_bind(rayo_session *s, const char *name, int64_t value, char *err,
                       size_t err_len) {
  if (!name || !isalpha((unsigned char)name[0])) {
    put_error(err, err_len, "a size symbol starts with a letter");
    return 1;
  }
  s->symbols[name] = value;
  return 0;
}

int rayo_session_set_input(rayo_session *s, const char *name, const void *data,
                            size_t n_bytes, char *err, size_t err_len) {
  bool known = false;
  for (auto &d : s->model->graph.inputs) known |= d.name == name;
  if (!known) {
    std::string msg = std::string("no input ") + name;
    put_error(err, err_len, msg.c_str());
    return 1;
  }
  auto &buf = s->inputs[name];
  buf.assign((const uint8_t *)data, (const uint8_t *)data + n_bytes);
  return 0;
}

int rayo_session_run(rayo_session *s, char *err, size_t err_len) {
  try {
    s->build_and_run();
    return 0;
  } catch (const std::exception &e) {
    put_error(err, err_len, e.what());
    return 1;
  }
}

const float *rayo_session_output(rayo_session *s, const char *name, int64_t ne[4], char *err,
                                  size_t err_len) {
  auto it = s->outputs.find(name);
  if (it == s->outputs.end()) {
    std::string msg = std::string("no output ") + name + " from the last run";
    put_error(err, err_len, msg.c_str());
    return nullptr;
  }
  for (int i = 0; i < 4; i++) ne[i] = it->second->ne[i];
  return (const float *)it->second->data;
}

void rayo_session_reset(rayo_session *s) {
  for (auto &st : s->states) memset(st.second->data, 0, ggml_nbytes(st.second));
}

}  // extern "C"
