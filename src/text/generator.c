/* A language model's text, a token at a time (rayo/generate.h). */
#include <stdlib.h>
#include <string.h>

#include "core/json.h"
#include "core/message.h"
#include "model.h"
#include "rayo/generate.h"
#include "rayo/session.h"
#include "session.h"

#define RECENT 64
#define MAX_STOPS 8

typedef enum turn_piece { FIRST, USER_END, NEXT, SYSTEM, AFTER_SYSTEM, PIECES } turn_piece;

static const char *const piece_names[PIECES] = {"first", "user_end", "next", "system",
                                                "after_system"};

struct rayo_generator {
  const rayo_model *model;
  const rayo_tokenizer *tokenizer;
  rayo_session *session;
  rayo_sampler *sampler;
  rayo_generator_options o;
  rayo_i32 context, chunk, vocab;
  rayo_i32 pos;
  rayo_i32 pending;   /* sampled, not yet run; -1: none */
  rayo_bool ready;    /* logits of the last position are there */
  rayo_bool ended;    /* the reply ended on a stop token */
  rayo_i32 generated; /* tokens of this reply */
  rayo_i32 stops[MAX_STOPS];
  int stop_count;
  rayo_i32 bos;
  rayo_bool add_bos;
  char *turns[PIECES]; /* NULL when the template has none */
  rayo_i32 recent[RECENT];
  size_t recent_count;
  rayo_i32 *ids; /* room for `context` */
};

static char *copy(const char *s, size_t n) {
  char *p = malloc(n + 1);
  if (p) {
    memcpy(p, s, n);
    p[n] = 0;
  }
  return p;
}

static rayo_status read_metadata(rayo_generator *g, rayo_message *msg) {
  const rayo_json *meta = rayo_json_get(g->model->pcwt.header, "metadata");
  const rayo_json *stops = rayo_json_get(meta, "eos_token_ids");
  const rayo_json *turns = rayo_json_get(meta, "chat_turns");
  const rayo_json *add = rayo_json_get(meta, "add_bos_token");
  size_t i;
  g->bos = -1;
  if (!rayo_json_as_int(rayo_json_get(meta, "bos_token_id"), &g->bos)) g->bos = -1;
  g->add_bos = add && add->kind == RAYO_JSON_BOOL && add->boolean;
  for (i = 0; stops && stops->kind == RAYO_JSON_ARRAY && i < stops->length; i++) {
    rayo_i32 id;
    if (g->stop_count < MAX_STOPS && rayo_json_as_int(stops->items[i], &id)) {
      g->stops[g->stop_count++] = id;
    }
  }
  for (i = 0; turns && i < PIECES; i++) {
    const rayo_json *v = rayo_json_get(turns, piece_names[i]);
    if (v && v->kind == RAYO_JSON_STRING) {
      g->turns[i] = copy(v->string, v->length);
      if (!g->turns[i]) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
    }
  }
  return RAYO_OK;
}

rayo_status rayo_generator_create(const rayo_model *model, const rayo_tokenizer *tokenizer,
                                  const rayo_generator_options *options, rayo_generator **out,
                                  rayo_message *msg) {
  rayo_generator *g;
  rayo_session_options so;
  rayo_symbol_limit limits[2];
  rayo_tensor_view in, logits;
  rayo_i32 t, p, most;
  rayo_status s;
  if (!model || !tokenizer || !out) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  }
  *out = NULL;
  t = rayo_graph_symbol(&model->graph, "T");
  p = rayo_graph_symbol(&model->graph, "P");
  if (t < 0 || p < 0 || model->graph.input_count != 1 ||
      model->graph.inputs[0].dtype != RAYO_DTYPE_I32 || model->graph.output_count < 1) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "not a decoder: a generator needs symbols T and P, int32 tokens in and "
                     "logits out");
  }
  g = calloc(1, sizeof *g);
  if (!g) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  g->model = model;
  g->tokenizer = tokenizer;
  if (options) g->o = *options;
  /* The model's longest, lowered to the options' (or the defaults). */
  most = model->graph.symbol_max[t];
  g->chunk = g->o.chunk > 0 ? g->o.chunk : RAYO_GENERATOR_CHUNK;
  if (g->chunk > most) g->chunk = most;
  most = model->graph.symbol_max[p] + 1;
  g->context = g->o.context > 0 ? g->o.context : RAYO_GENERATOR_CONTEXT;
  if (g->context > most) g->context = most;
  if (g->chunk > g->context) g->chunk = g->context;
  g->pending = -1;
  memset(&so, 0, sizeof so);
  so.threads = g->o.threads;
  so.cache = g->o.cache;
  /* The cache holds `context` rows; memory is planned for a full chunk at
   * its end. */
  so.state_rows = g->context;
  limits[0].name = "T";
  limits[0].max = g->chunk;
  limits[0].plan = g->chunk;
  limits[1].name = "P";
  limits[1].max = g->context - 1;
  limits[1].plan = g->context - g->chunk;
  so.limits = limits;
  so.limit_count = 2;
  s = rayo_session_create_with(model, &so, &g->session, msg);
  if (s == RAYO_OK) s = read_metadata(g, msg);
  if (s == RAYO_OK) {
    rayo_session_output(g->session, 0, &logits);
    rayo_session_input(g->session, 0, &in);
    g->vocab = logits.shape.rank == 1 ? logits.shape.dims[0] : 0;
    if (g->vocab < 1) {
      s = rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                    "the model's output is not one position's logits (convert it with "
                    "--positions last)");
    }
  }
  if (s == RAYO_OK) s = rayo_sampler_create(g->vocab, &g->o.sampling, &g->sampler, msg);
  if (s == RAYO_OK) {
    g->ids = malloc((size_t)g->context * sizeof *g->ids);
    if (!g->ids) s = rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  if (s != RAYO_OK) {
    rayo_generator_destroy(g);
    return s;
  }
  *out = g;
  return RAYO_OK;
}

void rayo_generator_destroy(rayo_generator *generator) {
  int i;
  if (!generator) return;
  rayo_session_destroy(generator->session);
  rayo_sampler_destroy(generator->sampler);
  for (i = 0; i < PIECES; i++) free(generator->turns[i]);
  free(generator->ids);
  free(generator);
}

/* Runs ids through the model, a chunk at a time. */
static rayo_status feed(rayo_generator *g, const rayo_i32 *ids, rayo_i32 n, rayo_message *msg) {
  rayo_i32 at = 0;
  if (g->pos + n > g->context) {
    return rayo_fail(msg, RAYO_ERR_SHAPE, "the context is full: %ld of %ld tokens",
                     (long)(g->pos + n), (long)g->context);
  }
  while (at < n) {
    const rayo_i32 c = n - at < g->chunk ? n - at : g->chunk;
    rayo_tensor_view in;
    rayo_status s = rayo_session_bind(g->session, "P", g->pos, msg);
    if (s == RAYO_OK) s = rayo_session_bind(g->session, "T", c, msg);
    if (s != RAYO_OK) return s;
    rayo_session_input(g->session, 0, &in);
    memcpy(in.data, ids + at, (size_t)c * sizeof *ids);
    s = rayo_session_run(g->session, msg);
    if (s != RAYO_OK) return s;
    g->pos += c;
    at += c;
  }
  g->ready = n > 0 || g->ready;
  return RAYO_OK;
}

/* Encodes text into g->ids after `first` ids already there, and runs them. */
static rayo_status feed_text(rayo_generator *g, const char *text, size_t size, size_t first,
                             rayo_message *msg) {
  size_t n = 0;
  const size_t room = (size_t)(g->context - g->pos);
  rayo_status s;
  if (first > room) return rayo_fail(msg, RAYO_ERR_SHAPE, "the context is full");
  s = rayo_tokenizer_encode(g->tokenizer, text, size, g->ids + first, room - first, &n, msg);
  if (s == RAYO_ERR_SHAPE) {
    return rayo_fail(msg, s, "the context is full: %lu more tokens, room for %lu",
                     (unsigned long)(first + n), (unsigned long)room);
  }
  if (s != RAYO_OK) return s;
  return feed(g, g->ids, (rayo_i32)(first + n), msg);
}

/* A new reply starts: the token sampled last (if the reply did not end on
 * a stop token) goes into the cache first. */
static rayo_status start(rayo_generator *g, rayo_message *msg) {
  rayo_status s = RAYO_OK;
  if (g->pending >= 0) {
    const rayo_i32 t = g->pending;
    g->pending = -1;
    s = feed(g, &t, 1, msg);
  }
  g->ended = RAYO_FALSE;
  g->generated = 0;
  return s;
}

rayo_status rayo_generator_prompt(rayo_generator *generator, const char *text, size_t size,
                                  rayo_message *msg) {
  rayo_generator *g = generator;
  size_t first = 0;
  rayo_status s;
  if (!g || (!text && size)) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  s = start(g, msg);
  if (s != RAYO_OK) return s;
  if (g->pos == 0 && g->add_bos && g->bos >= 0) g->ids[first++] = g->bos;
  return feed_text(g, text, size, first, msg);
}

rayo_status rayo_generator_chat(rayo_generator *generator, const char *system, const char *user,
                                rayo_message *msg) {
  rayo_generator *g = generator;
  const char *before, *between = NULL;
  char *text;
  size_t n;
  rayo_status s;
  if (!g || !user) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  if (!g->turns[FIRST] || !g->turns[USER_END] || !g->turns[NEXT]) {
    return rayo_fail(msg, RAYO_ERR_UNSUPPORTED,
                     "the model has no chat template (convert it with jinja2 installed)");
  }
  s = start(g, msg);
  if (s != RAYO_OK) return s;
  before = g->pos == 0 ? g->turns[FIRST] : g->turns[NEXT];
  if (system && g->pos == 0) {
    if (!g->turns[SYSTEM] || !g->turns[AFTER_SYSTEM]) {
      return rayo_fail(msg, RAYO_ERR_UNSUPPORTED, "the chat template takes no system text");
    }
    before = g->turns[SYSTEM];
    between = g->turns[AFTER_SYSTEM];
  }
  n = strlen(before) + (between ? strlen(system) + strlen(between) : 0) + strlen(user) +
      strlen(g->turns[USER_END]);
  text = malloc(n + 1);
  if (!text) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  strcpy(text, before);
  if (between) {
    strcat(text, system);
    strcat(text, between);
  }
  strcat(text, user);
  strcat(text, g->turns[USER_END]);
  s = feed_text(g, text, n, 0, msg);
  free(text);
  return s;
}

rayo_status rayo_generator_next(rayo_generator *generator, rayo_i32 *token, rayo_message *msg) {
  rayo_generator *g = generator;
  rayo_tensor_view logits;
  rayo_i32 id;
  int i;
  if (!g || !token) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  *token = -1;
  if (g->ended || (g->o.max_tokens > 0 && g->generated >= g->o.max_tokens)) return RAYO_OK;
  if (g->pending >= 0) {
    const rayo_i32 t = g->pending;
    rayo_status s;
    if (g->pos >= g->context) return RAYO_OK; /* full: the reply ends */
    g->pending = -1;
    s = feed(g, &t, 1, msg);
    if (s != RAYO_OK) return s;
  }
  if (!g->ready) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "no prompt yet");
  rayo_session_output(g->session, 0, &logits);
  id = rayo_sampler_next(g->sampler, (const float *)logits.data, g->recent, g->recent_count);
  for (i = 0; i < g->stop_count; i++) {
    if (id == g->stops[i]) {
      g->ended = RAYO_TRUE;
      return RAYO_OK;
    }
  }
  g->pending = id;
  g->generated++;
  if (g->recent_count < RECENT) {
    g->recent[g->recent_count++] = id;
  } else {
    memmove(g->recent, g->recent + 1, (RECENT - 1) * sizeof *g->recent);
    g->recent[RECENT - 1] = id;
  }
  *token = id;
  return RAYO_OK;
}

void rayo_generator_reset(rayo_generator *generator) {
  if (!generator) return;
  rayo_session_reset(generator->session);
  generator->pos = 0;
  generator->pending = -1;
  generator->ready = RAYO_FALSE;
  generator->ended = RAYO_FALSE;
  generator->generated = 0;
  generator->recent_count = 0;
}

rayo_i32 rayo_generator_position(const rayo_generator *generator) {
  return generator ? generator->pos : 0;
}

rayo_i32 rayo_generator_context(const rayo_generator *generator) {
  return generator ? generator->context : 0;
}

rayo_bool rayo_generator_can_chat(const rayo_generator *generator) {
  return generator && generator->turns[FIRST] && generator->turns[USER_END] &&
         generator->turns[NEXT];
}

int rayo_generator_threads(const rayo_generator *generator) {
  return generator ? rayo_session_threads(generator->session) : 1;
}

const char *rayo_generator_kernels(const rayo_generator *generator) {
  return generator ? rayo_session_kernels(generator->session) : "";
}
