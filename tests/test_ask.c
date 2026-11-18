/* The real Ask models (tests/ask, from Pesa Clarity) against TFLite:
 * the reader on the 30 questions TFLite's outputs were recorded for (every
 * label, probabilities within 1e-4), and the writer fed each recorded reply
 * token by token (TFLite's next-token choice at every step, its top logit
 * within 0.2: the size of TFLite's own XNNPACK and built-in kernels'
 * disagreement, from int8 rounding). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "test.h"

/* A file of tests/ask (or RAYO_TEST_ASK). */
static unsigned char *ask_file(const char *name, size_t *size) {
  const char *dir = getenv("RAYO_TEST_ASK");
  char path[512];
  unsigned char *p;
  if (!dir) dir = "tests/ask";
  if (strlen(dir) + strlen(name) + 2 > sizeof path) return NULL;
  strcpy(path, dir);
  strcat(path, "/");
  strcat(path, name);
  p = test_read_file(path, size);
  if (!p) fprintf(stderr, "cannot read %s\n", path);
  return p;
}

static rayo_json *load_json(rayo_arena *a, const char *name, unsigned char **text) {
  size_t size = 0;
  rayo_json *root = NULL;
  *text = ask_file(name, &size);
  if (*text && rayo_json_parse((const char *)*text, size, (size_t)-1, a, &root, NULL) != RAYO_OK)
    root = NULL;
  return root;
}

static rayo_session *open_model(const char *name, rayo_model **m, unsigned char **file) {
  size_t size;
  rayo_session *s = NULL;
  rayo_message msg;
  *m = NULL;
  *file = ask_file(name, &size);
  if (!*file) return NULL;
  if (rayo_model_create(*file, size, m, &msg) != RAYO_OK ||
      rayo_session_create(*m, &s, &msg) != RAYO_OK) {
    fprintf(stderr, "%s: %s\n", name, msg.text);
    return NULL;
  }
  test_choose_kernels(s);
  return s;
}

static size_t top_f(const float *p, size_t n) {
  size_t i, best = 0;
  for (i = 1; i < n; i++) best = p[i] > p[best] ? i : best;
  return best;
}

static size_t top_j(const rayo_json *row) {
  size_t i, best = 0;
  for (i = 1; i < row->length; i++)
    best = row->items[i]->number > row->items[best]->number ? i : best;
  return best;
}

static double worst_of(const float *p, const rayo_json *want, double worst) {
  size_t i;
  for (i = 0; i < want->length; i++) {
    double d = fabs(p[i] - want->items[i]->number);
    worst = d > worst ? d : worst;
  }
  return worst;
}

/* One reading; returns whether every label is TFLite's, and widens *worst. */
static int read_one(rayo_session *s, const rayo_json *r, double *worst) {
  const rayo_json *input = rayo_json_get(r, "input");
  const rayo_json *want_i = rayo_json_get(r, "intent"), *want_s = rayo_json_get(r, "slots");
  rayo_tensor_view tokens, shapes, slots, intent;
  const float *ps;
  int ok;
  size_t i;
  rayo_session_input(s, 0, &tokens);
  rayo_session_input(s, 1, &shapes);
  for (i = 0; i < 96; i++) {
    ((rayo_i32 *)tokens.data)[i] = (rayo_i32)input->items[i]->number;
    ((rayo_i32 *)shapes.data)[i] = (rayo_i32)input->items[96 + i]->number;
  }
  if (rayo_session_run(s, NULL) != RAYO_OK) return 0;
  rayo_session_output(s, 0, &slots);
  rayo_session_output(s, 1, &intent);
  ok = top_f(intent.data, want_i->length) == top_j(want_i);
  *worst = worst_of(intent.data, want_i, *worst);
  ps = slots.data;
  for (i = 0; i < want_s->length; i++) {
    const rayo_json *row = want_s->items[i];
    ok &= top_f(ps + i * row->length, row->length) == top_j(row);
    *worst = worst_of(ps + i * row->length, row, *worst);
  }
  return ok;
}

static void test_reader(void) {
  rayo_arena a;
  unsigned char *text, *file;
  rayo_model *m;
  rayo_session *s;
  const rayo_json *readings;
  rayo_json *root;
  double worst = 0;
  int same = 0;
  size_t i;
  a.head = NULL;
  a.bytes = 0;
  root = load_json(&a, "ask_reader_decodes.json", &text);
  s = open_model("ask-reader.weights", &m, &file);
  EXPECT(root && s);
  if (root && s) {
    readings = rayo_json_get(root, "readings");
    for (i = 0; i < readings->length; i++) same += read_one(s, readings->items[i], &worst);
    printf("  reader (%s): labels %d of %lu as TFLite, probabilities within %.2g\n",
           rayo_session_kernels(s), same, (unsigned long)readings->length, worst);
    EXPECT(same == (int)readings->length);
    EXPECT(worst <= 1e-4);
  }
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  rayo_arena_free(&a);
  free(text);
  free(file);
}

static int step(rayo_session *s, const rayo_i32 *tokens, rayo_i32 past, rayo_i32 count) {
  rayo_tensor_view v;
  if (rayo_session_bind(s, "P", past, NULL) != RAYO_OK) return 0;
  if (rayo_session_bind(s, "T", count, NULL) != RAYO_OK) return 0;
  rayo_session_input(s, 0, &v);
  memcpy(v.data, tokens + past, (size_t)count * sizeof(rayo_i32));
  return rayo_session_run(s, NULL) == RAYO_OK;
}

/* One recorded reply token by token; counts the steps and those where the
 * next-token choice is TFLite's. */
static void write_one(rayo_session *s, const rayo_json *seq, long *steps, long *same,
                      double *worst) {
  const rayo_json *toks = rayo_json_get(seq, "tokens"), *argmax = rayo_json_get(seq, "argmax");
  const rayo_json *top = rayo_json_get(seq, "top");
  rayo_i32 tokens[64], n0, n, p;
  size_t j;
  rayo_json_as_int(rayo_json_get(seq, "prefix"), &n0);
  n = (rayo_i32)toks->length;
  for (p = 0; p < n && p < 64; p++) tokens[p] = (rayo_i32)toks->items[p]->number;
  rayo_session_reset(s);
  for (j = 0, p = n0; j < argmax->length; j++, p++) {
    rayo_tensor_view v;
    const float *logits;
    const size_t want = (size_t)argmax->items[j]->number;
    double d;
    if (!(j == 0 ? step(s, tokens, 0, n0) : step(s, tokens, p - 1, 1))) {
      EXPECT(0);
      return;
    }
    rayo_session_output(s, 0, &v);
    logits = v.data;
    d = fabs(logits[want] - top->items[j]->number);
    *worst = d > *worst ? d : *worst;
    *same += top_f(logits, v.bytes / 4) == want;
    (*steps)++;
  }
}

static void test_writer(void) {
  rayo_arena a;
  unsigned char *text, *file;
  rayo_model *m;
  rayo_session *s;
  rayo_json *root;
  long steps = 0, same = 0;
  double worst = 0;
  size_t i;
  a.head = NULL;
  a.bytes = 0;
  root = load_json(&a, "ask_writer_tflite.json", &text);
  s = open_model("ask-writer.weights", &m, &file);
  EXPECT(root && s);
  if (root && s) {
    const rayo_json *seqs = rayo_json_get(root, "sequences");
    for (i = 0; i < seqs->length; i++) write_one(s, seqs->items[i], &steps, &same, &worst);
    printf("  writer (%s): next token as TFLite at %ld of %ld steps, top logit within %.3f\n",
           rayo_session_kernels(s), same, steps, worst);
    EXPECT(same == steps && steps > 0);
    EXPECT(worst < 0.2);
  }
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  rayo_arena_free(&a);
  free(text);
  free(file);
}

void test_ask(void) {
  test_reader();
  test_writer();
}
