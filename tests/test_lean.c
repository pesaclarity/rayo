/* What keeps a language model's own memory small, on the tiny Llama-style
 * model (tool/make_llama_test_model.py): the tokeniser's table read from
 * the model file (the same bytes and ids as its tokenizer.json), the int8
 * key/value cache against float32's, a session's context below the
 * model's, prompts in chunks of any size, a reset that gives pages back,
 * and the process memory report. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/json.h"
#include "test.h"

static void path_of(const char *name, char *out, size_t size) {
  const char *dir = getenv("RAYO_TEST_MODELS");
  if (!dir) dir = "tests/models";
  out[0] = 0;
  if (strlen(dir) + strlen(name) + 2 > size) return;
  strcpy(out, dir);
  strcat(out, "/");
  strcat(out, name);
}

static rayo_model *load(const char *name) {
  char path[512];
  rayo_model *m = NULL;
  rayo_message msg;
  path_of(name, path, sizeof path);
  if (rayo_model_load(path, NULL, &m, &msg) != RAYO_OK) fprintf(stderr, "  %s\n", msg.text);
  EXPECT(m != NULL);
  return m;
}

static const char *const texts[] = {
    "the cat sat on the mat, hello there! caf\xc3\xa9 \xf0\x9f\x99\x82",
    "<|im_start|>user\nhow are you today?<|im_end|>\n<|im_start|>assistant\n",
    "  spaces   and\ttabs\n\nnew lines 12345 zzz", ""};

/* The table in the model file is the one rayo builds from the
 * tokenizer.json, byte for byte, and encodes the same ids. */
static void table_from_model(void) {
  char path[512];
  rayo_model *m = load("llama-q4h.weights"), *old = load("llama-q4.weights");
  rayo_tokenizer *mapped = NULL, *json = NULL, *bad = NULL;
  const void *a = NULL, *b = NULL;
  size_t na = 0, nb = 0, i, k;
  unsigned char *copy;
  rayo_message msg;
  if (!m || !old) {
    rayo_model_destroy(m);
    rayo_model_destroy(old);
    return;
  }
  EXPECT(rayo_tokenizer_from_model(old, &bad, NULL) == RAYO_ERR_UNSUPPORTED && !bad);
  EXPECT(rayo_tokenizer_from_model(m, &mapped, &msg) == RAYO_OK);
  path_of("llama.tokenizer.json", path, sizeof path);
  EXPECT(rayo_tokenizer_load(path, &json, &msg) == RAYO_OK);
  if (!mapped || !json) {
    rayo_tokenizer_destroy(mapped);
    rayo_tokenizer_destroy(json);
    rayo_model_destroy(m);
    rayo_model_destroy(old);
    return;
  }
  EXPECT(rayo_tokenizer_table(mapped, &a, &na) == RAYO_OK);
  EXPECT(rayo_tokenizer_table(json, &b, &nb) == RAYO_OK);
  EXPECT(na == nb && na > 0 && memcmp(a, b, na) == 0);
  printf("  tokeniser table: %lu bytes, the converter's and rayo's the same: %s\n",
         (unsigned long)na, na == nb && memcmp(a, b, na) == 0 ? "yes" : "NO");
  EXPECT(rayo_tokenizer_vocab_size(mapped) == rayo_tokenizer_vocab_size(json));
  for (i = 0; i < sizeof texts / sizeof texts[0]; i++) {
    rayo_i32 x[256], y[256];
    size_t cx = 0, cy = 0;
    EXPECT(rayo_tokenizer_encode(mapped, texts[i], strlen(texts[i]), x, 256, &cx, NULL) == RAYO_OK);
    EXPECT(rayo_tokenizer_encode(json, texts[i], strlen(texts[i]), y, 256, &cy, NULL) == RAYO_OK);
    EXPECT(cx == cy && memcmp(x, y, cx * sizeof *x) == 0);
  }
  EXPECT(rayo_tokenizer_find(mapped, "<|im_end|>") == rayo_tokenizer_find(json, "<|im_end|>"));
  for (i = 0; i < (size_t)rayo_tokenizer_vocab_size(json); i++) {
    size_t sx, sy;
    const char *px = rayo_tokenizer_piece(mapped, (rayo_i32)i, &sx);
    const char *py = rayo_tokenizer_piece(json, (rayo_i32)i, &sy);
    EXPECT((px == NULL) == (py == NULL) && sx == sy && (!px || memcmp(px, py, sx) == 0));
  }
  /* A damaged table is refused, not misread: each header word, and bytes
   * through the records, changed in turn. */
  copy = malloc(na);
  for (k = 0; copy && k < 2 * (na / 4); k++) {
    rayo_tokenizer *t = NULL;
    size_t at = k < 32 ? 4 * k : 4 * ((k * 7919) % (na / 4));
    memcpy(copy, a, na);
    copy[at] ^= (unsigned char)(k < 32 ? 0x40 : 0xA5);
    copy[at + 3] ^= (unsigned char)(k % 3 == 0 ? 0x80 : 0);
    if (rayo_tokenizer_create_table(copy, na, &t, NULL) == RAYO_OK) {
      rayo_i32 ids[256];
      size_t n = 0;
      /* Accepted: it must still encode within bounds. */
      for (i = 0; i < sizeof texts / sizeof texts[0]; i++) {
        rayo_tokenizer_encode(t, texts[i], strlen(texts[i]), ids, 256, &n, NULL);
      }
      rayo_tokenizer_destroy(t);
    }
  }
  free(copy);
  rayo_tokenizer_destroy(mapped);
  rayo_tokenizer_destroy(json);
  rayo_model_destroy(m);
  rayo_model_destroy(old);
}

static void run_chunks(rayo_session *s, const rayo_i32 *ids, rayo_i32 n, rayo_i32 chunk) {
  rayo_i32 at = 0;
  while (at < n) {
    const rayo_i32 c = n - at < chunk ? n - at : chunk;
    rayo_tensor_view in;
    EXPECT(rayo_session_bind(s, "P", at, NULL) == RAYO_OK);
    EXPECT(rayo_session_bind(s, "T", c, NULL) == RAYO_OK);
    rayo_session_input(s, 0, &in);
    memcpy(in.data, ids + at, (size_t)c * sizeof *ids);
    EXPECT(rayo_session_run(s, NULL) == RAYO_OK);
    at += c;
  }
}

/* A session as a generator makes it: `context` rows, chunks of `chunk`. */
static rayo_session *session(const rayo_model *m, int cache, rayo_i32 context, rayo_i32 chunk) {
  rayo_session_options o;
  rayo_symbol_limit lim[2];
  rayo_session *s = NULL;
  rayo_message msg;
  memset(&o, 0, sizeof o);
  o.cache = cache;
  o.state_rows = context;
  lim[0].name = "T";
  lim[0].max = lim[0].plan = chunk;
  lim[1].name = "P";
  lim[1].max = context - 1;
  lim[1].plan = context - chunk;
  o.limits = lim;
  o.limit_count = 2;
  if (rayo_session_create_with(m, &o, &s, &msg) != RAYO_OK) fprintf(stderr, "  %s\n", msg.text);
  EXPECT(s != NULL);
  if (s) test_choose_kernels(s);
  return s;
}

static double worst_of(const float *a, const float *b, size_t n, double *top) {
  double worst = 0;
  size_t i;
  *top = 0;
  for (i = 0; i < n; i++) {
    const double d = fabs((double)a[i] - b[i]);
    worst = d > worst ? d : worst;
    *top = fabs(a[i]) > *top ? fabs(a[i]) : *top;
  }
  return worst;
}

/* The int8 cache against float32's, chunks against one token at a time,
 * and a context below the model's. */
static void cache_and_chunks(void) {
  rayo_model *m = load("llama-q4h.weights");
  rayo_session *f = NULL, *q = NULL, *one = NULL, *small = NULL, *bad = NULL;
  rayo_session_options o;
  rayo_symbol_limit lim;
  rayo_memory_info mf, mq;
  rayo_tensor_view out;
  rayo_i32 ids[100];
  float *lf = NULL, *lq = NULL, *l1 = NULL;
  size_t vocab = 0, i;
  int agree = 0, steps = 0;
  double worst_q = 0;
  if (!m) return;
  for (i = 0; i < 100; i++) ids[i] = (rayo_i32)((i * 37 + 11) % 290);
  f = session(m, RAYO_CACHE_FLOAT32, 128, 16);
  q = session(m, RAYO_CACHE_INT8, 128, 16);
  one = session(m, RAYO_CACHE_FLOAT32, 128, 1);
  small = session(m, RAYO_CACHE_FLOAT32, 40, 8);
  if (!f || !q || !one || !small) goto done;
  rayo_session_output(f, 0, &out);
  vocab = (size_t)out.shape.dims[0];
  lf = malloc(vocab * sizeof *lf);
  lq = malloc(vocab * sizeof *lq);
  l1 = malloc(vocab * sizeof *l1);
  if (!lf || !lq || !l1) goto done;
  /* The int8 cache: a quarter of float32's, plus a scale per head and row. */
  rayo_session_memory(f, &mf);
  rayo_session_memory(q, &mq);
  EXPECT(mq.states * 3 < mf.states && mq.states > 0);
  /* Position by position: the int8 cache's logits within 2% of the largest,
   * and the same most likely token nearly everywhere. */
  for (i = 1; i <= 100; i += 9) {
    double t, w;
    size_t j, bf = 0, bq = 0;
    rayo_session_reset(f);
    rayo_session_reset(q);
    run_chunks(f, ids, (rayo_i32)i, 16);
    run_chunks(q, ids, (rayo_i32)i, 16);
    rayo_session_output(f, 0, &out);
    memcpy(lf, out.data, vocab * sizeof *lf);
    rayo_session_output(q, 0, &out);
    memcpy(lq, out.data, vocab * sizeof *lq);
    /* Two statements: the call sets t, and C does not order a call's
     * side effects before the other operand of /. */
    w = worst_of(lf, lq, vocab, &t);
    w /= t > 0 ? t : 1;
    worst_q = w > worst_q ? w : worst_q;
    for (j = 1; j < vocab; j++) {
      if (lf[j] > lf[bf]) bf = j;
      if (lq[j] > lq[bq]) bq = j;
    }
    agree += bf == bq;
    steps++;
  }
  printf(
      "  int8 cache: logits within %.2g of the largest of float32's; top token the same at "
      "%d of %d\n",
      worst_q, agree, steps);
  EXPECT(worst_q < 0.02);
  EXPECT(agree >= steps - 1);
  /* Chunks of 16 and of 1 give the same logits. */
  rayo_session_reset(f);
  rayo_session_reset(one);
  run_chunks(f, ids, 100, 16);
  run_chunks(one, ids, 100, 1);
  rayo_session_output(f, 0, &out);
  memcpy(lf, out.data, vocab * sizeof *lf);
  rayo_session_output(one, 0, &out);
  memcpy(l1, out.data, vocab * sizeof *l1);
  {
    double t;
    const double w = worst_of(lf, l1, vocab, &t);
    printf("  chunks of 16 and of 1: logits within %.2g (largest %.2f)\n", w, t);
    EXPECT(w <= 1e-5 * (1.0 + t));
  }
  /* A context of 40: P + T beyond it is refused; the arena is smaller. */
  rayo_session_memory(small, &mq);
  EXPECT(mq.arena < mf.arena);
  EXPECT(mq.states * 2 < mf.states);
  EXPECT(rayo_session_bind(small, "T", 9, NULL) == RAYO_ERR_SHAPE);
  EXPECT(rayo_session_bind(small, "P", 35, NULL) == RAYO_ERR_SHAPE); /* 35 + 8 > 40 */
  EXPECT(rayo_session_bind(small, "T", 1, NULL) == RAYO_OK);
  EXPECT(rayo_session_bind(small, "P", 39, NULL) == RAYO_OK);
  EXPECT(rayo_session_bind(small, "P", 40, NULL) == RAYO_ERR_SHAPE);
  /* Limits beyond the graph's, or of no symbol, are refused. */
  memset(&o, 0, sizeof o);
  lim.name = "T";
  lim.max = 100000;
  lim.plan = 1;
  o.limits = &lim;
  o.limit_count = 1;
  EXPECT(rayo_session_create_with(m, &o, &bad, NULL) == RAYO_ERR_INVALID_ARGUMENT && !bad);
  lim.name = "nope";
  lim.max = 1;
  EXPECT(rayo_session_create_with(m, &o, &bad, NULL) == RAYO_ERR_INVALID_ARGUMENT && !bad);
  o.limit_count = 0;
  o.cache = 7;
  EXPECT(rayo_session_create_with(m, &o, &bad, NULL) == RAYO_ERR_INVALID_ARGUMENT && !bad);
done:
  free(lf);
  free(lq);
  free(l1);
  rayo_session_destroy(f);
  rayo_session_destroy(q);
  rayo_session_destroy(one);
  rayo_session_destroy(small);
  rayo_model_destroy(m);
}

/* The generator with its own context, chunk and cache; its tokeniser from
 * the model file; no allocation per token. */
static void generator(void) {
  rayo_model *m = load("llama-q3.weights");
  rayo_tokenizer *t = NULL;
  rayo_generator *g = NULL;
  rayo_generator_options o;
  rayo_message msg;
  rayo_i32 token;
  long before;
  int n = 0;
  if (!m) return;
  EXPECT(rayo_tokenizer_from_model(m, &t, &msg) == RAYO_OK);
  memset(&o, 0, sizeof o);
  o.context = 48;
  o.chunk = 5;
  o.cache = RAYO_CACHE_INT8;
  o.max_tokens = 6;
  o.threads = 2;
  EXPECT(t && rayo_generator_create(m, t, &o, &g, &msg) == RAYO_OK);
  if (!g) {
    rayo_tokenizer_destroy(t);
    rayo_model_destroy(m);
    return;
  }
  EXPECT(rayo_generator_context(g) == 48);
  EXPECT(rayo_generator_chat(g, NULL, "hello there, how are you today?", &msg) == RAYO_OK);
  before = test_allocations();
  while (rayo_generator_next(g, &token, NULL) == RAYO_OK && token >= 0) n++;
  EXPECT(before < 0 || test_allocations() == before);
  EXPECT(n > 0 && n <= 6);
  /* The context is full well before 20 prompts of this. */
  rayo_generator_reset(g);
  for (n = 0; n < 20; n++) {
    if (rayo_generator_prompt(g, "the cat sat on the mat.", 23, &msg) != RAYO_OK) break;
    EXPECT(rayo_generator_position(g) <= 48);
  }
  EXPECT(n < 20);
  rayo_generator_destroy(g);
  /* By default the context is the model's or RAYO_GENERATOR_CONTEXT,
   * whichever is less. */
  memset(&o, 0, sizeof o);
  EXPECT(rayo_generator_create(m, t, &o, &g, &msg) == RAYO_OK);
  EXPECT(g && rayo_generator_context(g) ==
                  (256 < RAYO_GENERATOR_CONTEXT ? 256 : RAYO_GENERATOR_CONTEXT));
  rayo_generator_destroy(g);
  rayo_tokenizer_destroy(t);
  rayo_model_destroy(m);
}

/* A reset gives the cache's pages back and leaves it zero; the memory
 * report adds up. */
static void memory(void) {
  rayo_model *m = load("llama-q4h.weights");
  rayo_session *s;
  rayo_process_memory_info pm;
  rayo_status st;
  rayo_i32 ids[64];
  size_t i;
  if (!m) return;
  for (i = 0; i < 64; i++) ids[i] = (rayo_i32)(i % 200);
  s = session(m, RAYO_CACHE_INT8, 256, 16);
  if (s) {
    float first[8];
    rayo_tensor_view out;
    run_chunks(s, ids, 64, 16);
    rayo_session_output(s, 0, &out);
    memcpy(first, out.data, sizeof first);
    rayo_session_reset(s);
    run_chunks(s, ids, 64, 16);
    rayo_session_output(s, 0, &out);
    EXPECT(memcmp(first, out.data, sizeof first) == 0);
  }
  rayo_session_destroy(s);
  st = rayo_process_memory(&pm);
  EXPECT(st == RAYO_OK || st == RAYO_ERR_UNSUPPORTED);
  if (st == RAYO_OK) {
    printf("  process: private %.1f MB, file-backed %.1f MB resident\n",
           (double)pm.private_bytes / 1048576.0, (double)pm.file_bytes / 1048576.0);
    EXPECT(pm.private_bytes > 0 && pm.resident_bytes >= pm.private_bytes);
  }
  EXPECT(rayo_process_memory(NULL) == RAYO_ERR_INVALID_ARGUMENT);
  EXPECT(rayo_cpu_default_threads() >= 1 && rayo_cpu_default_threads() <= 4);
  rayo_model_destroy(m);
}

void test_lean(void) {
  table_from_model();
  cache_and_chunks();
  generator();
  memory();
}
