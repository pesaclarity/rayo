/* A tiny Llama-style language model (tool/make_llama_test_model.py), float32
 * and q4, against the NumPy reference: the logits after a prompt run in
 * two chunks, greedy decoding with the key/value cache, the same results
 * with threads; and the tokeniser, the sampler and the generator on it. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/json.h"
#include "test.h"

static void model_path(const char *name, char *out, size_t size) {
  const char *dir = getenv("RAYO_TEST_MODELS");
  if (!dir) dir = "tests/models";
  out[0] = 0;
  if (strlen(dir) + strlen(name) + 2 > size) return;
  strcpy(out, dir);
  strcat(out, "/");
  strcat(out, name);
}

static rayo_i32 argmax(const float *v, size_t n) {
  size_t i, best = 0;
  for (i = 1; i < n; i++) {
    if (v[i] > v[best]) best = i;
  }
  return (rayo_i32)best;
}

static void run(rayo_session *s, const rayo_i32 *ids, rayo_i32 p, rayo_i32 t) {
  rayo_tensor_view in;
  EXPECT(rayo_session_bind(s, "P", p, NULL) == RAYO_OK);
  EXPECT(rayo_session_bind(s, "T", t, NULL) == RAYO_OK);
  rayo_session_input(s, 0, &in);
  memcpy(in.data, ids, (size_t)t * sizeof *ids);
  EXPECT(rayo_session_run(s, NULL) == RAYO_OK);
}

/* The prompt in chunks of 16, then greedy: the logits after the prompt in
 * `logits` (vocab of them), the tokens in `greedy` (8). */
static void decode(rayo_session *s, const rayo_i32 *prompt, rayo_i32 n, float *logits, size_t vocab,
                   rayo_i32 *greedy) {
  rayo_tensor_view out;
  rayo_i32 i, at = 0;
  test_choose_kernels(s);
  while (at < n) {
    const rayo_i32 c = n - at < 16 ? n - at : 16;
    run(s, prompt + at, at, c);
    at += c;
  }
  rayo_session_output(s, 0, &out);
  EXPECT(out.bytes == vocab * sizeof(float));
  memcpy(logits, out.data, vocab * sizeof(float));
  for (i = 0; i < 8; i++) {
    rayo_session_output(s, 0, &out);
    greedy[i] = argmax(out.data, vocab);
    run(s, &greedy[i], n + i, 1);
  }
}

static void ints(const rayo_json *a, rayo_i32 *out, size_t n) {
  size_t i;
  for (i = 0; i < n && a && i < a->length; i++) rayo_json_as_int(a->items[i], &out[i]);
}

void test_llama(void) {
  char path[512];
  rayo_arena arena;
  rayo_json *want = NULL;
  unsigned char *text;
  size_t size, vocab, i;
  rayo_i32 prompt[20], greedy[8], got[8], again[8];
  float *logits, *logits3;
  int k;
  memset(&arena, 0, sizeof arena);
  text = test_model_file("llama_expected.json", &size);
  if (!text || rayo_json_parse((const char *)text, size, &arena, &want, NULL) != RAYO_OK) {
    EXPECT(0);
    free(text);
    return;
  }
  vocab = rayo_json_get(want, "logits")->length;
  ints(rayo_json_get(want, "prompt"), prompt, 20);
  ints(rayo_json_get(want, "greedy"), greedy, 8);
  logits = malloc(vocab * sizeof *logits);
  logits3 = malloc(vocab * sizeof *logits3);
  for (k = 0; k < 2 && logits && logits3; k++) {
    const rayo_json *ref = rayo_json_get(want, k ? "logits_q4" : "logits");
    rayo_model *m = NULL;
    rayo_session *s = NULL, *s3 = NULL;
    rayo_session_options o;
    rayo_message msg;
    double worst = 0, top = 0;
    model_path(k ? "llama-q4.weights" : "llama.weights", path, sizeof path);
    EXPECT(rayo_model_load(path, NULL, &m, &msg) == RAYO_OK);
    if (!m) {
      fprintf(stderr, "  %s\n", msg.text);
      continue;
    }
    memset(&o, 0, sizeof o);
    o.threads = 3;
    EXPECT(rayo_session_create(m, &s, NULL) == RAYO_OK);
    EXPECT(rayo_session_create_with(m, &o, &s3, NULL) == RAYO_OK);
    if (s && s3) {
      decode(s, prompt, 20, logits, vocab, got);
      decode(s3, prompt, 20, logits3, vocab, again);
      for (i = 0; i < vocab; i++) {
        const double d = fabs(logits[i] - ref->items[i]->number);
        worst = d > worst ? d : worst;
        top = fabs(ref->items[i]->number) > top ? fabs(ref->items[i]->number) : top;
      }
      printf("  llama %s (%s): logits within %.2g of NumPy (largest %.2f)\n", k ? "q4" : "float32",
             rayo_session_kernels(s), worst, top);
      /* float32: as NumPy to float32 rounding; q4: against NumPy on the same
       * 4-bit weights, the activations' int8 rounding. */
      EXPECT(worst < (k ? 0.03 * top : 1e-4 * (1 + top)));
      EXPECT(memcmp(logits, logits3, vocab * sizeof *logits) == 0);
      EXPECT(memcmp(got, again, sizeof got) == 0);
      EXPECT(rayo_session_threads(s3) == 3 || rayo_session_threads(s3) == 1);
      if (!k) EXPECT(memcmp(got, greedy, sizeof got) == 0);
    }
    rayo_session_destroy(s);
    rayo_session_destroy(s3);
    rayo_model_destroy(m);
  }
  free(logits);
  free(logits3);
  free(text);
  rayo_arena_free(&arena);
}

void test_text(void) {
  static const char sample[] = "the cat sat on the mat, hello there! caf\xc3\xa9 \xf0\x9f\x99\x82";
  char path[512];
  rayo_tokenizer *t = NULL;
  rayo_sampler *sm = NULL;
  rayo_sampler_options so;
  rayo_message msg;
  rayo_i32 ids[128], end;
  char joined[256];
  size_t n = 0, len = 0, i, size;
  float logits[6] = {0.5f, 2.0f, -1.0f, 1.9f, 0.0f, -3.0f};
  rayo_i32 recent[1] = {1}, a[16], b[16];
  model_path("llama.tokenizer.json", path, sizeof path);
  EXPECT(rayo_tokenizer_load(path, &t, &msg) == RAYO_OK);
  if (!t) {
    fprintf(stderr, "  %s\n", msg.text);
    return;
  }
  EXPECT(rayo_tokenizer_encode(t, sample, strlen(sample), ids, 128, &n, NULL) == RAYO_OK);
  EXPECT(n > 5 && n < strlen(sample)); /* merges joined some bytes */
  for (i = 0; i < n && n <= 128; i++) {
    const char *p = rayo_tokenizer_piece(t, ids[i], &size);
    EXPECT(p != NULL);
    if (p && len + size < sizeof joined) {
      memcpy(joined + len, p, size);
      len += size;
    }
  }
  EXPECT(len == strlen(sample) && memcmp(joined, sample, len) == 0);
  end = rayo_tokenizer_find(t, "<|im_end|>");
  EXPECT(end >= 0);
  EXPECT(rayo_tokenizer_encode(t, "a<|im_end|>", 11, ids, 128, &n, NULL) == RAYO_OK);
  EXPECT(n == 2 && ids[1] == end);
  EXPECT(rayo_tokenizer_encode(t, sample, strlen(sample), ids, 2, &n, NULL) == RAYO_ERR_SHAPE);
  rayo_tokenizer_destroy(t);
  /* The sampler: greedy, top-k 1, the penalty, and the seed. */
  memset(&so, 0, sizeof so);
  EXPECT(rayo_sampler_create(6, &so, &sm, NULL) == RAYO_OK);
  EXPECT(rayo_sampler_next(sm, logits, NULL, 0) == 1);
  EXPECT(rayo_sampler_next(sm, logits, recent, 1) == 1); /* no penalty asked */
  rayo_sampler_destroy(sm);
  so.repetition_penalty = 1.2f;
  EXPECT(rayo_sampler_create(6, &so, &sm, NULL) == RAYO_OK);
  EXPECT(rayo_sampler_next(sm, logits, recent, 1) == 3); /* 2.0 / 1.2 < 1.9 */
  rayo_sampler_destroy(sm);
  so.repetition_penalty = 0.0f;
  so.temperature = 1.0f;
  so.top_k = 1;
  EXPECT(rayo_sampler_create(6, &so, &sm, NULL) == RAYO_OK);
  EXPECT(rayo_sampler_next(sm, logits, NULL, 0) == 1);
  rayo_sampler_destroy(sm);
  so.top_k = 0;
  so.top_p = 0.95f;
  so.seed = 42;
  for (i = 0; i < 2; i++) {
    size_t j;
    EXPECT(rayo_sampler_create(6, &so, &sm, NULL) == RAYO_OK);
    if (!sm) return;
    for (j = 0; j < 16; j++) (i ? b : a)[j] = rayo_sampler_next(sm, logits, NULL, 0);
    rayo_sampler_destroy(sm);
  }
  EXPECT(memcmp(a, b, sizeof a) == 0);
  for (i = 0; i < 16; i++) EXPECT(a[i] >= 0 && a[i] < 6 && a[i] != 5); /* 5 is below top-p */
}

void test_generator(void) {
  char path[512];
  rayo_model *m = NULL;
  rayo_tokenizer *t = NULL;
  rayo_generator *g = NULL;
  rayo_generator_options o;
  rayo_message msg;
  rayo_i32 token, first[4], n = 0, at;
  long before;
  memset(&o, 0, sizeof o);
  o.max_tokens = 4;
  model_path("llama-q4.weights", path, sizeof path);
  EXPECT(rayo_model_load(path, NULL, &m, &msg) == RAYO_OK);
  model_path("llama.tokenizer.json", path, sizeof path);
  EXPECT(rayo_tokenizer_load(path, &t, &msg) == RAYO_OK);
  EXPECT(m && t && rayo_generator_create(m, t, &o, &g, &msg) == RAYO_OK);
  if (!g) {
    fprintf(stderr, "  %s\n", msg.text);
    rayo_tokenizer_destroy(t);
    rayo_model_destroy(m);
    return;
  }
  EXPECT(rayo_generator_can_chat(g) && rayo_generator_context(g) == 256);
  EXPECT(rayo_generator_next(g, &token, NULL) != RAYO_OK); /* no prompt yet */
  EXPECT(rayo_generator_chat(g, "be brief", "hello there", &msg) == RAYO_OK);
  at = rayo_generator_position(g);
  EXPECT(at > 5);
  /* A reply's tokens run with no allocation. */
  before = test_allocations();
  while (rayo_generator_next(g, &token, NULL) == RAYO_OK && token >= 0 && n < 4) first[n++] = token;
  EXPECT(before < 0 || test_allocations() == before);
  EXPECT(n == 4);
  /* A second turn goes on in the cache. */
  EXPECT(rayo_generator_chat(g, NULL, "and the cat?", &msg) == RAYO_OK);
  EXPECT(rayo_generator_position(g) > at + 4);
  /* Again from the start: greedy, so the same first reply. */
  rayo_generator_reset(g);
  EXPECT(rayo_generator_position(g) == 0);
  EXPECT(rayo_generator_chat(g, "be brief", "hello there", &msg) == RAYO_OK);
  for (n = 0; n < 4; n++) {
    EXPECT(rayo_generator_next(g, &token, NULL) == RAYO_OK && token == first[n]);
  }
  /* The context fills up: refused, not overrun. */
  rayo_generator_reset(g);
  for (n = 0; n < 20; n++) {
    rayo_status st =
        rayo_generator_prompt(g, "the cat sat on the mat. the dog sat on the log.", 47, &msg);
    EXPECT(rayo_generator_position(g) <= 256);
    if (st != RAYO_OK) {
      EXPECT(st == RAYO_ERR_SHAPE);
      break;
    }
  }
  EXPECT(n < 20);
  rayo_generator_destroy(g);
  rayo_tokenizer_destroy(t);
  rayo_model_destroy(m);
}
