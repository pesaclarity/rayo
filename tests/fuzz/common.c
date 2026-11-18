/* What the fuzz targets share. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuzz.h"
#include "rayo/rayo.h"
#include "rayo/text.h"
#include "session.h"

unsigned char *rayo_fuzz_read(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  unsigned char *p = NULL;
  size_t n = 0, cap = 0;
  if (!f) return NULL;
  for (;;) {
    size_t got;
    if (n + 4096 + 1 > cap) {
      unsigned char *q;
      cap = cap ? cap * 2 : 65536;
      q = (unsigned char *)realloc(p, cap);
      if (!q) {
        free(p);
        fclose(f);
        return NULL;
      }
      p = q;
    }
    got = fread(p + n, 1, 4096, f);
    n += got;
    if (got < 4096) break;
  }
  fclose(f);
  p[n] = 0;
  *size = n;
  return p;
}

void rayo_fuzz_encode(const rayo_tokenizer *t, const char *text, size_t size) {
  rayo_i32 ids[256];
  size_t count = 0, i;
  const rayo_status s = rayo_tokenizer_encode(t, text, size, ids, 256, &count, NULL);
  if (s != RAYO_OK && s != RAYO_ERR_SHAPE) return;
  for (i = 0; i < count && i < 256; i++) {
    size_t n = 0;
    const char *piece = rayo_tokenizer_piece(t, ids[i], &n);
    if (piece && n) {
      volatile char c = piece[n - 1];
      (void)c;
    }
    if (ids[i] < 0 || ids[i] >= rayo_tokenizer_vocab_size(t)) abort();
  }
}

/* A deterministic value for int32 input element i: mostly small ids, some
 * out of any range. */
static rayo_i32 id_at(size_t i, unsigned seed) {
  const unsigned h = (unsigned)(i * 2654435761u) ^ seed;
  if ((h & 63u) == 0) return (h & 64u) ? -1 : 2147483647;
  return (rayo_i32)(h % 9u);
}

static void fill_inputs(rayo_session *s, unsigned seed) {
  int i, n = rayo_model_input_count(s->model);
  for (i = 0; i < n; i++) {
    rayo_tensor_view v;
    size_t k, count;
    if (rayo_session_input(s, i, &v) != RAYO_OK) abort();
    if (v.dtype == RAYO_DTYPE_I32) {
      count = v.bytes / 4;
      for (k = 0; k < count; k++) ((rayo_i32 *)v.data)[k] = id_at(k, seed);
    } else {
      count = v.bytes / 4;
      for (k = 0; k < count; k++) ((float *)v.data)[k] = (float)((int)(k % 7) - 3) * 0.25f;
    }
  }
}

static void read_outputs(const rayo_session *s) {
  int i, n = rayo_model_output_count(s->model);
  for (i = 0; i < n; i++) {
    rayo_tensor_view v;
    if (rayo_session_output(s, i, &v) != RAYO_OK) abort();
    if (v.bytes) {
      volatile unsigned char c = ((const unsigned char *)v.data)[v.bytes - 1];
      (void)c;
    }
  }
}

static void run_session(const rayo_model *m, const rayo_session_options *o, unsigned seed) {
  rayo_session *s = NULL;
  rayo_memory_info mem;
  rayo_i32 k;
  int round;
  if (rayo_session_create_with(m, o, &s, NULL) != RAYO_OK) return;
  rayo_session_memory(s, &mem);
  if (mem.arena > (size_t)RAYO_FUZZ_MAX_ARENA) {
    rayo_session_destroy(s);
    return;
  }
  /* Small sizes: every symbol 1, then a few of them otherwise. */
  for (round = 0; round < 3; round++) {
    for (k = 0; k < s->model->graph.symbol_count; k++) {
      const rayo_i32 v =
          round == 0 ? 1 : (rayo_i32)((seed >> (k % 8)) % (RAYO_FUZZ_MAX_SYMBOL + 1));
      rayo_session_bind(s, s->model->graph.symbols[k], v, NULL);
    }
    fill_inputs(s, seed + (unsigned)round);
    if (rayo_session_run(s, NULL) == RAYO_OK) read_outputs(s);
  }
  rayo_session_reset(s);
  rayo_session_destroy(s);
}

void rayo_fuzz_model(const unsigned char *data, size_t size) {
  rayo_model *m = NULL;
  rayo_tokenizer *t = NULL;
  rayo_session_options o;
  unsigned seed = size ? data[size - 1] * 16777619u : 0u;
  if (rayo_model_create(data, size, &m, NULL) != RAYO_OK) return;
  {
    int i;
    rayo_tensor_view v;
    for (i = 0; i < rayo_model_input_count(m); i++) rayo_model_input_info(m, i, &v);
    for (i = 0; i < rayo_model_output_count(m); i++) rayo_model_output_info(m, i, &v);
    (void)strlen(rayo_model_name(m));
    (void)strlen(rayo_model_metadata_json(m));
  }
  if (rayo_tokenizer_from_model(m, &t, NULL) == RAYO_OK) {
    static const char text[] = "Hello there, friends! <|im_end|> 1234 caf\xc3\xa9 \xff\xfe";
    rayo_fuzz_encode(t, text, sizeof text - 1);
    rayo_tokenizer_destroy(t);
  }
  run_session(m, NULL, seed);
  memset(&o, 0, sizeof o);
  o.cache = RAYO_CACHE_INT8;
  o.state_rows = 4;
  run_session(m, &o, seed ^ 0x5bd1e995u);
  rayo_model_destroy(m);
}
