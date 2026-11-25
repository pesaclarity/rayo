/* Spike: a minimal C ABI for running rayo as WebAssembly. Not part of the
 * library. The host copies model bytes into memory from rw_alloc, opens a
 * model, binds dimensions, writes inputs through rw_input, runs, and reads
 * outputs through rw_output / rw_output_bytes. */
#include <stdlib.h>
#include <string.h>
#include "rayo/rayo.h"

typedef struct rw_handle {
  rayo_model *model;
  rayo_session *session;
  void *bytes; /* model bytes; rayo reads them in place */
} rw_handle;

static rayo_message last;

void *rw_alloc(size_t n) { return malloc(n); }
void rw_free(void *p) { free(p); }
const char *rw_error(void) { return last.text; }

/* Takes ownership of bytes (from rw_alloc). */
rw_handle *rw_open(void *bytes, size_t size) {
  rw_handle *h = (rw_handle *)calloc(1, sizeof *h);
  if (!h) return NULL;
  h->bytes = bytes;
  if (rayo_model_create(bytes, size, &h->model, &last) != RAYO_OK) { free(h); return NULL; }
  if (rayo_session_create(h->model, &h->session, &last) != RAYO_OK) {
    rayo_model_destroy(h->model); free(h); return NULL;
  }
  return h;
}

int rw_bind(rw_handle *h, const char *name, int value) {
  return (int)rayo_session_bind(h->session, name, (rayo_i32)value, &last);
}

void *rw_input(rw_handle *h, int index, size_t *bytes) {
  rayo_tensor_view v;
  if (rayo_session_input(h->session, index, &v) != RAYO_OK) return NULL;
  *bytes = v.bytes;
  return v.data;
}

int rw_run(rw_handle *h) { return (int)rayo_session_run(h->session, &last); }

void *rw_output(rw_handle *h, int index, size_t *bytes) {
  rayo_tensor_view v;
  if (rayo_session_output(h->session, index, &v) != RAYO_OK) return NULL;
  *bytes = v.bytes;
  return v.data;
}

const char *rw_kernels(rw_handle *h) { return rayo_session_kernels(h->session); }

void rw_close(rw_handle *h) {
  if (!h) return;
  rayo_session_destroy(h->session);
  rayo_model_destroy(h->model);
  free(h->bytes);
  free(h);
}
