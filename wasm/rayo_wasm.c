/* A small C ABI for running rayo as WebAssembly from a JavaScript host
 * (docs/WASM.md). Every argument and result is an i32 (a pointer, a size or
 * a status), so a host needs no bindings generator: wasm/rayo.mjs is one.
 *
 * The host copies a model's bytes into memory from rayo_wasm_alloc, opens
 * it (rayo copies what it needs, so the bytes may be freed right after),
 * binds dimensions, writes inputs in place through rayo_wasm_input, runs,
 * and reads outputs in place through rayo_wasm_output. Built only for wasm
 * (wasm/build.sh); not part of the native library. */
#include <stdlib.h>

#include "rayo/rayo.h"

#if defined(__wasm__)
#define RAYO_WASM_EXPORT(name) __attribute__((export_name(#name)))
#else
#define RAYO_WASM_EXPORT(name)
#endif

typedef struct rayo_wasm_handle {
  rayo_model *model;
  rayo_session *session;
} rayo_wasm_handle;

/* The last error's text, for rayo_wasm_error. */
static rayo_message last;

RAYO_WASM_EXPORT(rayo_wasm_alloc) void *rayo_wasm_alloc(size_t n) {
  return malloc(n ? n : 1);
}

RAYO_WASM_EXPORT(rayo_wasm_free) void rayo_wasm_free(void *p) {
  free(p);
}

/* The text of the last failure, NUL-terminated. */
RAYO_WASM_EXPORT(rayo_wasm_error) const char *rayo_wasm_error(void) {
  return last.text;
}

/* A model and a session for it, or NULL (rayo_wasm_error says why). */
RAYO_WASM_EXPORT(rayo_wasm_open)
rayo_wasm_handle *rayo_wasm_open(const void *bytes, size_t size) {
  rayo_wasm_handle *h = (rayo_wasm_handle *)calloc(1, sizeof *h);
  if (!h) return NULL;
  if (rayo_model_create(bytes, size, &h->model, &last) != RAYO_OK) {
    free(h);
    return NULL;
  }
  if (rayo_session_create(h->model, &h->session, &last) != RAYO_OK) {
    rayo_model_destroy(h->model);
    free(h);
    return NULL;
  }
  return h;
}

RAYO_WASM_EXPORT(rayo_wasm_inputs) int rayo_wasm_inputs(const rayo_wasm_handle *h) {
  return rayo_model_input_count(h->model);
}

RAYO_WASM_EXPORT(rayo_wasm_outputs) int rayo_wasm_outputs(const rayo_wasm_handle *h) {
  return rayo_model_output_count(h->model);
}

/* 0 or a rayo_status. */
RAYO_WASM_EXPORT(rayo_wasm_bind)
int rayo_wasm_bind(rayo_wasm_handle *h, const char *name, int value) {
  return (int)rayo_session_bind(h->session, name, (rayo_i32)value, &last);
}

/* Where input `index` lives, its size in *bytes; NULL if there is none. */
RAYO_WASM_EXPORT(rayo_wasm_input)
void *rayo_wasm_input(rayo_wasm_handle *h, int index, size_t *bytes) {
  rayo_tensor_view v;
  if (rayo_session_input(h->session, index, &v) != RAYO_OK) return NULL;
  *bytes = v.bytes;
  return v.data;
}

/* 0 or a rayo_status. */
RAYO_WASM_EXPORT(rayo_wasm_run) int rayo_wasm_run(rayo_wasm_handle *h) {
  return (int)rayo_session_run(h->session, &last);
}

/* Where output `index` lives after a run, its size in *bytes. */
RAYO_WASM_EXPORT(rayo_wasm_output)
void *rayo_wasm_output(rayo_wasm_handle *h, int index, size_t *bytes) {
  rayo_tensor_view v;
  if (rayo_session_output(h->session, index, &v) != RAYO_OK) return NULL;
  *bytes = v.bytes;
  return v.data;
}

/* The kernel set the session runs ("generic" or "wasm-simd128"). */
RAYO_WASM_EXPORT(rayo_wasm_kernels) const char *rayo_wasm_kernels(const rayo_wasm_handle *h) {
  return rayo_session_kernels(h->session);
}

RAYO_WASM_EXPORT(rayo_wasm_close) void rayo_wasm_close(rayo_wasm_handle *h) {
  if (!h) return;
  rayo_session_destroy(h->session);
  rayo_model_destroy(h->model);
  free(h);
}
