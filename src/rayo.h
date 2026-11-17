// rayo: a small on-device inference runtime on ggml.
//
// A model is a GGUF file holding its weights and, under the metadata key
// "rayo.graph", the graph that runs them (docs/FORMAT.md). A session runs
// one model with its own threads, scratch memory and state (such as a
// transformer's key/value cache).
//
// Every function is safe to call from any thread, but one session must not
// be used from two threads at once. Errors are reported as a nonzero status
// and a message written into `err` (NUL-terminated, at most `err_len` bytes).

#ifndef RAYO_H
#define RAYO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RAYO_API __declspec(dllexport)
#else
#define RAYO_API __attribute__((visibility("default")))
#endif

typedef struct rayo_model rayo_model;
typedef struct rayo_session rayo_session;

enum rayo_dtype {
  RAYO_F32 = 0,
  RAYO_I32 = 1,
};

/// The runtime's version, such as "0.1.0 (ggml 1.0.0 417a024)".
RAYO_API const char *rayo_version(void);

/// Writes the SHA-256 of `size` bytes at `data` into `out` (32 bytes).
RAYO_API void rayo_sha256(const void *data, size_t size, uint8_t out[32]);

/// Reads a model from `size` bytes at `data` (copied; the caller may free
/// them afterwards). Null on error.
RAYO_API rayo_model *rayo_model_load(const void *data, size_t size,
                                        char *err, size_t err_len);

RAYO_API void rayo_model_free(rayo_model *model);

/// The model's metadata, inputs, states and outputs as a JSON object. Owned
/// by the model.
RAYO_API const char *rayo_model_info(const rayo_model *model);

/// A session on `model` with `n_threads` threads (1 runs on the calling
/// thread). The model must outlive it. Null on error.
RAYO_API rayo_session *rayo_session_new(rayo_model *model, int n_threads,
                                           char *err, size_t err_len);

RAYO_API void rayo_session_free(rayo_session *session);

/// Gives the size symbol `name` (such as "T") a value for the next runs.
RAYO_API int rayo_session_bind(rayo_session *session, const char *name,
                                 int64_t value, char *err, size_t err_len);

/// Copies `n_bytes` of input `name` (in the dtype the graph declares) for the
/// next run.
RAYO_API int rayo_session_set_input(rayo_session *session, const char *name,
                                      const void *data, size_t n_bytes,
                                      char *err, size_t err_len);

/// Runs the graph on the inputs and bindings given.
RAYO_API int rayo_session_run(rayo_session *session, char *err,
                                size_t err_len);

/// Output `name` of the last run as float32: its data (valid until the next
/// run or free) and its shape, innermost dimension first, in `ne[4]`.
RAYO_API const float *rayo_session_output(rayo_session *session,
                                            const char *name, int64_t ne[4],
                                            char *err, size_t err_len);

/// Zeroes the session's state tensors.
RAYO_API void rayo_session_reset(rayo_session *session);

#ifdef __cplusplus
}
#endif

#endif
