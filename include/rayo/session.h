/* rayo/session.h: running a model. */
#ifndef RAYO_SESSION_H
#define RAYO_SESSION_H

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/model.h"
#include "rayo/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rayo_session rayo_session;

/* A session plans all of its memory when it is created, for the largest
 * sizes the graph allows: rayo_session_run allocates nothing. On success
 * `*out` is a new session the caller owns and frees with
 * rayo_session_destroy, before the model. One session runs on one thread
 * at a time; separate sessions may run at once. */
RAYO_API rayo_status rayo_session_create(const rayo_model *model, rayo_session **out,
                                         rayo_message *msg);

/* How a session keeps the states that are key/value caches. */
typedef enum rayo_cache_type {
  RAYO_CACHE_FLOAT32 = 0, /* as the graph declares them */
  /* int8, with a float32 scale per row (token) and head: a quarter of the
   * memory. Applies to each state that only cache_write nodes write and only
   * attention nodes read; the head size is the attention's (a state that
   * attention reads with another head size stays float32). Attention's
   * outputs differ from float32's by the rounding of the keys and values
   * (docs/OPS.md, cache_write). */
  RAYO_CACHE_INT8 = 1
} rayo_cache_type;

/* A lower limit for size symbol `name`: at most `max` (at most the graph's
 * largest), and memory planned for `plan` (at most `max`), in place of the
 * graph's "plan". */
typedef struct rayo_symbol_limit {
  const char *name;
  rayo_i32 max;
  rayo_i32 plan;
} rayo_symbol_limit;

/* Options for rayo_session_create_with. Zero them (memset) before setting
 * what you need, so options added later keep their defaults. */
typedef struct rayo_session_options {
  /* The threads a run uses, the caller's included: 0 or 1 for one (the
   * default), -1 for rayo_cpu_fast_cores(). The results are the same, bit
   * for bit, with any number: work is split by output, never within a sum. */
  int threads;
  /* RAYO_CACHE_FLOAT32 (0) or RAYO_CACHE_INT8. */
  int cache;
  /* At most this many rows (the first dimension) in each state, such as a
   * cache's tokens: 0 for the graph's sizes. With it, give `limits` that keep
   * the symbols within the rows (a decoder's P + T at most state_rows). */
  rayo_i32 state_rows;
  /* `limit_count` lower limits of size symbols (NULL: none). */
  const rayo_symbol_limit *limits;
  int limit_count;
} rayo_session_options;

/* rayo_session_create with `options` (NULL: the defaults). */
RAYO_API rayo_status rayo_session_create_with(const rayo_model *model,
                                              const rayo_session_options *options,
                                              rayo_session **out, rayo_message *msg);

/* The threads the session runs on. */
RAYO_API int rayo_session_threads(const rayo_session *session);

/* The CPU cores worth running on: a phone's big cores (from /sys on
 * Android and Linux), a Mac's performance or physical cores, else the cores
 * online; at least 1. */
RAYO_API int rayo_cpu_fast_cores(void);

/* The threads a language model runs on by default: at most 2 of the fast
 * cores on Android, at most 4 elsewhere; at least 1. */
RAYO_API int rayo_cpu_default_threads(void);

/* Frees `session` (NULL is fine). */
RAYO_API void rayo_session_destroy(rayo_session *session);

/* Sets size symbol `name` (such as "T") to `value`, at most the largest the
 * graph declares, for the next runs. */
RAYO_API rayo_status rayo_session_bind(rayo_session *session, const char *name, rayo_i32 value,
                                       rayo_message *msg);

/* Input `index`, at the current sizes, for the caller to write into. The
 * memory belongs to the session. */
RAYO_API rayo_status rayo_session_input(rayo_session *session, int index, rayo_tensor_view *out);

/* Runs the graph on the inputs written. */
RAYO_API rayo_status rayo_session_run(rayo_session *session, rayo_message *msg);

/* Output `index` of the last run, for the caller to read. The memory
 * belongs to the session and changes at the next run. */
RAYO_API rayo_status rayo_session_output(const rayo_session *session, int index,
                                         rayo_tensor_view *out);

/* Zeroes the session's states (such as a decoder's key/value cache), to
 * start a new sequence; their whole pages go back to the system until they
 * are written again. NULL is fine. */
RAYO_API void rayo_session_reset(rayo_session *session);

/* The kernels the session runs, such as "neon-dotprod" or "avx2". Static. */
RAYO_API const char *rayo_session_kernels(const rayo_session *session);

RAYO_API void rayo_session_memory(const rayo_session *session, rayo_memory_info *out);

#ifdef __cplusplus
}
#endif

#endif
