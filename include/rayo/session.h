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
 * start a new sequence. NULL is fine. */
RAYO_API void rayo_session_reset(rayo_session *session);

/* The kernels the session runs, such as "neon-dotprod" or "avx2". Static. */
RAYO_API const char *rayo_session_kernels(const rayo_session *session);

RAYO_API void rayo_session_memory(const rayo_session *session, rayo_memory_info *out);

#ifdef __cplusplus
}
#endif

#endif
