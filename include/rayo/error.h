/* rayo/error.h: status codes and error messages. */
#ifndef RAYO_ERROR_H
#define RAYO_ERROR_H

#include "rayo/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rayo_status {
  RAYO_OK = 0,
  RAYO_ERR_INVALID_ARGUMENT = 1, /* a null pointer, an index out of range */
  RAYO_ERR_FORMAT = 2,           /* the model file is damaged or not one */
  RAYO_ERR_UNSUPPORTED = 3,      /* a version, op or dtype this rayo does not run */
  RAYO_ERR_OUT_OF_MEMORY = 4,
  RAYO_ERR_SHAPE = 5, /* sizes that do not fit the graph */
} rayo_status;

/* A human-readable message for a failed call. Functions that can fail take
 * an optional `rayo_message *` and fill it in; pass NULL to skip it. */
typedef struct rayo_message {
  char text[256];
} rayo_message;

/* A static name for `status`, such as "format error". */
RAYO_API const char *rayo_status_str(rayo_status status);

#ifdef __cplusplus
}
#endif

#endif
