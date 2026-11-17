/* Filling in a caller's rayo_message. */
#ifndef RAYO_CORE_MESSAGE_H
#define RAYO_CORE_MESSAGE_H

#include "rayo/error.h"

/* Writes the formatted text into `msg` (if not NULL) and returns `status`,
 * so a function can `return rayo_fail(msg, RAYO_ERR_FORMAT, "...")`. */
rayo_status rayo_fail(rayo_message *msg, rayo_status status, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

#endif
