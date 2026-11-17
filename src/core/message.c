#include "core/message.h"

#include <stdarg.h>
#include <stdio.h>

rayo_status rayo_fail(rayo_message *msg, rayo_status status, const char *fmt, ...) {
  if (msg) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg->text, sizeof msg->text, fmt, ap);
    va_end(ap);
  }
  return status;
}

const char *rayo_status_str(rayo_status status) {
  switch (status) {
    case RAYO_OK:
      return "ok";
    case RAYO_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case RAYO_ERR_FORMAT:
      return "format error";
    case RAYO_ERR_UNSUPPORTED:
      return "unsupported";
    case RAYO_ERR_OUT_OF_MEMORY:
      return "out of memory";
    case RAYO_ERR_SHAPE:
      return "shape error";
  }
  return "unknown status";
}
