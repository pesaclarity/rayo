#include "rayo/version.h"

#define RAYO_STR2(x) #x
#define RAYO_STR(x) RAYO_STR2(x)

const char *rayo_version_string(void) {
  return RAYO_STR(RAYO_VERSION_MAJOR) "." RAYO_STR(RAYO_VERSION_MINOR) "." RAYO_STR(
      RAYO_VERSION_PATCH);
}
