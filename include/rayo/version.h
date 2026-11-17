/* rayo/version.h: the library's version. */
#ifndef RAYO_VERSION_H
#define RAYO_VERSION_H

#include "rayo/export.h"

#define RAYO_VERSION_MAJOR 0
#define RAYO_VERSION_MINOR 1
#define RAYO_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

/* "0.1.0". Static storage. */
RAYO_API const char *rayo_version_string(void);

#ifdef __cplusplus
}
#endif

#endif
