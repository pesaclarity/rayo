/* rayo/sha256.h: SHA-256, so a caller can check a model file before
 * loading it without another library. */
#ifndef RAYO_PUBLIC_SHA256_H
#define RAYO_PUBLIC_SHA256_H

#include <stddef.h>

#include "rayo/export.h"
#include "rayo/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Writes the 32-byte digest of the `size` bytes at `data` into `out`. */
RAYO_API void rayo_sha256(const void *data, size_t size, rayo_u8 out[32]);

#ifdef __cplusplus
}
#endif

#endif
