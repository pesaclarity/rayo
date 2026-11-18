/* rayo/verify.h: checking a model file before rayo reads it, by its SHA-256
 * and its Ed25519 signature (RFC 8032), so an app can require that a model
 * it downloaded is the one its publisher signed (docs/SECURITY.md). No
 * other library: rayo has its own SHA-256, SHA-512 and Ed25519. */
#ifndef RAYO_VERIFY_H
#define RAYO_VERIFY_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What a model file must match. Zero it (memset) and set what you need:
 * each part that is set is checked, and every check must pass. */
typedef struct rayo_integrity {
  /* The 32 bytes of the file's SHA-256, or NULL. */
  const rayo_u8 *sha256;
  /* An Ed25519 public key (32 bytes) and the file's signature by it (64
   * bytes: "pure" Ed25519 over the whole file, as RFC 8032 signs a
   * message, such as `openssl pkeyutl -sign -rawin` or
   * tool/sign_model.py writes), or both NULL. */
  const rayo_u8 *public_key;
  const rayo_u8 *signature;
} rayo_integrity;

/* RAYO_OK when the `size` bytes at `data` match `want`;
 * RAYO_ERR_INTEGRITY, with a message saying which check failed, when they
 * do not; RAYO_ERR_INVALID_ARGUMENT when `want` asks for nothing or gives a
 * public key without a signature (or the reverse). */
RAYO_API rayo_status rayo_verify(const void *data, size_t size, const rayo_integrity *want,
                                 rayo_message *msg);

/* The same for the file at `path`, read (mapped where the system can) and
 * released again. To use the bytes that were checked, pass `want` to
 * rayo_model_load or rayo_model_create_with instead (rayo_model_options'
 * `integrity`): the file could change between a check and a later load. */
RAYO_API rayo_status rayo_verify_file(const char *path, const rayo_integrity *want,
                                      rayo_message *msg);

#ifdef __cplusplus
}
#endif

#endif
