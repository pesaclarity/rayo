/* Ed25519 signature verification (RFC 8032, "pure" Ed25519: the message
 * itself is signed), for checking model files. The field and curve code is
 * Daniel Beer's c25519 (public domain, LICENSES/c25519.txt), made C89 and
 * cut to what verifying needs; SHA-512 is crypto/sha512.c. */
#ifndef RAYO_CRYPTO_ED25519_H
#define RAYO_CRYPTO_ED25519_H

#include <stddef.h>

#include "rayo/types.h"

/* Whether `signature` (64 bytes: R, then S) is `public_key`'s (32 bytes)
 * over the `size` bytes at `message`. Strict: S must be below the group's
 * order and R and the key canonical encodings of points on the curve, so a
 * signature has one form. Variable time in nothing secret: all is public. */
rayo_bool rayo_ed25519_verify(const rayo_u8 signature[64], const rayo_u8 public_key[32],
                              const void *message, size_t size);

#endif
