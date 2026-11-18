/* SHA-512 (FIPS 180-4), for Ed25519 signatures. C89 has no 64-bit
 * integer, so each 64-bit word is two 32-bit halves. */
#ifndef RAYO_CRYPTO_SHA512_H
#define RAYO_CRYPTO_SHA512_H

#include <stddef.h>

#include "rayo/types.h"

typedef struct rayo_sha512 {
  rayo_u32 h[16]; /* the state: 8 words, each high half then low half */
  rayo_u8 block[128];
  size_t fill;       /* bytes in `block` */
  rayo_u32 bytes_lo; /* bytes hashed so far, mod 2^64 */
  rayo_u32 bytes_hi;
} rayo_sha512;

void rayo_sha512_init(rayo_sha512 *s);
void rayo_sha512_update(rayo_sha512 *s, const void *data, size_t size);
void rayo_sha512_final(rayo_sha512 *s, rayo_u8 out[64]);

#endif
