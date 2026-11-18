#include "rayo/verify.h"

#include <string.h>

#include "core/message.h"
#include "crypto/ed25519.h"
#include "platform/platform.h"
#include "rayo/sha256.h"

#include "core/small_begin.h"

rayo_status rayo_verify(const void *data, size_t size, const rayo_integrity *want,
                        rayo_message *msg) {
  if (!want || (!data && size) || (!want->public_key) != (!want->signature) ||
      (!want->sha256 && !want->public_key)) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT,
                     "an integrity check needs a SHA-256, or a public key and a signature");
  }
  if (want->sha256) {
    rayo_u8 digest[32];
    unsigned diff = 0;
    int i;
    rayo_sha256(data, size, digest);
    for (i = 0; i < 32; i++) diff |= (unsigned)(digest[i] ^ want->sha256[i]);
    if (diff) return rayo_fail(msg, RAYO_ERR_INTEGRITY, "the model does not match its SHA-256");
  }
  if (want->public_key &&
      !rayo_ed25519_verify(want->signature, want->public_key, data ? data : "", size)) {
    return rayo_fail(msg, RAYO_ERR_INTEGRITY,
                     "the model's signature is not the public key's (Ed25519)");
  }
  return RAYO_OK;
}

rayo_status rayo_verify_file(const char *path, const rayo_integrity *want, rayo_message *msg) {
  rayo_file_map map;
  rayo_status s;
  if (!path) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  s = rayo_map_file(path, RAYO_FALSE, &map, msg);
  if (s != RAYO_OK) return s;
  rayo_map_advise(&map, RAYO_ADVICE_SEQUENTIAL);
  s = rayo_verify(map.data, map.size, want, msg);
  rayo_unmap_file(&map);
  return s;
}

#include "core/small_end.h"
