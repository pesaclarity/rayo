#include "sha256/sha256.h"

#include <string.h>

static const rayo_u32 k256[64] = {
    0x428a2f98ul, 0x71374491ul, 0xb5c0fbcful, 0xe9b5dba5ul, 0x3956c25bul, 0x59f111f1ul,
    0x923f82a4ul, 0xab1c5ed5ul, 0xd807aa98ul, 0x12835b01ul, 0x243185beul, 0x550c7dc3ul,
    0x72be5d74ul, 0x80deb1feul, 0x9bdc06a7ul, 0xc19bf174ul, 0xe49b69c1ul, 0xefbe4786ul,
    0x0fc19dc6ul, 0x240ca1ccul, 0x2de92c6ful, 0x4a7484aaul, 0x5cb0a9dcul, 0x76f988daul,
    0x983e5152ul, 0xa831c66dul, 0xb00327c8ul, 0xbf597fc7ul, 0xc6e00bf3ul, 0xd5a79147ul,
    0x06ca6351ul, 0x14292967ul, 0x27b70a85ul, 0x2e1b2138ul, 0x4d2c6dfcul, 0x53380d13ul,
    0x650a7354ul, 0x766a0abbul, 0x81c2c92eul, 0x92722c85ul, 0xa2bfe8a1ul, 0xa81a664bul,
    0xc24b8b70ul, 0xc76c51a3ul, 0xd192e819ul, 0xd6990624ul, 0xf40e3585ul, 0x106aa070ul,
    0x19a4c116ul, 0x1e376c08ul, 0x2748774cul, 0x34b0bcb5ul, 0x391c0cb3ul, 0x4ed8aa4aul,
    0x5b9cca4ful, 0x682e6ff3ul, 0x748f82eeul, 0x78a5636ful, 0x84c87814ul, 0x8cc70208ul,
    0x90befffaul, 0xa4506cebul, 0xbef9a3f7ul, 0xc67178f2ul};

static rayo_u32 rotr(rayo_u32 x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void block(rayo_u32 h[8], const rayo_u8 *p) {
  rayo_u32 w[64], a, b, c, d, e, f, g, k;
  int i;
  for (i = 0; i < 16; i++) {
    w[i] = (rayo_u32)p[4 * i] << 24 | (rayo_u32)p[4 * i + 1] << 16 | (rayo_u32)p[4 * i + 2] << 8 |
           (rayo_u32)p[4 * i + 3];
  }
  for (i = 16; i < 64; i++) {
    rayo_u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    rayo_u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  a = h[0];
  b = h[1];
  c = h[2];
  d = h[3];
  e = h[4];
  f = h[5];
  g = h[6];
  k = h[7];
  for (i = 0; i < 64; i++) {
    rayo_u32 t1 =
        k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k256[i] + w[i];
    rayo_u32 t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += k;
}

void rayo_sha256(const void *data, size_t size, rayo_u8 out[32]) {
  rayo_u32 h[8];
  const rayo_u8 *p = (const rayo_u8 *)data;
  size_t left = size, n, i;
  rayo_u8 tail[128];
  /* The length in bits as two 32-bit words: C89 has no 64-bit type. */
  const rayo_u32 bits_lo = (rayo_u32)(size << 3) & 0xffffffffu;
  const rayo_u32 bits_hi = (rayo_u32)(size >> 29);
  h[0] = 0x6a09e667ul;
  h[1] = 0xbb67ae85ul;
  h[2] = 0x3c6ef372ul;
  h[3] = 0xa54ff53aul;
  h[4] = 0x510e527ful;
  h[5] = 0x9b05688cul;
  h[6] = 0x1f83d9abul;
  h[7] = 0x5be0cd19ul;
  for (; left >= 64; p += 64, left -= 64) block(h, p);
  memset(tail, 0, sizeof tail);
  if (left) memcpy(tail, p, left);
  tail[left] = 0x80;
  n = left < 56 ? 64 : 128;
  for (i = 0; i < 4; i++) {
    tail[n - 1 - i] = (rayo_u8)(bits_lo >> (8u * i));
    tail[n - 5 - i] = (rayo_u8)(bits_hi >> (8u * i));
  }
  block(h, tail);
  if (n == 128) block(h, tail + 64);
  for (i = 0; i < 8; i++) {
    out[4 * i] = (rayo_u8)(h[i] >> 24);
    out[4 * i + 1] = (rayo_u8)(h[i] >> 16);
    out[4 * i + 2] = (rayo_u8)(h[i] >> 8);
    out[4 * i + 3] = (rayo_u8)h[i];
  }
}
