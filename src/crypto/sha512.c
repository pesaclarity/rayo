#include "crypto/sha512.h"

#include <string.h>

#include "core/small_begin.h"

/* The round constants, each high half then low half. */
static const rayo_u32 k512[160] = {
    0x428a2f98ul, 0xd728ae22ul, 0x71374491ul, 0x23ef65cdul, 0xb5c0fbcful, 0xec4d3b2ful,
    0xe9b5dba5ul, 0x8189dbbcul, 0x3956c25bul, 0xf348b538ul, 0x59f111f1ul, 0xb605d019ul,
    0x923f82a4ul, 0xaf194f9bul, 0xab1c5ed5ul, 0xda6d8118ul, 0xd807aa98ul, 0xa3030242ul,
    0x12835b01ul, 0x45706fbeul, 0x243185beul, 0x4ee4b28cul, 0x550c7dc3ul, 0xd5ffb4e2ul,
    0x72be5d74ul, 0xf27b896ful, 0x80deb1feul, 0x3b1696b1ul, 0x9bdc06a7ul, 0x25c71235ul,
    0xc19bf174ul, 0xcf692694ul, 0xe49b69c1ul, 0x9ef14ad2ul, 0xefbe4786ul, 0x384f25e3ul,
    0x0fc19dc6ul, 0x8b8cd5b5ul, 0x240ca1ccul, 0x77ac9c65ul, 0x2de92c6ful, 0x592b0275ul,
    0x4a7484aaul, 0x6ea6e483ul, 0x5cb0a9dcul, 0xbd41fbd4ul, 0x76f988daul, 0x831153b5ul,
    0x983e5152ul, 0xee66dfabul, 0xa831c66dul, 0x2db43210ul, 0xb00327c8ul, 0x98fb213ful,
    0xbf597fc7ul, 0xbeef0ee4ul, 0xc6e00bf3ul, 0x3da88fc2ul, 0xd5a79147ul, 0x930aa725ul,
    0x06ca6351ul, 0xe003826ful, 0x14292967ul, 0x0a0e6e70ul, 0x27b70a85ul, 0x46d22ffcul,
    0x2e1b2138ul, 0x5c26c926ul, 0x4d2c6dfcul, 0x5ac42aedul, 0x53380d13ul, 0x9d95b3dful,
    0x650a7354ul, 0x8baf63deul, 0x766a0abbul, 0x3c77b2a8ul, 0x81c2c92eul, 0x47edaee6ul,
    0x92722c85ul, 0x1482353bul, 0xa2bfe8a1ul, 0x4cf10364ul, 0xa81a664bul, 0xbc423001ul,
    0xc24b8b70ul, 0xd0f89791ul, 0xc76c51a3ul, 0x0654be30ul, 0xd192e819ul, 0xd6ef5218ul,
    0xd6990624ul, 0x5565a910ul, 0xf40e3585ul, 0x5771202aul, 0x106aa070ul, 0x32bbd1b8ul,
    0x19a4c116ul, 0xb8d2d0c8ul, 0x1e376c08ul, 0x5141ab53ul, 0x2748774cul, 0xdf8eeb99ul,
    0x34b0bcb5ul, 0xe19b48a8ul, 0x391c0cb3ul, 0xc5c95a63ul, 0x4ed8aa4aul, 0xe3418acbul,
    0x5b9cca4ful, 0x7763e373ul, 0x682e6ff3ul, 0xd6b2b8a3ul, 0x748f82eeul, 0x5defb2fcul,
    0x78a5636ful, 0x43172f60ul, 0x84c87814ul, 0xa1f0ab72ul, 0x8cc70208ul, 0x1a6439ecul,
    0x90befffaul, 0x23631e28ul, 0xa4506cebul, 0xde82bde9ul, 0xbef9a3f7ul, 0xb2c67915ul,
    0xc67178f2ul, 0xe372532bul, 0xca273eceul, 0xea26619cul, 0xd186b8c7ul, 0x21c0c207ul,
    0xeada7dd6ul, 0xcde0eb1eul, 0xf57d4f7ful, 0xee6ed178ul, 0x06f067aaul, 0x72176fbaul,
    0x0a637dc5ul, 0xa2c898a6ul, 0x113f9804ul, 0xbef90daeul, 0x1b710b35ul, 0x131c471bul,
    0x28db77f5ul, 0x23047d84ul, 0x32caab7bul, 0x40c72493ul, 0x3c9ebe0aul, 0x15c9bebcul,
    0x431d67c4ul, 0x9c100d4cul, 0x4cc5d4beul, 0xcb3e42b6ul, 0x597f299cul, 0xfc657e2aul,
    0x5fcb6fabul, 0x3ad6faecul, 0x6c44198cul, 0x4a475817ul};

static const rayo_u32 h512[16] = {0x6a09e667ul, 0xf3bcc908ul, 0xbb67ae85ul, 0x84caa73bul,
                                  0x3c6ef372ul, 0xfe94f82bul, 0xa54ff53aul, 0x5f1d36f1ul,
                                  0x510e527ful, 0xade682d1ul, 0x9b05688cul, 0x2b3e6c1ful,
                                  0x1f83d9abul, 0xfb41bd6bul, 0x5be0cd19ul, 0x137e2179ul};

/* A 64-bit word as its halves. */
typedef struct w64 {
  rayo_u32 hi, lo;
} w64;

static w64 add(w64 a, w64 b) {
  w64 r;
  r.lo = (a.lo + b.lo) & 0xffffffffu;
  r.hi = (a.hi + b.hi + (r.lo < a.lo ? 1u : 0u)) & 0xffffffffu;
  return r;
}

static w64 xor3(w64 a, w64 b, w64 c) {
  w64 r;
  r.hi = a.hi ^ b.hi ^ c.hi;
  r.lo = a.lo ^ b.lo ^ c.lo;
  return r;
}

/* Rotated right by n, 0 < n < 64, n != 32. */
static w64 rotr(w64 x, unsigned n) {
  w64 r;
  if (n > 32) {
    const rayo_u32 t = x.hi;
    x.hi = x.lo;
    x.lo = t;
    n -= 32;
  }
  r.lo = ((x.lo >> n) | (x.hi << (32u - n))) & 0xffffffffu;
  r.hi = ((x.hi >> n) | (x.lo << (32u - n))) & 0xffffffffu;
  return r;
}

/* Shifted right by n, 0 < n < 32. */
static w64 shr(w64 x, unsigned n) {
  w64 r;
  r.lo = ((x.lo >> n) | (x.hi << (32u - n))) & 0xffffffffu;
  r.hi = x.hi >> n;
  return r;
}

static w64 at(const rayo_u32 *p, int i) {
  w64 r;
  r.hi = p[2 * i];
  r.lo = p[2 * i + 1];
  return r;
}

static rayo_u32 load32(const rayo_u8 *p) {
  return (rayo_u32)p[0] << 24 | (rayo_u32)p[1] << 16 | (rayo_u32)p[2] << 8 | (rayo_u32)p[3];
}

static void compress(rayo_u32 h[16], const rayo_u8 *p) {
  w64 w[80], v[8];
  int i;
  for (i = 0; i < 16; i++) {
    w[i].hi = load32(p + 8 * i);
    w[i].lo = load32(p + 8 * i + 4);
  }
  for (i = 16; i < 80; i++) {
    const w64 s0 = xor3(rotr(w[i - 15], 1), rotr(w[i - 15], 8), shr(w[i - 15], 7));
    const w64 s1 = xor3(rotr(w[i - 2], 19), rotr(w[i - 2], 61), shr(w[i - 2], 6));
    w[i] = add(add(w[i - 16], s0), add(w[i - 7], s1));
  }
  for (i = 0; i < 8; i++) v[i] = at(h, i);
  for (i = 0; i < 80; i++) {
    w64 ch, maj, t1, t2;
    ch.hi = (v[4].hi & v[5].hi) ^ (~v[4].hi & v[6].hi);
    ch.lo = (v[4].lo & v[5].lo) ^ (~v[4].lo & v[6].lo);
    maj.hi = (v[0].hi & v[1].hi) ^ (v[0].hi & v[2].hi) ^ (v[1].hi & v[2].hi);
    maj.lo = (v[0].lo & v[1].lo) ^ (v[0].lo & v[2].lo) ^ (v[1].lo & v[2].lo);
    t1 = add(
        add(add(v[7], xor3(rotr(v[4], 14), rotr(v[4], 18), rotr(v[4], 41))), add(ch, at(k512, i))),
        w[i]);
    t2 = add(xor3(rotr(v[0], 28), rotr(v[0], 34), rotr(v[0], 39)), maj);
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = add(v[3], t1);
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = add(t1, t2);
  }
  for (i = 0; i < 8; i++) {
    const w64 s = add(at(h, i), v[i]);
    h[2 * i] = s.hi;
    h[2 * i + 1] = s.lo;
  }
}

void rayo_sha512_init(rayo_sha512 *s) {
  memcpy(s->h, h512, sizeof s->h);
  s->fill = 0;
  s->bytes_lo = s->bytes_hi = 0;
}

void rayo_sha512_update(rayo_sha512 *s, const void *data, size_t size) {
  const rayo_u8 *p = (const rayo_u8 *)data;
  while (size > 0) {
    size_t n = 128 - s->fill;
    if (n > size) n = size;
    if (s->fill == 0 && size >= 128) {
      compress(s->h, p);
      n = 128;
    } else {
      memcpy(s->block + s->fill, p, n);
      s->fill += n;
      if (s->fill == 128) {
        compress(s->h, s->block);
        s->fill = 0;
      }
    }
    s->bytes_lo = (s->bytes_lo + (rayo_u32)n) & 0xffffffffu;
    if (s->bytes_lo < (rayo_u32)n) s->bytes_hi = (s->bytes_hi + 1u) & 0xffffffffu;
    p += n;
    size -= n;
  }
}

void rayo_sha512_final(rayo_sha512 *s, rayo_u8 out[64]) {
  /* The length in bits: the bytes times 8, in the last 16 bytes. */
  const rayo_u32 bits_lo = (s->bytes_lo << 3) & 0xffffffffu;
  const rayo_u32 bits_hi = ((s->bytes_hi << 3) | (s->bytes_lo >> 29)) & 0xffffffffu;
  const rayo_u32 bits_top = s->bytes_hi >> 29;
  int i;
  s->block[s->fill++] = 0x80;
  if (s->fill > 112) {
    memset(s->block + s->fill, 0, 128 - s->fill);
    compress(s->h, s->block);
    s->fill = 0;
  }
  memset(s->block + s->fill, 0, 128 - s->fill);
  for (i = 0; i < 4; i++) {
    s->block[119 - i] = (rayo_u8)((bits_top >> (8 * i)) & 255u);
    s->block[123 - i] = (rayo_u8)((bits_hi >> (8 * i)) & 255u);
    s->block[127 - i] = (rayo_u8)((bits_lo >> (8 * i)) & 255u);
  }
  compress(s->h, s->block);
  for (i = 0; i < 16; i++) {
    out[4 * i] = (rayo_u8)((s->h[i] >> 24) & 255u);
    out[4 * i + 1] = (rayo_u8)((s->h[i] >> 16) & 255u);
    out[4 * i + 2] = (rayo_u8)((s->h[i] >> 8) & 255u);
    out[4 * i + 3] = (rayo_u8)(s->h[i] & 255u);
  }
}

#include "core/small_end.h"
