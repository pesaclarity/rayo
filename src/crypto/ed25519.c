/* Ed25519 verification. The arithmetic mod p = 2^255 - 19 and mod the
 * group order, and the Edwards curve operations, are Daniel Beer's c25519
 * (2017-10-05; f25519.c, fprime.c, ed25519.c, edsign.c, public domain:
 * LICENSES/c25519.txt), changed for rayo: C89 (no stdint.h, no inline, no
 * designated initialisers), rayo's integer types with explicit
 * conversions, only what verifying needs, SHA-512 streamed from
 * crypto/sha512.c, and strict checks of S and of the encodings. Numbers are
 * 32 little-endian bytes; each operation's comments are c25519's. */
#include "crypto/ed25519.h"

#include <string.h>

#include "core/compat.h"
#include "crypto/sha512.h"

#include "core/small_begin.h"

#define FE_SIZE 32

typedef rayo_u8 fe[FE_SIZE];

/* An extended projective point (X:Y:T:Z), x = X/Z, y = Y/Z, xy = T/Z. */
typedef struct point {
  fe x, y, t, z;
} point;

static const fe fe_one = {1};

/* d = -121665/121666 */
static const fe ed_d = {0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75, 0xab, 0xd8, 0x41,
                        0x41, 0x4d, 0x0a, 0x70, 0x00, 0x98, 0xe8, 0x79, 0x77, 0x79, 0x40,
                        0xc7, 0x8c, 0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52};

/* k = 2d */
static const fe ed_k = {0x59, 0xf1, 0xb2, 0x26, 0x94, 0x9b, 0xd6, 0xeb, 0x56, 0xb1, 0x83,
                        0x82, 0x9a, 0x14, 0xe0, 0x00, 0x30, 0xd1, 0xf3, 0xee, 0xf2, 0x80,
                        0x8e, 0x19, 0xe7, 0xfc, 0xdf, 0x56, 0xdc, 0xd9, 0x06, 0x24};

/* The group's order, L = 2^252 + 27742317777372353535851937790883648493. */
static const fe ed_order = {0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7,
                            0xa2, 0xde, 0xf9, 0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10};

/* The base point B. */
static const point ed_base = {{0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9, 0xb2, 0xa7, 0x25,
                               0x95, 0x60, 0xc7, 0x2c, 0x69, 0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2,
                               0xa4, 0xc0, 0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36, 0x69, 0x21},
                              {0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                               0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                               0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66},
                              {0xa3, 0xdd, 0xb7, 0xa5, 0xb3, 0x8a, 0xde, 0x6d, 0xf5, 0x52, 0x51,
                               0x77, 0x80, 0x9f, 0xf0, 0x20, 0x7d, 0xe3, 0xab, 0x64, 0x8e, 0x4e,
                               0xea, 0x66, 0x65, 0x76, 0x8b, 0xd7, 0x0f, 0x5f, 0x87, 0x67},
                              {1}};

static const point ed_neutral = {{0}, {1}, {0}, {1}};

static rayo_u8 lo8(rayo_u32 v) {
  return (rayo_u8)(v & 255u);
}

/* ---- arithmetic mod p (f25519.c) ------------------------------------------------------------ */

RAYO_SMALL static void fe_load(rayo_u8 *x, rayo_u32 c) {
  int i;
  for (i = 0; i < 4; i++) {
    x[i] = lo8(c);
    c >>= 8;
  }
  for (; i < FE_SIZE; i++) x[i] = 0;
}

/* dst = condition ? one : zero, in constant time. */
RAYO_SMALL static void fe_select(rayo_u8 *dst, const rayo_u8 *zero, const rayo_u8 *one,
                                 rayo_u8 condition) {
  const rayo_u8 mask = (rayo_u8)(0u - (unsigned)condition);
  int i;
  for (i = 0; i < FE_SIZE; i++) dst[i] = (rayo_u8)(zero[i] ^ (mask & (one[i] ^ zero[i])));
}

RAYO_SMALL static void fe_normalize(rayo_u8 *x) {
  rayo_u8 minusp[FE_SIZE];
  rayo_u32 c;
  int i;
  /* Reduce using 2^255 = 19 mod p */
  c = (rayo_u32)(x[31] >> 7) * 19u;
  x[31] &= 127;
  for (i = 0; i < FE_SIZE; i++) {
    c += x[i];
    x[i] = lo8(c);
    c >>= 8;
  }
  /* The number is now less than 2^255 + 18, and therefore less than 2p.
   * Try subtracting p, and conditionally load the subtracted value if
   * underflow did not occur. */
  c = 19;
  for (i = 0; i + 1 < FE_SIZE; i++) {
    c += x[i];
    minusp[i] = lo8(c);
    c >>= 8;
  }
  c = (c + (rayo_u32)x[i] - 128u) & 0xffffu; /* c25519's 16-bit arithmetic */
  minusp[31] = lo8(c);
  /* Load x-p if no underflow */
  fe_select(x, minusp, x, (rayo_u8)((c >> 15) & 1u));
}

RAYO_SMALL static rayo_u8 fe_eq(const rayo_u8 *x, const rayo_u8 *y) {
  unsigned sum = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) sum |= (unsigned)(x[i] ^ y[i]);
  sum |= sum >> 4;
  sum |= sum >> 2;
  sum |= sum >> 1;
  return (rayo_u8)((sum ^ 1u) & 1u);
}

/* Reduce with 2^255 = 19 mod p: r[31]'s top bit and the carry `c` (the
 * bits above 2^255) folded back in. */
RAYO_SMALL static void fe_fold(rayo_u8 *r, rayo_u32 c) {
  int i;
  r[31] &= 127;
  c = (c >> 7) * 19u;
  for (i = 0; i < FE_SIZE; i++) {
    c += r[i];
    r[i] = lo8(c);
    c >>= 8;
  }
}

RAYO_SMALL static void fe_add(rayo_u8 *r, const rayo_u8 *a, const rayo_u8 *b) {
  rayo_u32 c = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) {
    c >>= 8;
    c += (rayo_u32)a[i] + (rayo_u32)b[i];
    r[i] = lo8(c);
  }
  fe_fold(r, c);
}

RAYO_SMALL static void fe_sub(rayo_u8 *r, const rayo_u8 *a, const rayo_u8 *b) {
  rayo_u32 c;
  int i;
  /* Calculate a + 2p - b, to avoid underflow */
  c = 218;
  for (i = 0; i + 1 < FE_SIZE; i++) {
    c += 65280u + (rayo_u32)a[i] - (rayo_u32)b[i];
    r[i] = lo8(c);
    c >>= 8;
  }
  c += (rayo_u32)a[31] - (rayo_u32)b[31];
  r[31] = (rayo_u8)(c & 127u);
  c = (c >> 7) * 19u;
  for (i = 0; i < FE_SIZE; i++) {
    c += r[i];
    r[i] = lo8(c);
    c >>= 8;
  }
}

RAYO_SMALL static void fe_neg(rayo_u8 *r, const rayo_u8 *a) {
  rayo_u32 c;
  int i;
  /* Calculate 2p - a, to avoid underflow */
  c = 218;
  for (i = 0; i + 1 < FE_SIZE; i++) {
    c += 65280u - (rayo_u32)a[i];
    r[i] = lo8(c);
    c >>= 8;
  }
  c -= (rayo_u32)a[31];
  r[31] = (rayo_u8)(c & 127u);
  c = (c >> 7) * 19u;
  for (i = 0; i < FE_SIZE; i++) {
    c += r[i];
    r[i] = lo8(c);
    c >>= 8;
  }
}

/* r = a * b; r must not be a or b. */
RAYO_SMALL static void fe_mul(rayo_u8 *r, const rayo_u8 *a, const rayo_u8 *b) {
  rayo_u32 c = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) {
    int j;
    c >>= 8;
    for (j = 0; j <= i; j++) c += (rayo_u32)a[j] * (rayo_u32)b[i - j];
    for (; j < FE_SIZE; j++) c += (rayo_u32)a[j] * (rayo_u32)b[i + FE_SIZE - j] * 38u;
    r[i] = lo8(c);
  }
  fe_fold(r, c);
}

RAYO_SMALL static void fe_mul_c(rayo_u8 *r, const rayo_u8 *a, rayo_u32 b) {
  rayo_u32 c = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) {
    c >>= 8;
    c += b * (rayo_u32)a[i];
    r[i] = lo8(c);
  }
  fe_fold(r, c);
}

/* r = 1 / x (x^(p-2), Fermat); r must not be x. */
RAYO_SMALL static void fe_inv(rayo_u8 *r, const rayo_u8 *x) {
  fe s;
  int i;
  /* 1 1 */
  fe_mul(s, x, x);
  fe_mul(r, s, x);
  /* 1 x 248 */
  for (i = 0; i < 248; i++) {
    fe_mul(s, r, r);
    fe_mul(r, s, x);
  }
  /* 0 */
  fe_mul(s, r, r);
  /* 1 */
  fe_mul(r, s, s);
  fe_mul(s, r, x);
  /* 0 */
  fe_mul(r, s, s);
  /* 1 */
  fe_mul(s, r, r);
  fe_mul(r, s, x);
  /* 1 */
  fe_mul(s, r, r);
  fe_mul(r, s, x);
}

/* r = x^((p-5)/8) = x^(2^252-3), with s for temporary storage. */
RAYO_SMALL static void exp2523(rayo_u8 *r, const rayo_u8 *x, rayo_u8 *s) {
  int i;
  /* 1 1 */
  fe_mul(r, x, x);
  fe_mul(s, r, x);
  /* 1 x 248 */
  for (i = 0; i < 248; i++) {
    fe_mul(r, s, s);
    fe_mul(s, r, x);
  }
  /* 0 */
  fe_mul(r, s, s);
  /* 1 */
  fe_mul(s, r, r);
  fe_mul(r, s, x);
}

RAYO_SMALL static void fe_sqrt(rayo_u8 *r, const rayo_u8 *a) {
  fe v, i, x, y;
  /* v = (2a)^((p-5)/8) [x = 2a] */
  fe_mul_c(x, a, 2);
  exp2523(v, x, y);
  /* i = 2av^2 - 1 */
  fe_mul(y, v, v);
  fe_mul(i, x, y);
  fe_load(y, 1);
  fe_sub(i, i, y);
  /* r = avi */
  fe_mul(x, v, a);
  fe_mul(r, x, i);
}

/* ---- arithmetic mod L (fprime.c) ------------------------------------------------------------ */

RAYO_SMALL static void raw_try_sub(rayo_u8 *x, const rayo_u8 *p) {
  rayo_u8 minusp[FE_SIZE];
  rayo_u32 c = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) {
    c = ((rayo_u32)x[i] - (rayo_u32)p[i] - c) & 0xffffu;
    minusp[i] = lo8(c);
    c = (c >> 8) & 1u;
  }
  fe_select(x, minusp, x, (rayo_u8)c);
}

RAYO_SMALL static void shift_n_bits(rayo_u8 *x, int n) {
  rayo_u32 c = 0;
  int i;
  for (i = 0; i < FE_SIZE; i++) {
    c |= (rayo_u32)x[i] << n;
    x[i] = lo8(c);
    c >>= 8;
  }
}

/* n = the `len` bytes at x (little-endian) mod L. L's top bit is bit 252. */
RAYO_SMALL static void reduce_order(rayo_u8 *n, const rayo_u8 *x, size_t len) {
  const int msb = 252;
  const int total = (int)len * 8;
  const int preload_total = msb - 1 < total ? msb - 1 : total;
  const int preload_bytes = preload_total >> 3;
  const int preload_bits = preload_total & 7;
  const int rbits = total - preload_total;
  int i;
  memset(n, 0, FE_SIZE);
  for (i = 0; i < preload_bytes; i++) n[i] = x[len - (size_t)preload_bytes + (size_t)i];
  if (preload_bits) {
    shift_n_bits(n, preload_bits);
    n[0] = (rayo_u8)(n[0] | (x[len - (size_t)preload_bytes - 1] >> (8 - preload_bits)));
  }
  for (i = rbits - 1; i >= 0; i--) {
    const rayo_u8 bit = (rayo_u8)((x[i >> 3] >> (i & 7)) & 1);
    shift_n_bits(n, 1);
    n[0] = (rayo_u8)(n[0] | bit);
    raw_try_sub(n, ed_order);
  }
}

/* ---- the curve (ed25519.c) ------------------------------------------------------------------ */

RAYO_SMALL static void ed_project(point *p, const rayo_u8 *x, const rayo_u8 *y) {
  memcpy(p->x, x, FE_SIZE);
  memcpy(p->y, y, FE_SIZE);
  fe_load(p->z, 1);
  fe_mul(p->t, x, y);
}

/* The encoding of p: y, and x's parity in the top bit. */
RAYO_SMALL static void ed_pack(rayo_u8 *c, const point *p) {
  fe z1, x, y;
  fe_inv(z1, p->z);
  fe_mul(x, p->x, z1);
  fe_mul(y, p->y, z1);
  fe_normalize(x);
  fe_normalize(y);
  memcpy(c, y, FE_SIZE);
  c[31] = (rayo_u8)(c[31] | ((x[0] & 1) << 7));
}

/* The point encoded at `comp`, if it is a canonical encoding of one:
 * x = sqrt((y^2-1) / (1+dy^2)). */
RAYO_SMALL static rayo_bool ed_unpack(point *p, const rayo_u8 *comp) {
  const int parity = comp[31] >> 7;
  fe x, y, a, b, c;
  rayo_u8 ok;
  /* Unpack y */
  memcpy(y, comp, FE_SIZE);
  y[31] &= 127;
  /* Canonical: y < p (y normalised is y). */
  memcpy(a, y, FE_SIZE);
  fe_normalize(a);
  ok = fe_eq(a, y);
  /* Compute c = y^2 */
  fe_mul(c, y, y);
  /* Compute b = (1+dy^2)^-1 */
  fe_mul(b, c, ed_d);
  fe_add(a, b, fe_one);
  fe_inv(b, a);
  /* Compute a = y^2-1 */
  fe_sub(a, c, fe_one);
  /* Compute c = a*b = (y^2-1)/(1-dy^2) */
  fe_mul(c, a, b);
  /* Compute a, b = +/-sqrt(c), if c is square */
  fe_sqrt(a, c);
  fe_neg(b, a);
  /* Select one of them, based on the compressed parity bit */
  fe_select(x, a, b, (rayo_u8)((a[0] ^ parity) & 1));
  /* Verify that x^2 = c */
  fe_mul(a, x, x);
  fe_normalize(a);
  fe_normalize(c);
  ok &= fe_eq(a, c);
  /* x = 0 has no negative: an encoding of it with the sign bit set is not
   * canonical. */
  fe_normalize(x);
  memset(a, 0, FE_SIZE);
  if (parity && fe_eq(x, a)) ok = 0;
  ed_project(p, x, y);
  return ok != 0;
}

/* r = p1 + p2: add-2008-hwcd-3 (Hisil, Wong, Carter, Dawson). */
RAYO_SMALL static void ed_add(point *r, const point *p1, const point *p2) {
  fe a, b, c, d, e, f, g, h;
  /* A = (Y1-X1)(Y2-X2) */
  fe_sub(c, p1->y, p1->x);
  fe_sub(d, p2->y, p2->x);
  fe_mul(a, c, d);
  /* B = (Y1+X1)(Y2+X2) */
  fe_add(c, p1->y, p1->x);
  fe_add(d, p2->y, p2->x);
  fe_mul(b, c, d);
  /* C = T1 k T2 */
  fe_mul(d, p1->t, p2->t);
  fe_mul(c, d, ed_k);
  /* D = Z1 2 Z2 */
  fe_mul(d, p1->z, p2->z);
  fe_add(d, d, d);
  /* E = B - A, F = D - C, G = D + C, H = B + A */
  fe_sub(e, b, a);
  fe_sub(f, d, c);
  fe_add(g, d, c);
  fe_add(h, b, a);
  /* X3 = E F, Y3 = G H, T3 = E H, Z3 = F G */
  fe_mul(r->x, e, f);
  fe_mul(r->y, g, h);
  fe_mul(r->t, e, h);
  fe_mul(r->z, f, g);
}

/* r = 2p: dbl-2008-hwcd. */
RAYO_SMALL static void ed_double(point *r, const point *p) {
  fe a, b, c, e, f, g, h;
  /* A = X1^2, B = Y1^2, C = 2 Z1^2 */
  fe_mul(a, p->x, p->x);
  fe_mul(b, p->y, p->y);
  fe_mul(c, p->z, p->z);
  fe_add(c, c, c);
  /* E = (X1+Y1)^2-A-B */
  fe_add(f, p->x, p->y);
  fe_mul(e, f, f);
  fe_sub(e, e, a);
  fe_sub(e, e, b);
  /* G = D + B (D = -A) */
  fe_sub(g, b, a);
  /* F = G - C */
  fe_sub(f, g, c);
  /* H = D - B */
  fe_neg(h, b);
  fe_sub(h, h, a);
  /* X3 = E F, Y3 = G H, T3 = E H, Z3 = F G */
  fe_mul(r->x, e, f);
  fe_mul(r->y, g, h);
  fe_mul(r->t, e, h);
  fe_mul(r->z, f, g);
}

/* r = e p, e a 256-bit little-endian number. */
RAYO_SMALL static void ed_smult(point *r_out, const point *p, const rayo_u8 *e) {
  point r;
  int i;
  r = ed_neutral;
  for (i = 255; i >= 0; i--) {
    const rayo_u8 bit = (rayo_u8)((e[i >> 3] >> (i & 7)) & 1);
    point s;
    ed_double(&r, &r);
    ed_add(&s, &r, p);
    fe_select(r.x, r.x, s.x, bit);
    fe_select(r.y, r.y, s.y, bit);
    fe_select(r.z, r.z, s.z, bit);
    fe_select(r.t, r.t, s.t, bit);
  }
  *r_out = r;
}

/* ---- verification (edsign.c) ---------------------------------------------------------------- */

/* Whether the 32 bytes at s, little-endian, are below L. */
RAYO_SMALL static rayo_bool below_order(const rayo_u8 *s) {
  int i;
  for (i = FE_SIZE - 1; i >= 0; i--) {
    if (s[i] != ed_order[i]) return s[i] < ed_order[i];
  }
  return RAYO_FALSE;
}

RAYO_SMALL rayo_bool rayo_ed25519_verify(const rayo_u8 signature[64], const rayo_u8 public_key[32],
                                         const void *message, size_t size) {
  rayo_sha512 h;
  rayo_u8 digest[64];
  fe z, lhs, rhs;
  point p, q;
  rayo_bool ok;
  if (!below_order(signature + 32)) return RAYO_FALSE;
  /* z = H(R, A, M) mod L */
  rayo_sha512_init(&h);
  rayo_sha512_update(&h, signature, 32);
  rayo_sha512_update(&h, public_key, 32);
  rayo_sha512_update(&h, message, size);
  rayo_sha512_final(&h, digest);
  reduce_order(z, digest, sizeof digest);
  /* sB = (ze + k)B = ... */
  ed_smult(&p, &ed_base, signature + 32);
  ed_pack(lhs, &p);
  /* ... = zA + R */
  ok = ed_unpack(&p, public_key);
  ed_smult(&p, &p, z);
  ok = ed_unpack(&q, signature) && ok;
  ed_add(&p, &p, &q);
  ed_pack(rhs, &p);
  /* Equal? */
  return ok && fe_eq(lhs, rhs);
}

#include "core/small_end.h"
