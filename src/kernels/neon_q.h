/* The group-wise products (kernels.h, "int4" and "int3") with NEON, for
 * neon.c and neon_dotprod.c to include once each, after arm_neon.h,
 * kernels/half.h and kernels/neon_unpack.h, with:
 *
 * - RAYO_Q_ATTR: the functions' target attribute (or nothing);
 * - RAYO_Q_DOT4(acc, x, y): acc plus, in lane l, the sum of x[4l .. 4l + 3]
 *   times y[4l .. 4l + 3] (int8 by int8, exact in int32);
 * - RAYO_Q_STORED: 1 to unpack the weights as stored (below), 0 to
 *   subtract their offset (where a dot product costs more than that).
 *
 * Four rows of weights at a time, by one or two rows of activations: each
 * group of activations is loaded once for the four rows, each group of
 * weights unpacked once for the two, and the four rows' scales (converted
 * from float16 once a call) are multiplied by the activations' in one
 * vector. Each output's lanes are those of the generic kernel, summed in
 * the same order, so the result is the same bit for bit.
 *
 * With RAYO_Q_STORED, the weights are unpacked as stored (0 to 15, or 0
 * to 7) and the accumulators start from minus 8 (or 4) times the
 * activations' sums of four: the same exact integers as subtracting 8 (or
 * 4) from each weight, with a dot product a group of activations (shared
 * by four rows of weights) instead of a subtraction a group of weights. */
#ifndef RAYO_KERNELS_NEON_Q_H
#define RAYO_KERNELS_NEON_Q_H

/* A group of weights, as stored or minus their offset (RAYO_Q_STORED). */
static RAYO_Q_ATTR RAYO_NEON_INLINE void q_unpack(const rayo_u8 *g, size_t bytes, int8x16_t *wl,
                                                  int8x16_t *wh) {
  if (bytes == RAYO_Q4_GROUP_BYTES) {
    const uint8x16_t b = vld1q_u8(g);
    *wl = vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(15)));
    *wh = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
  } else {
    neon_unpack_q3_stored(g, wl, wh);
  }
#if !RAYO_Q_STORED
  {
    const int8x16_t off = vdupq_n_s8(bytes == RAYO_Q4_GROUP_BYTES ? 8 : 4);
    *wl = vsubq_s8(*wl, off);
    *wh = vsubq_s8(*wh, off);
  }
#endif
}

/* What the accumulators of a group of activations x start from. */
static RAYO_Q_ATTR RAYO_NEON_INLINE int32x4_t q_start(int8x16_t x, size_t bytes) {
#if RAYO_Q_STORED
  return RAYO_Q_DOT4(vdupq_n_s32(0), x, vdupq_n_s8(bytes == RAYO_Q4_GROUP_BYTES ? -8 : -4));
#else
  (void)x;
  (void)bytes;
  return vdupq_n_s32(0);
#endif
}

/* One weight row's group by one row's activations, into its eight lanes. */
static RAYO_Q_ATTR RAYO_NEON_INLINE void q_madd(float32x4_t *lo, float32x4_t *hi, int32x4_t cl,
                                                int32x4_t ch, int8x16_t xl, int8x16_t xh,
                                                int8x16_t wl, int8x16_t wh, float d) {
  float32x4_t t;
  t = vmulq_n_f32(vcvtq_f32_s32(RAYO_Q_DOT4(cl, xl, wl)), d);
  *lo = vaddq_f32(*lo, t);
  t = vmulq_n_f32(vcvtq_f32_s32(RAYO_Q_DOT4(ch, xh, wh)), d);
  *hi = vaddq_f32(*hi, t);
}

static RAYO_Q_ATTR RAYO_NEON_INLINE float q_sum(float32x4_t lo, float32x4_t hi) {
  float lanes[8];
  vst1q_f32(lanes, lo);
  vst1q_f32(lanes + 4, hi);
  return rayo_q4_lanes_sum(lanes);
}

/* Rows a0 (and a1, with `two`) of activations by weight rows w .. w + 3
 * (`stride` bytes apart), with `st` holding the four rows' scales group by
 * group. */
static RAYO_Q_ATTR RAYO_NEON_INLINE void q_block_of(const rayo_i8 *a0, const rayo_i8 *a1,
                                                    const float *as0, const float *as1,
                                                    rayo_bool two, const rayo_u8 *w, size_t stride,
                                                    const float *st, size_t groups, size_t bytes,
                                                    float *c0, float *c1) {
  float32x4_t l00 = vdupq_n_f32(0.0f), h00 = l00, l01 = l00, h01 = l00, l02 = l00, h02 = l00;
  float32x4_t l03 = l00, h03 = l00, l10 = l00, h10 = l00, l11 = l00, h11 = l00, l12 = l00;
  float32x4_t h12 = l00, l13 = l00, h13 = l00;
  size_t g;
  for (g = 0; g < groups; g++) {
    const rayo_u8 *wg = w + g * bytes;
    const int8x16_t xl0 = vld1q_s8(a0 + g * 32), xh0 = vld1q_s8(a0 + g * 32 + 16);
    const int32x4_t cl0 = q_start(xl0, bytes), ch0 = q_start(xh0, bytes);
    const float32x4_t s = vld1q_f32(st + 4 * g);
    const float32x4_t d0 = vmulq_n_f32(s, as0[g]);
    int8x16_t wl, wh;
    if (two) {
      const int8x16_t xl1 = vld1q_s8(a1 + g * 32), xh1 = vld1q_s8(a1 + g * 32 + 16);
      const int32x4_t cl1 = q_start(xl1, bytes), ch1 = q_start(xh1, bytes);
      const float32x4_t d1 = vmulq_n_f32(s, as1[g]);
      q_unpack(wg, bytes, &wl, &wh);
      q_madd(&l00, &h00, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 0));
      q_madd(&l10, &h10, cl1, ch1, xl1, xh1, wl, wh, vgetq_lane_f32(d1, 0));
      q_unpack(wg + stride, bytes, &wl, &wh);
      q_madd(&l01, &h01, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 1));
      q_madd(&l11, &h11, cl1, ch1, xl1, xh1, wl, wh, vgetq_lane_f32(d1, 1));
      q_unpack(wg + 2 * stride, bytes, &wl, &wh);
      q_madd(&l02, &h02, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 2));
      q_madd(&l12, &h12, cl1, ch1, xl1, xh1, wl, wh, vgetq_lane_f32(d1, 2));
      q_unpack(wg + 3 * stride, bytes, &wl, &wh);
      q_madd(&l03, &h03, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 3));
      q_madd(&l13, &h13, cl1, ch1, xl1, xh1, wl, wh, vgetq_lane_f32(d1, 3));
    } else {
      q_unpack(wg, bytes, &wl, &wh);
      q_madd(&l00, &h00, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 0));
      q_unpack(wg + stride, bytes, &wl, &wh);
      q_madd(&l01, &h01, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 1));
      q_unpack(wg + 2 * stride, bytes, &wl, &wh);
      q_madd(&l02, &h02, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 2));
      q_unpack(wg + 3 * stride, bytes, &wl, &wh);
      q_madd(&l03, &h03, cl0, ch0, xl0, xh0, wl, wh, vgetq_lane_f32(d0, 3));
    }
  }
  c0[0] = q_sum(l00, h00);
  c0[1] = q_sum(l01, h01);
  c0[2] = q_sum(l02, h02);
  c0[3] = q_sum(l03, h03);
  if (two) {
    c1[0] = q_sum(l10, h10);
    c1[1] = q_sum(l11, h11);
    c1[2] = q_sum(l12, h12);
    c1[3] = q_sum(l13, h13);
  }
}

/* q_block_of specialised, out of line: 4-bit weights by one row and by
 * two (the decoder's and the prompt's products, each with no test in its
 * loop), and 3-bit weights by either. */
static RAYO_Q_ATTR RAYO_NEON_NOINLINE void q4_block1(const rayo_i8 *a0, const float *as0,
                                                     const rayo_u8 *w, size_t stride,
                                                     const float *st, size_t groups, float *c0) {
  q_block_of(a0, a0, as0, as0, RAYO_FALSE, w, stride, st, groups, RAYO_Q4_GROUP_BYTES, c0, c0);
}

static RAYO_Q_ATTR RAYO_NEON_NOINLINE void q4_block2(const rayo_i8 *a0, const rayo_i8 *a1,
                                                     const float *as0, const float *as1,
                                                     const rayo_u8 *w, size_t stride,
                                                     const float *st, size_t groups, float *c0,
                                                     float *c1) {
  q_block_of(a0, a1, as0, as1, RAYO_TRUE, w, stride, st, groups, RAYO_Q4_GROUP_BYTES, c0, c1);
}

static RAYO_Q_ATTR RAYO_NEON_NOINLINE void q3_block(const rayo_i8 *a0, const rayo_i8 *a1,
                                                    const float *as0, const float *as1,
                                                    rayo_bool two, const rayo_u8 *w, size_t stride,
                                                    const float *st, size_t groups, float *c0,
                                                    float *c1) {
  q_block_of(a0, a1, as0, as1, two, w, stride, st, groups, RAYO_Q3_GROUP_BYTES, c0, c1);
}

static RAYO_Q_ATTR RAYO_NEON_INLINE void q_block(const rayo_i8 *a0, const rayo_i8 *a1,
                                                 const float *as0, const float *as1, rayo_bool two,
                                                 const rayo_u8 *w, size_t stride, const float *st,
                                                 size_t groups, size_t bytes, float *c0,
                                                 float *c1) {
  if (bytes == RAYO_Q3_GROUP_BYTES) {
    q3_block(a0, a1, as0, as1, two, w, stride, st, groups, c0, c1);
  } else if (two) {
    q4_block2(a0, a1, as0, as1, w, stride, st, groups, c0, c1);
  } else {
    q4_block1(a0, as0, w, stride, st, groups, c0);
  }
}

/* One output, for the rows left over and rows of more than RAYO_SCALE_ROW
 * groups: scales from `ws` (float16 with `half`) from group `first`. */
static RAYO_Q_ATTR RAYO_NEON_NOINLINE float q_dot(const rayo_i8 *a, const float *as,
                                                  const rayo_u8 *w, const void *ws, rayo_bool half,
                                                  size_t first, size_t groups, size_t bytes) {
  float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
  size_t g;
  for (g = 0; g < groups; g++) {
    const int8x16_t xl = vld1q_s8(a + g * 32), xh = vld1q_s8(a + g * 32 + 16);
    int8x16_t wl, wh;
    q_unpack(w + g * bytes, bytes, &wl, &wh);
    q_madd(&lo, &hi, q_start(xl, bytes), q_start(xh, bytes), xl, xh, wl, wh,
           as[g] * rayo_group_scale(ws, half, first + g));
  }
  return q_sum(lo, hi);
}

/* Four rows' scales (rows j .. j + 3 of `wscale`), group by group:
 * st[4 * g + r] for row j + r. */
static RAYO_Q_ATTR RAYO_NEON_NOINLINE void q_scales4(const void *wscale, rayo_bool half, size_t j,
                                                     size_t groups, float *row, float *st) {
  size_t r, g;
  for (r = 0; r < 4; r++) {
    const float *s = row;
    if (half) {
      neon_half_row((const rayo_u16 *)wscale + (j + r) * groups, groups, row);
    } else {
      s = (const float *)wscale + (j + r) * groups;
    }
    for (g = 0; g < groups; g++) st[4 * g + r] = s[g];
  }
}

static RAYO_Q_ATTR RAYO_NEON_NOINLINE void q_gemm(const rayo_i8 *RAYO_RESTRICT a,
                                                  const float *RAYO_RESTRICT ascale, size_t m,
                                                  const rayo_u8 *RAYO_RESTRICT w,
                                                  const void *wscale, rayo_bool half, size_t n,
                                                  size_t k, float *RAYO_RESTRICT c, size_t ldc,
                                                  size_t bytes) {
  const size_t groups = k / RAYO_Q4_GROUP, stride = groups * bytes;
  float row[RAYO_SCALE_ROW], st[4 * RAYO_SCALE_ROW];
  size_t i, j = 0;
  if (groups <= RAYO_SCALE_ROW) {
    for (; j + 4 <= n; j += 4) {
      const rayo_u8 *wj = w + j * stride;
      q_scales4(wscale, half, j, groups, row, st);
      for (i = 0; i + 2 <= m; i += 2) {
        q_block(a + i * k, a + (i + 1) * k, ascale + i * groups, ascale + (i + 1) * groups,
                RAYO_TRUE, wj, stride, st, groups, bytes, c + i * ldc + j, c + (i + 1) * ldc + j);
      }
      if (i < m) {
        q_block(a + i * k, a + i * k, ascale + i * groups, ascale + i * groups, RAYO_FALSE, wj,
                stride, st, groups, bytes, c + i * ldc + j, c + i * ldc + j);
      }
    }
  }
  for (; j < n; j++) {
    const rayo_u8 *wj = w + j * stride;
    /* A row's float16 scales converted once, for all its groups and rows. */
    const rayo_bool conv = half && groups <= RAYO_SCALE_ROW;
    const void *ws = conv ? (const void *)row : wscale;
    const rayo_bool h = conv ? RAYO_FALSE : half;
    const size_t first = conv ? 0 : j * groups;
    if (conv) neon_half_row((const rayo_u16 *)wscale + j * groups, groups, row);
    for (i = 0; i < m; i++) {
      c[i * ldc + j] = q_dot(a + i * k, ascale + i * groups, wj, ws, h, first, groups, bytes);
    }
  }
}

#endif
