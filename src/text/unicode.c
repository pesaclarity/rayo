#include "text/unicode.h"

static rayo_bool in(const rayo_unicode_range *r, size_t n, rayo_u32 c) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (c < r[mid].first) {
      hi = mid;
    } else if (c > r[mid].last) {
      lo = mid + 1;
    } else {
      return RAYO_TRUE;
    }
  }
  return RAYO_FALSE;
}

rayo_bool rayo_unicode_is_letter(rayo_u32 c) {
  if (c < 128) return (c | 32u) >= 'a' && (c | 32u) <= 'z';
  return in(rayo_unicode_letters, rayo_unicode_letters_count, c);
}

rayo_bool rayo_unicode_is_number(rayo_u32 c) {
  if (c < 128) return c >= '0' && c <= '9';
  return in(rayo_unicode_numbers, rayo_unicode_numbers_count, c);
}

/* White space as the regex engines of tokenizers read \s: Unicode's
 * White_Space property. */
rayo_bool rayo_unicode_is_space(rayo_u32 c) {
  return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

rayo_u32 rayo_utf8_next(const unsigned char *s, size_t n, size_t *len) {
  const rayo_u32 b = s[0];
  rayo_u32 c;
  size_t need, i;
  if (b < 0x80) {
    *len = 1;
    return b;
  }
  if (b >= 0xC2 && b <= 0xDF) {
    need = 2;
    c = b & 0x1Fu;
  } else if (b >= 0xE0 && b <= 0xEF) {
    need = 3;
    c = b & 0x0Fu;
  } else if (b >= 0xF0 && b <= 0xF4) {
    need = 4;
    c = b & 0x07u;
  } else {
    *len = 1;
    return b;
  }
  if (n < need) {
    *len = 1;
    return b;
  }
  for (i = 1; i < need; i++) {
    if ((s[i] & 0xC0u) != 0x80u) {
      *len = 1;
      return b;
    }
    c = c << 6 | (s[i] & 0x3Fu);
  }
  /* Overlong forms, surrogates and values past U+10FFFF are not UTF-8. */
  if ((need == 3 && (c < 0x800 || (c >= 0xD800 && c <= 0xDFFF))) ||
      (need == 4 && (c < 0x10000 || c > 0x10FFFF))) {
    *len = 1;
    return b;
  }
  *len = need;
  return c;
}
