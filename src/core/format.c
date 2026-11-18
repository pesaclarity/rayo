#include "core/format.h"

#include "core/small_begin.h"

typedef struct sink {
  char *out;
  size_t size;
  size_t at;
} sink;

static void put(sink *s, char c) {
  if (s->at + 1 < s->size) s->out[s->at++] = c;
}

static void put_text(sink *s, const char *t) {
  if (!t) t = "(null)";
  while (*t) put(s, *t++);
}

static void put_unsigned(sink *s, unsigned long v) {
  char digits[24];
  int n = 0;
  do {
    digits[n++] = (char)('0' + (int)(v % 10u));
    v /= 10u;
  } while (v && n < 24);
  while (n > 0) put(s, digits[--n]);
}

static void put_signed(sink *s, long v) {
  if (v < 0) {
    put(s, '-');
    put_unsigned(s, 0ul - (unsigned long)v);
  } else {
    put_unsigned(s, (unsigned long)v);
  }
}

void rayo_vformat(char *out, size_t size, const char *fmt, va_list ap) {
  sink s;
  s.out = out;
  s.size = size;
  s.at = 0;
  for (; *fmt; fmt++) {
    int is_long = 0;
    if (*fmt != '%') {
      put(&s, *fmt);
      continue;
    }
    fmt++;
    if (*fmt == 'l') {
      is_long = 1;
      fmt++;
    }
    switch (*fmt) {
      case 's':
        put_text(&s, va_arg(ap, const char *));
        break;
      case 'd':
        put_signed(&s, is_long ? va_arg(ap, long) : (long)va_arg(ap, int));
        break;
      case 'u':
        put_unsigned(&s, is_long ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned));
        break;
      case 'c':
        put(&s, (char)va_arg(ap, int));
        break;
      case '%':
        put(&s, '%');
        break;
      case '\0':
        fmt--;
        break;
      default:
        put(&s, '?');
    }
  }
  if (size) out[s.at] = '\0';
}

#include "core/small_end.h"
