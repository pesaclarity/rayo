/* Bounded formatting: C89 has no snprintf, so rayo formats its messages
 * itself. %s, %d, %u, %ld, %lu, %c and %% only. */
#ifndef RAYO_CORE_FORMAT_H
#define RAYO_CORE_FORMAT_H

#include <stdarg.h>
#include <stddef.h>

/* Writes at most `size - 1` characters and a NUL into `out` (size > 0). */
void rayo_vformat(char *out, size_t size, const char *fmt, va_list ap);

#endif
