// Copyright (c) 2013-2018. DIKU, University of Copenhagen
/* Crash and burn. */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#define BFAST_ASSERT(x) do { \
  if (!(x)) { \
    panic(-1, "%s:%d: Assertion failed:\n\t%s\n", __FILE__, __LINE__, #x); \
  } } while(0);

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
__attribute__ ((unused)) static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char *)malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}
