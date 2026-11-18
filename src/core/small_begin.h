/* Included after a file's other includes, with core/small_end.h at its end:
 * every function between them is compiled for size (clang's minsize, as
 * -Oz), whatever the build's -O. For code that runs once a load, such as
 * the parsers and checks of untrusted files, where size matters on a phone
 * (CI's Android budget) and speed does not. Not a header guard: it is
 * meant to be included once per file. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#endif
