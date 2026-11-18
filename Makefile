# Builds the chat example (examples/chat) and a static library with any C89
# compiler: no other tool, nothing downloaded. CI and tool/check build with
# tool/ci/*.sh; this is for trying rayo by hand (docs/RUNNING-SLMS.md).
#
#   make                         the chat program, for this computer
#   make CC=$NDK/.../aarch64-linux-android24-clang LIBS=-lm   for an Android phone
#
# rayo picks SIMD kernels at run time (SSE4.1, AVX2, AVX-VNNI; NEON, SDOT), so
# the default flags suit any computer of the architecture. -ffp-contract=off
# (no fused multiply-adds the source does not ask for) is always on, whatever
# CFLAGS: it keeps the floats the same as every other build's.

CC ?= cc
# Hardened (docs/SECURITY.md); a compiler other than gcc or clang may need
# CFLAGS=-O2 alone.
CFLAGS ?= -O2 -fstack-protector-strong -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -Wformat \
	-Wformat-security
LIBS ?= -lm -lpthread
RAYO_FLAGS = -std=c89 -ffp-contract=off -Iinclude -Isrc
SOURCES = $(sort $(wildcard src/*.c src/*/*.c))
OBJECTS = $(SOURCES:.c=.o)

all: examples/chat/chat

examples/chat/chat: $(SOURCES) examples/chat/chat.c
	$(CC) $(RAYO_FLAGS) $(CFLAGS) $(SOURCES) examples/chat/chat.c $(LIBS) -o $@

librayo.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o: %.c
	$(CC) $(RAYO_FLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f examples/chat/chat librayo.a $(OBJECTS)

.PHONY: all clean
