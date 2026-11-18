# Builds the chat example (examples/chat) and a static library with any C89
# compiler: no other tool, nothing downloaded. CI and tool/check build with
# tool/ci/*.sh; this is for trying rayo by hand (docs/RUNNING-SLMS.md).
#
#   make                         the chat program, for this computer
#   make CC=$NDK/.../aarch64-linux-android24-clang LIBS=-lm   for an Android phone
#
# rayo picks SIMD kernels at run time (SSE4.1, AVX2, AVX-VNNI; NEON, SDOT), so
# the default flags suit any computer of the architecture.

CC ?= cc
CFLAGS ?= -O2
LIBS ?= -lm -lpthread
RAYO_FLAGS = -std=c89 -Iinclude -Isrc
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
