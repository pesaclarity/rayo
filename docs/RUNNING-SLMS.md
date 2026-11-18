# Running small language models with rayo

rayo runs decoder-only language models of about 100M to 1.5B parameters on the CPU, offline:
the weights mapped from the file in 4-bit groups, a key/value cache, a few threads on the
fast cores. No GPU, no Google library, nothing downloaded at run time. This page goes from a
Hugging Face model to a chat on an Android phone, then on a computer.

What you need: a computer with `git`, a C compiler, `make` and Python 3.10 or later (for the
converter, which needs NumPy and jinja2 only: no torch, no TensorFlow), and for a phone the
Android NDK and `adb`.

## Models

Models whose licences let you run and ship them, and that rayo-convert reads (Llama-style
decoders):

| Model | Licence | Parameters | q4 file | Memory at 2048 tokens (estimate) |
| --- | --- | --- | --- | --- |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 | 135M | 80 MB | about 95 MB in use, up to 190 MB with a full cache |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache-2.0 | 362M | about 226 MB (estimate) | about 260 MB, up to 420 MB (estimates) |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | 494M | 295 MB | about 310 MB, up to 360 MB |

The memory is the mapped weights the model reads (all of them, every token) plus the cache
for the tokens so far: `2 x layers x kv_heads x head_dim x 4 bytes` a token (SmolLM2-135M:
46 KB, Qwen2.5-0.5B: 24 KB), plus about 1 MB of working memory. The 135M and 0.5B figures were
measured with models of their shapes and random weights (`tool/make_synthetic_slm.py`); the
real files are the same size. A phone with 2 to 3 GB runs the first and the third; prefer
`--context 1024` on the smallest phones.

## On an Android phone

Works on any arm64 or armv7 phone with Android 7 or later, with or without Google Play
services (Huawei and HarmonyOS phones, de-Googled ROMs): rayo uses only the C library and the
kernel. It picks NEON, and the ARMv8.2 dot product (SDOT) where the CPU has it, at run time.

1. **Get rayo and the converter** (on your computer):

   ```sh
   git clone https://github.com/pesaclarity/rayo.git && cd rayo
   git checkout feat/slm-spike
   python3 -m venv .venv && . .venv/bin/activate      # or: uv venv && . .venv/bin/activate
   pip install "./convert[chat]"                       # or: uv pip install "./convert[chat]"
   ```

2. **Download a model** from Hugging Face (about 270 MB for SmolLM2-135M, 1 GB for
   Qwen2.5-0.5B):

   ```sh
   m=HuggingFaceTB/SmolLM2-135M-Instruct              # or Qwen/Qwen2.5-0.5B-Instruct
   mkdir -p models/src && cd models/src
   for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
            model.safetensors; do
     curl -fLO "https://huggingface.co/$m/resolve/main/$f"
   done
   cd ../..
   ```

3. **Convert it** to 4-bit weights:

   ```sh
   rayo-convert models/src -o models/smollm2.weights --weights q4 --context 2048
   ```

   This writes `models/smollm2.weights` and `models/smollm2.tokenizer.json`. The downloaded
   files are no longer needed: `rm -r models/src`.

4. **Build the chat program for the phone** with the NDK (`$NDK` is its directory, such as
   `~/Android/Sdk/ndk/28.2.13676358`; `HOST` is `linux-x86_64` or `darwin-x86_64`):

   ```sh
   bin=$NDK/toolchains/llvm/prebuilt/$HOST/bin
   make CC=$bin/aarch64-linux-android24-clang LIBS=-lm                          # arm64
   # 32-bit phones: make CC=$bin/armv7a-linux-androideabi24-clang CFLAGS="-O2 -mfpu=neon" LIBS=-lm
   ```

5. **Copy it over and chat** (USB debugging on):

   ```sh
   adb push examples/chat/chat models/smollm2.weights models/smollm2.tokenizer.json \
     /data/local/tmp/
   adb shell
   cd /data/local/tmp && chmod +x chat && ./chat smollm2.weights
   ```

   Type a line and press Enter; the reply streams. `/reset` starts again, `/quit` stops, and
   on the way out it prints the prompt and reply speeds and the peak memory.

**Threads.** By default the program runs on the phone's big cores (read from
`/sys/devices/system/cpu/*/cpu_capacity`, or their highest clocks), at most 4: the little
cores add little, since decoding is bound by memory bandwidth, and cost battery. On a phone
that gets hot, or to leave room for the app, use `--threads 2`. The answer is the same with
any number of threads, bit for bit.

**Other options:** `--temp 0` for greedy replies, `--temp`, `--top-k`, `--top-p`, `--min-p`,
`--repeat-penalty`, `--seed`, `--max-tokens`, `--system "..."`, and `-p "text"` to continue a
raw prompt instead of chatting. The quantisation is chosen when converting (`--weights q4`,
`int8` or `float32`), and the longest context too (`--context`).

### In an app, through Dart

The rayo Dart package builds the same C code into the app (a build hook, from source, so an
F-Droid build has no prebuilt library) and gives the generator to Dart:

```dart
import 'package:rayo/rayo.dart';

final model = RayoModel.map('$dir/smollm2.weights');       // mapped, not copied
final tokenizer = RayoTokenizer.load('$dir/smollm2.tokenizer.json');
final chat = RayoGenerator(model, tokenizer, temperature: 0.7, maxTokens: 200);
chat.chat('How do I save money on transport?');
for (final piece in chat.reply()) {
  // show piece; break (or chat.cancel()) to stop early
}
chat.close(); tokenizer.close(); model.close();
```

Run it in a background isolate (`Isolate.run`, or a long-lived isolate fed through a port) so
the UI never waits: each step is one short native call (a prompt chunk or one token), so
cancelling is just not asking for the next one. Ship the model as a downloaded file or an
asset copied to the app's files directory, since it is mapped from a path.

## On a computer, to try it

### Intel Mac (macOS)

1. Xcode's command line tools give `clang`, `make` and `git` (Homebrew is not needed):

   ```sh
   xcode-select --install
   ```

2. Steps 1 to 3 above (clone, the converter, download, convert).
3. Build and run:

   ```sh
   make
   ./examples/chat/chat models/smollm2.weights
   ```

rayo picks SSE4.1 or AVX2 (and AVX-VNNI) at run time with cpuid, so the same build runs on
every Intel Mac, AVX2 or not. The default threads are the physical cores (`hw.physicalcpu`)
up to 4. Apple silicon Macs build and run the same way, with NEON and the dot product.

### Linux

```sh
sudo apt install build-essential git python3-venv   # or your distribution's equivalent
```

Then steps 1 to 3, `make` and `./examples/chat/chat models/smollm2.weights`, as on a Mac.

### Windows (optional, not yet checked in CI)

With [MSYS2](https://www.msys2.org) (MinGW-w64 shell): `pacman -S make mingw-w64-x86_64-gcc
python`, then `make LIBS=-lm` and the same steps. The file is mapped with `MapViewOfFile` and
the threads are Win32 threads.

## What to expect

Measured on one 2.1 GHz Xeon core (x86_64, AVX-VNNI) with models of the real shapes and random
weights, 128 prompt tokens then 64 decoded (`tools/slm_bench.c`). No phone has been measured
yet: please send numbers from a real phone (`./chat` prints them on the way out).

| Model, weights | File | Peak memory | Prompt tokens/s (1 / 2 / 4 threads) | Reply tokens/s (1 / 2 / 4 threads) |
| --- | --- | --- | --- | --- |
| SmolLM2-135M shape, q4 | 80 MB | 95 MB | 116 / 154 / 181 | 42 / 69 / 54 |
| SmolLM2-135M shape, int8 | 129 MB | 146 MB | 281 / 347 / 364 | 46 / 64 / 26 |
| SmolLM2-135M shape, float32 | 513 MB | 528 MB | 48 / 134 / 182 | 18 / 29 / 32 |
| Qwen2.5-0.5B shape, q4 | 295 MB | 308 MB | 48 / 76 / 115 | 12 / 21 / 30 |
| Qwen2.5-0.5B shape, int8 | 473 MB | 491 MB | 82 / 135 / 225 | 18 / 28 / 32 |

Four threads on this shared virtual machine are noisy; on a phone, measure 2 and 4.
