# rayo benchmarks

Written by CI after every push to `main` ([`.github/workflows/ci.yml`](https://github.com/pesaclarity/rayo/blob/main/.github/workflows/ci.yml), `bench` and `bench-publish`); nothing here is edited by hand. GitHub-hosted runners are shared virtual machines: these numbers are indicative and vary from run to run. They are not a phone's.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="table-dark.png">
  <img alt="rayo benchmarks" src="table-light.png">
</picture>

## Latest: 2026-11-18, [`b7838f6`](https://github.com/pesaclarity/rayo/commit/b7838f6648e8f100674d0e70412c092bb2c6bb6d)

| | **x86_64**<br>ubuntu-24.04, 4 vCPU | **arm64**<br>ubuntu-24.04-arm, 4 vCPU |
| --- | --- | --- |
| **Machine** | | |
| CPU | Intel Xeon 6975P-C | Neoverse-N2 |
| Kernels chosen at run time | avx-vnni | neon-dotprod |
| **Pesa Clarity's Ask models · int8 · 1 thread · median** | | |
| Reader, per question | 0.15 ms | 0.28 ms |
| Writer, per reply (≈13 tokens) | 0.72 ms | 0.80 ms |
| **SmolLM2-135M shape · synthetic weights† · 128-token prompt, 64 decoded** | | |
| q4 prompt tokens/s (1 / 2 threads) | 188 / 296 | 99.4 / 177 |
| q4 decode tokens/s (1 / 2 threads) | 88.0 / 144 | 51.8 / 87.5 |
| q4 peak memory | 86 MB | 85 MB |
| int8 prompt tokens/s (1 / 2 threads) | 470 / 654 | 306 / 454 |
| int8 decode tokens/s (1 / 2 threads) | 98.8 / 160 | 67.8 / 97.9 |
| int8 peak memory | 145 MB | 144 MB |
| **Library size · stripped · whole runtime with the language-model path** | | |
| Android library (.so) | arm64-v8a 174 KB · armeabi-v7a 139 KB · x86_64 193 KB |  |

† SmolLM2-135M shape with synthetic (random) weights, written by tool/make_synthetic_slm.py: every matrix product, the cache and the memory are the real model's size, but the text it writes means nothing.

The run: https://github.com/pesaclarity/rayo/actions/runs/44274420673

## History (the last 10 runs; all in [history.csv](history.csv))

| date | commit | runner | reader ms | writer ms | q4 decode tok/s 1 / 2 threads | int8 decode tok/s 1 / 2 threads |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | [`b7838f6`](https://github.com/pesaclarity/rayo/commit/b7838f6648e8f100674d0e70412c092bb2c6bb6d) | ubuntu-24.04-arm | 0.3 | 0.8 | 51.8 / 87.5 | 67.8 / 97.9 |
| 2026-11-18 | [`b7838f6`](https://github.com/pesaclarity/rayo/commit/b7838f6648e8f100674d0e70412c092bb2c6bb6d) | ubuntu-24.04 | 0.1 | 0.7 | 88.0 / 144 | 98.8 / 160 |
| 2026-11-18 | [`054c831`](https://github.com/pesaclarity/rayo/commit/054c8311619f60b38fed07c8042cf9564e36126f) | ubuntu-24.04-arm | 0.3 | 0.7 | 64.8 / 107 | 76.8 / 120 |
| 2026-11-18 | [`054c831`](https://github.com/pesaclarity/rayo/commit/054c8311619f60b38fed07c8042cf9564e36126f) | ubuntu-24.04 | 0.3 | 1.0 | 82.1 / 139 | 63.5 / 110 |
| 2026-11-18 | [`9d46ee3`](https://github.com/pesaclarity/rayo/commit/9d46ee3d660190c255d58513d2e358031f57a283) | ubuntu-24.04-arm | 0.3 | 0.7 | 62.7 / 103 | 72.7 / 115 |
| 2026-11-18 | [`9d46ee3`](https://github.com/pesaclarity/rayo/commit/9d46ee3d660190c255d58513d2e358031f57a283) | ubuntu-24.04 | 0.1 | 0.7 | 57.4 / 89.1 | 92.2 / 158 |

Each run's full numbers (CPU flags, p90s, per-step times, file sizes) are in [results/](results/), one JSON file a run; the oldest are removed as new ones come.
