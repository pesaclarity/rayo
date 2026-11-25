# rayo benchmarks

Written by CI after every push to `main` ([`.github/workflows/ci.yml`](https://github.com/pesaclarity/rayo/blob/main/.github/workflows/ci.yml), `bench` and `bench-publish`); nothing here is edited by hand. GitHub-hosted runners are shared virtual machines: these numbers are indicative and vary from run to run. They are not a phone's.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="table-dark.png">
  <img alt="rayo benchmarks" src="table-light.png">
</picture>

## Latest: 2026-11-25, [`0d699ef`](https://github.com/pesaclarity/rayo/commit/0d699efcee7216698b3438cdab65139344456741)

| | **x86_64**<br>ubuntu-24.04, 4 vCPU | **arm64**<br>ubuntu-24.04-arm, 4 vCPU |
| --- | --- | --- |
| **Machine** | | |
| CPU | Intel Xeon 6975P-C | Neoverse-N2 |
| Kernels chosen at run time | avx-vnni | neon-i8mm |
| **Pesa Clarity's Ask models · int8 · 1 thread · median** | | |
| Reader, per question | 0.16 ms | 0.27 ms |
| Writer, per reply (≈13 tokens) | 0.52 ms | 0.81 ms |
| **SmolLM2-135M shape · synthetic weights† · 128-token prompt, 64 decoded** | | |
| q4 prompt tokens/s (1 / 2 threads) | 193 / 317 | 114 / 198 |
| q4 decode tokens/s (1 / 2 threads) | 67.0 / 95.0 | 53.4 / 88.2 |
| q4 peak memory | 86 MB | 86 MB |
| int8 prompt tokens/s (1 / 2 threads) | 526 / 747 | 395 / 586 |
| int8 decode tokens/s (1 / 2 threads) | 85.3 / 134 | 95.4 / 125 |
| int8 peak memory | 145 MB | 144 MB |
| **Library size · stripped · whole runtime with the language-model path** | | |
| Android library (.so) | arm64-v8a 191 KB · armeabi-v7a 152 KB · x86_64 197 KB |  |

† SmolLM2-135M shape with synthetic (random) weights, written by tool/make_synthetic_slm.py: every matrix product, the cache and the memory are the real model's size, but the text it writes means nothing.

The run: https://github.com/pesaclarity/rayo/actions/runs/45569471897

## History (the last 10 runs; all in [history.csv](history.csv))

| date | commit | runner | reader ms | writer ms | q4 decode tok/s 1 / 2 threads | int8 decode tok/s 1 / 2 threads |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-25 | [`0d699ef`](https://github.com/pesaclarity/rayo/commit/0d699efcee7216698b3438cdab65139344456741) | ubuntu-24.04-arm | 0.3 | 0.8 | 53.4 / 88.2 | 95.4 / 125 |
| 2026-11-25 | [`0d699ef`](https://github.com/pesaclarity/rayo/commit/0d699efcee7216698b3438cdab65139344456741) | ubuntu-24.04 | 0.2 | 0.5 | 67.0 / 95.0 | 85.3 / 134 |
| 2026-11-19 | [`a1eb966`](https://github.com/pesaclarity/rayo/commit/a1eb966e976eba144c1098dd352a81de02812449) | ubuntu-24.04-arm | 0.3 | 0.8 | 51.2 / 82.8 | 86.2 / 118 |
| 2026-11-19 | [`a1eb966`](https://github.com/pesaclarity/rayo/commit/a1eb966e976eba144c1098dd352a81de02812449) | ubuntu-24.04 | 0.2 | 0.5 | 81.0 / 134 | 95.0 / 148 |
| 2026-11-18 | [`cdd7f55`](https://github.com/pesaclarity/rayo/commit/cdd7f559b6cfe88fa917d4760bebca9bdc5dad37) | ubuntu-24.04-arm | 0.3 | 0.8 | 55.9 / 89.6 | 86.2 / 115 |
| 2026-11-18 | [`cdd7f55`](https://github.com/pesaclarity/rayo/commit/cdd7f559b6cfe88fa917d4760bebca9bdc5dad37) | ubuntu-24.04 | 0.3 | 0.7 | 73.0 / 105 | 50.5 / 80.4 |
| 2026-11-18 | [`ec33200`](https://github.com/pesaclarity/rayo/commit/ec33200d47f454eb128cd5ddf9ed00658af16f87) | ubuntu-24.04-arm | 0.3 | 0.8 | 51.3 / 84.6 | 67.5 / 101 |
| 2026-11-18 | [`ec33200`](https://github.com/pesaclarity/rayo/commit/ec33200d47f454eb128cd5ddf9ed00658af16f87) | ubuntu-24.04 | 0.3 | 0.8 | 66.4 / 106 | 54.9 / 88.9 |
| 2026-11-18 | [`21c953e`](https://github.com/pesaclarity/rayo/commit/21c953ebad8cb88790add8fcb8462e371cd6229d) | ubuntu-24.04-arm | 0.3 | 0.8 | 53.3 / 91.1 | 72.8 / 112 |
| 2026-11-18 | [`21c953e`](https://github.com/pesaclarity/rayo/commit/21c953ebad8cb88790add8fcb8462e371cd6229d) | ubuntu-24.04 | 0.2 | 0.5 | 89.0 / 147 | 99.3 / 174 |
| 2026-11-18 | [`b7838f6`](https://github.com/pesaclarity/rayo/commit/b7838f6648e8f100674d0e70412c092bb2c6bb6d) | ubuntu-24.04-arm | 0.3 | 0.8 | 51.8 / 87.5 | 67.8 / 97.9 |
| 2026-11-18 | [`b7838f6`](https://github.com/pesaclarity/rayo/commit/b7838f6648e8f100674d0e70412c092bb2c6bb6d) | ubuntu-24.04 | 0.1 | 0.7 | 88.0 / 144 | 98.8 / 160 |
| 2026-11-18 | [`054c831`](https://github.com/pesaclarity/rayo/commit/054c8311619f60b38fed07c8042cf9564e36126f) | ubuntu-24.04-arm | 0.3 | 0.7 | 64.8 / 107 | 76.8 / 120 |
| 2026-11-18 | [`054c831`](https://github.com/pesaclarity/rayo/commit/054c8311619f60b38fed07c8042cf9564e36126f) | ubuntu-24.04 | 0.3 | 1.0 | 82.1 / 139 | 63.5 / 110 |
| 2026-11-18 | [`9d46ee3`](https://github.com/pesaclarity/rayo/commit/9d46ee3d660190c255d58513d2e358031f57a283) | ubuntu-24.04-arm | 0.3 | 0.7 | 62.7 / 103 | 72.7 / 115 |
| 2026-11-18 | [`9d46ee3`](https://github.com/pesaclarity/rayo/commit/9d46ee3d660190c255d58513d2e358031f57a283) | ubuntu-24.04 | 0.1 | 0.7 | 57.4 / 89.1 | 92.2 / 158 |

Each run's full numbers (CPU flags, p90s, per-step times, file sizes) are in [results/](results/), one JSON file a run; the oldest are removed as new ones come.
