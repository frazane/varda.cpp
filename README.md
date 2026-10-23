# varda.cpp

A from-scratch C++/CUDA inference engine for the **forecaster** of
[MeteoSwiss Varda-single-1.0](https://huggingface.co/MeteoSwiss/Varda-single-1.0)
(`varda-forecaster-det-sgm/model.ckpt`): an anemoi GraphTransformer
encoder–processor–decoder on a stretched grid (1 688 650 data nodes: 1 km over the Alps,
31 km globally; 189 809 hidden nodes; 6-hourly steps).

> **Not an official MeteoSwiss product.** varda.cpp is an independent project. It is not developed, endorsed or
> supported by MeteoSwiss, and its forecasts are not MeteoSwiss forecasts. See [Licence and disclaimer](#licence-and-disclaimer).

The checkpoint is identified by its sha256 (`1ad7a7a6…a9c7d8`) and uuid; file and directory
names use `varda`. The engine is verified against golden activations, reference rollouts and
calibrated tolerances produced by running the original anemoi model; it replaces PyTorch with
C++17, OpenMP, BLAS, CUDA and cuBLAS.

## Quick start

A forecast from today's open data (KENDA-CH1 + ECMWF IFS), GRIB out, on a Slurm GPU cluster
(machine-specific paths and Slurm arguments go in `tools/env.local.sh`, from `tools/env.local.example.sh`):

```
./varda setup                       # once: build, check (or create) the assets bundle
./varda forecast latest             # or a date: ./varda forecast 2026-10-22T15 [--lead-time 48] [--precision fp32]
```

GRIB files land in `./varda-out/<YYYYmmddTHH>/outputs/`. What each step does, where data go, other machines and
troubleshooting: `docs/quickstart.md`.

## Goals

1. **Convert once, load a clean format.** `python/convert.py` turns the pickled checkpoint
   into one safetensors file (weights + config JSON) without executing the pickle. The
   engine never sees Python.
2. **Write the forward pass from scratch in C++**: normalisation, encoder, 16 processor
   blocks, decoder, residual, bounding, de-normalisation. A CPU fp32 path is the
   readable oracle; a CUDA path is the fast one.
3. **Match the reference numerically**, activation by activation, against the golden
   tensors of the original anemoi model.
4. **Make it fast**: beat the original on an A100.
5. **Run the autoregressive rollout** the way anemoi-inference does.

## Definition of done

1. `python/convert.py` converts the HF checkpoint; every tensor is bitwise equal to an
   independent export of the checkpoint's weights (`weights.pt`), and loading is strict (`docs/format.md` §1).
2. The engine (`libvarda` + `apps/`) runs `predict_step` from the converted file alone:
   no Python, no libtorch.
3. CPU fp32: all 30 contract activations (`docs/format.md` §2) of Varda and of
   the dev model (a smaller, non-public checkpoint of the same architecture, used for fast
   iteration; any small anemoi GraphTransformer checkpoint serves) are within the
   `cuda-fp32` tolerances (≤ 1e-5 rel_absmax) of the `cpu-fp32-pyg` golden data.
4. CUDA fp32 within `cuda-fp32`; CUDA fp16 within `cuda-fp16` and
   `cuda-fp16-vs-ref-fp16` (`tolerances/*.json`).
5. Benchmarked against the original (anemoi, triton backend, fp16 autocast and fp32),
   hook-free, warm, in the same job.
6. Autoregressive rollout with anemoi-inference semantics, verified per step against the
   original model driven by anemoi-inference's own loop, with stated tolerances.
7. `docs/` records the design, the results and the methodology.

## Results

All seven items of the definition of done are met (details and run configurations: `docs/results.md`).

| check | result |
|---|---|
| converter vs an independent weight export | bitwise, both models (`docs/tools.md`) |
| engine dependencies | C++17, OpenMP, OpenBLAS, CUDA runtime, cuBLAS/cuBLASLt; no Python, no libtorch |
| CPU fp32, 30 activations vs anemoi CPU fp32 | Varda worst 0.18 of tolerance (max 1.8e-6 rel), dev model 0.10; bitwise across thread counts |
| CUDA fp32 / fp16 vs anemoi | Varda worst 0.33 / 0.67 (and 0.62 vs anemoi's own fp16); dev model 0.31 / 0.66 / 0.73 |
| generality: 6 synthetic models vs float64 reference | ≤ 9.2e-7 fp32 (tol 1e-5), ≤ 1.4e-3 fp16 (tol 3e-3), CPU and CUDA |
| computed forcings vs anemoi-inference | bitwise (0 of 194M values differ) |
| rollout, 24 × 6 h steps vs anemoi-inference's loop | CPU fp32 0.58, CUDA fp32 0.50, fp16 0.73 / 0.84 of per-step tolerances; forcings, dates and device loop == host loop bitwise |
| real-data forecast, KENDA-CH1 + IFS open data, 2026-10-22T15 (`docs/forecast.md`) | `x0` and every input stage bitwise vs anemoi-inference's dumps; GRIB writer byte-identical to anemoi's (676/676 messages, fed its states) and for step 0 (140/140); 4-step rollout vs anemoi-inference fp16 worst 0.49 of per-step tolerance (0/16 above); GRIB in to GRIB out, fp16, 4 steps, A100, graph on, process wall clock: **about 13 s for the first run in a fresh allocation** (12.9 / 13.6 s; 8.3-8.7 s once the shared libraries are cached, with the model and input file caches dropped) and 7.8 s warm (`docs/results.md` section 5). Against anemoi-inference, current binary, same jobs, node and inputs (`docs/results.md` section 5): **8.2 s vs 135.8 s warm (16.6x)**, 12.9 / 13.6 s vs 252.7 / 248.2 s for the first run in a job (18-20x); an earlier run with the previous binary gave fp32 41 s vs 163 s (4.0x) |

Speed, A100-SXM4-80GB, one warm forward pass, both implementations in the same job:

| model | precision | original anemoi | **varda.cpp** |
|---|---|---|---|
| Varda | fp16 | 1338 ms | **724 ms** (1.85x) |
| Varda | fp32 | 8068 ms | **7439 ms** (1.08x) |
| dev model | fp16 | 79.0 ms | **31.7 ms** (2.49x) |
| dev model | fp32 | 379 ms | **330 ms** (1.15x) |

A 24-step (6-day) Varda rollout runs at 727 ms per step in fp16, which includes the state
update (2.2 ms). The first call takes 0.9 s, against 4.9 s for the original.
On CPU (64 cores, fp32), a Varda forward pass takes 60 s with a 25 GB peak.
Design and methodology: `docs/design.md`.

## Layout

```
src/varda/      engine library: format (safetensors mmap), config, graph (CSR), model,
                cpu/ (fp32 oracle: OpenMP + OpenBLAS), cuda/ (CUDA kernels + cuBLAS)
apps/           varda-info, varda-run, varda-parity, varda-bench, varda-rollout;
                apps/io/: varda-prepare, varda-grib-write, varda-grib-compare, varda-forecast
tests/          C++ unit tests (ctest)
python/         converter, golden export, reference drivers (tools only, never the engine)
tolerances/     per-checkpoint tolerances, keyed by uuid (calibrated on anemoi reference runs)
docs/           design.md (start here), format.md (contracts), results.md, cpu.md, cuda.md,
                rollout-engine.md, rollout.md, forcings.md, tools.md, testing.md,
                forecast.md (real data: GRIB in, GRIB out), spec-io.md, io-input.md, io-output.md,
                realdata-reference.md
bench/          same-job benchmark driver (original anemoi, varda.cpp)
```

## Build

```
cp tools/env.local.example.sh tools/env.local.sh   # once: toolchain, scratch and data paths for this machine
source tools/env.sh
cmake -S . -B build && cmake --build build -j
.venv/bin/python -I python/convert.py MODEL.ckpt model.safetensors     # once per checkpoint
build/varda-run --model model.safetensors --input golden.safetensors --device cuda --precision fp16
build/varda-rollout ...                                                  # docs/rollout-engine.md
```

`build/varda-forecast` runs a forecast from real data (KENDA-CH1 and IFS open data in, GRIB out, as
anemoi-inference does); it needs ecCodes (`-DVARDA_GRIB=ON`, `tools/env.sh`): `docs/forecast.md`.

## Licence and disclaimer

- The code is licensed under the BSD 3-Clause License (`LICENSE`), for every version of this repository.
- `third_party/nlohmann/json.hpp` is © Niels Lohmann, MIT License (notice in the file).
- The model weights are not part of this repository. Varda-single-1.0 is published by MeteoSwiss on
  [Hugging Face](https://huggingface.co/MeteoSwiss/Varda-single-1.0) under CC BY 4.0. Files taken from or derived
  from that repository by the tools here (the converted model, the assets bundle, the GRIB templates) remain under
  CC BY 4.0: credit MeteoSwiss when you share them.
- varda.cpp is not an official MeteoSwiss product. It is not developed, endorsed or supported by MeteoSwiss, and
  forecasts made with it are not MeteoSwiss forecasts. It comes without warranty (see `LICENSE`).
