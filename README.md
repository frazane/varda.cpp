# varda.cpp

A from-scratch C++/CUDA/Metal inference engine for
[MeteoSwiss Varda-single-1.0](https://huggingface.co/MeteoSwiss/Varda-single-1.0): the **6-hourly forecaster**
(`varda-forecaster-det-sgm/model.ckpt`) and the **1-hourly temporal downscaler** (`varda-temporal-downscaler-det-sgm/model.ckpt`),
two anemoi GraphTransformer encoder–processor–decoders on a stretched grid (1 688 650 data nodes: 1 km over the Alps,
31 km globally; 189 809 hidden nodes). The downscaler reconstructs the hourly states between the forecaster's
6-hourly ones (`docs/downscaler.md`). Both models run on CPU and CUDA; the Metal backend runs the forecaster
only (`docs/metal.md`).

> **Not an official MeteoSwiss product.** varda.cpp is an independent project. It is not developed, endorsed or
> supported by MeteoSwiss, and its forecasts are not MeteoSwiss forecasts. See [Licence and disclaimer](#licence-and-disclaimer).

The checkpoints are identified by their sha256 (forecaster `1ad7a7a6…a9c7d8`, downscaler `452fda53…a21d0e`) and
uuid; file and directory names use `varda` and `varda-td`. The engine is verified against golden activations,
reference rollouts and calibrated tolerances produced by running the original anemoi models; it replaces PyTorch
with C++17, OpenMP, BLAS, CUDA and cuBLAS.

## Quick start

A forecast from today's open data (KENDA-CH1 + ECMWF IFS), GRIB out, on a Slurm GPU cluster
(machine-specific paths and Slurm arguments go in `tools/env.local.sh`, from `tools/env.local.example.sh`):

```
./varda setup                       # once: build, check (or create) the assets bundle
./varda forecast latest             # or a date: ./varda forecast 2026-10-22T15 [--lead-time 48] [--precision fp32]
./varda forecast latest --gpus 2    # the temporal downscaler on a second GPU (default: both models on one)
```

GRIB files land in `./varda-out/<YYYYmmddTHH>/outputs/`: the forecaster's 6-hourly files and, when the bundle holds the
temporal downscaler, the hourly `varda-single_*.grib` files of the full Varda-single-1.0 forecast (`docs/downscaler.md`).
What each step does, where data go, other machines and troubleshooting: `docs/quickstart.md`.

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
6. **Run the temporal downscaler** after it, the way the Hugging Face notebook does: the hourly
   states between the 6-hourly ones, as GRIB, from one command.

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
8. The temporal downscaler, the second model of Varda-single-1.0, runs on the same engine (CPU and CUDA),
   verified like the forecaster (golden activations made with the original model, calibrated tolerances, the
   loop and the hourly GRIB files against anemoi-inference's `temporal_downscaler` runner; CPU and CUDA, Metal
   forecaster only), so that one `varda-forecast` run produces the full hourly forecast of the Hugging Face notebook.

## Results

All eight items of the definition of done are met (details and run configurations: `docs/results.md`; the
downscaler: `docs/downscaler.md`).

| check | result |
|---|---|
| converter vs an independent weight export | bitwise, both models (`docs/tools.md`) |
| engine dependencies | C++17, OpenMP, OpenBLAS, CUDA runtime, cuBLAS/cuBLASLt; no Python, no libtorch |
| CPU fp32, 30 activations vs anemoi CPU fp32 | Varda worst 0.18 of tolerance (max 1.8e-6 rel), dev model 0.10; bitwise across thread counts |
| CUDA fp32 / fp16 vs anemoi | Varda worst 0.33 / 0.67 (and 0.62 vs anemoi's own fp16); dev model 0.31 / 0.66 / 0.73 |
| generality: 6 synthetic models (a-f; the interpolators g, h are in the downscaler row) vs float64 reference | ≤ 9.2e-7 fp32 (tol 1e-5), ≤ 1.4e-3 fp16 (tol 3e-3), CPU and CUDA |
| computed forcings vs anemoi-inference | bitwise (0 of 194M values differ) |
| rollout, 24 × 6 h steps vs anemoi-inference's loop | CPU fp32 0.58, CUDA fp32 0.50, fp16 0.73 / 0.84 of per-step tolerances; forcings, dates and device loop == host loop bitwise |
| temporal downscaler (`varda-td`, 6 hourly states per call; `docs/downscaler.md`), 30 activations vs anemoi (golden made here) | CPU fp32 worst 0.37 of tolerance, CUDA fp32 0.52, fp16 0.64 / 0.76 vs anemoi's own fp16; synthetic interpolators g, h within the generality rules; 732 ms per call fp16 on an A100 (forecaster 720 ms in the same job) |
| downscaling loop, 4 windows x 6 h, vs anemoi-inference's `temporal_downscaler` runner on the real forecaster run | on the reference's exact inputs: CPU fp32 0.34, CUDA fp32 0.33 and fp16 0.85 of per-window, per-hour tolerances (`tolerances/downscale/varda-td.json`, calibrated on 2026-10-22T15); on a second date, 2026-10-30T15, with the same tolerances: CPU fp32 0.42, CUDA fp32 0.43, fp16 0.89 / 0.85, every table 24/24 within; inputs assembled in memory differ from anemoi's GRIB-decoded ones by ≤ 1.2e-7 (not measurable on the outputs) |
| real-data forecast with the downscaler: 6-hourly + 25 hourly GRIB files, 24 h, 2026-10-22T15 (`docs/downscaler.md` §7) | hourly files vs anemoi-inference's `temporal_downscaler` runner: worst message 1.5e-2 of the field max in fp16 (anemoi's own fp16-vs-fp32 chain difference is 1.04e-2) and 1.4e-5 in fp32; hour 0: 62 of 66 messages byte-identical (the other 4 differ at re-quantisation level, as in anemoi's own re-encode); two runs byte-identical (35 files, 2350 messages); **14.3-14.4 s warm (32 CPUs), 11.2 s (64 CPUs)** process wall clock for both models and 35 files (A100, fp16; 44 s fp32 with the downscaler on a second GPU, `./varda forecast --gpus 2`; a 5-day forecast, 163 files, **33.2-33.5 s** with 64 CPUs and 44.2 s with 32: `docs/results.md` section 6), against 215-247 s + 192-194 s for the notebook's two anemoi-inference commands (same job as an earlier binary's 23.7 s, about 17x then, about 29x now at 32 CPUs; fp32 250 + 217 s, 7x then, about 10x now; the node was shared with another job of ours) |
| real-data forecast, KENDA-CH1 + IFS open data, 2026-10-22T15 (`docs/forecast.md`; forecaster only, 4 steps) | `x0` and every input stage bitwise vs anemoi-inference's dumps; GRIB writer byte-identical to anemoi's (676/676 messages, fed its states) and for step 0 (140/140); 4-step rollout vs anemoi-inference fp16 worst 0.49 of per-step tolerance (0/16 above); GRIB in to GRIB out, fp16, 4 steps, A100, graph on, process wall clock: **about 13 s for the first run in a fresh allocation** (12.9 / 13.6 s; 8.3-8.7 s once the shared libraries are cached, with the model and input file caches dropped) and 7.8 s warm (`docs/results.md` section 5). Against anemoi-inference, current binary, same jobs, node and inputs (`docs/results.md` section 5): **8.2 s vs 135.8 s warm (16.6x)**, 12.9 / 13.6 s vs 252.7 / 248.2 s for the first run in a job (18-20x); an earlier run with the previous binary gave fp32 41 s vs 163 s (4.0x) |

Speed, A100-SXM4-80GB, one warm forward pass, both implementations in the same job:

| model | precision | original anemoi | **varda.cpp** |
|---|---|---|---|
| Varda | fp16 | 1338 ms | **724 ms** (1.85x) |
| Varda | fp32 | 8068 ms | **7439 ms** (1.08x) |
| dev model | fp16 | 79.0 ms | **31.7 ms** (2.49x) |
| dev model | fp32 | 379 ms | **330 ms** (1.15x) |

A 24-step (6-day) Varda rollout runs at 727 ms per step in fp16, which includes the state
update (2.2 ms). The first call takes 0.9 s, against 4.9 s for the original. The temporal downscaler
takes 732 ms per 6-hour window in fp16 (six hourly states; forecaster 720 ms in the same job); the original takes
about 1.7 s per window warm under anemoi-inference (a different job, `docs/results.md` section 6), about 2.3x. In
`varda-forecast` the downscaler's loop is device-resident (its inputs assembled on the GPU from the rollout's outputs)
and runs during the rollout, on the same GPU or a second one; the hourly GRIB files are encoded from prepared
handles, several hours at once, byte-identical to anemoi's. A 5-day forecast (41 six-hourly + 121 hourly files, 24 GB)
takes 33 s with 64 CPUs and 44 s with 32.
On CPU (64 cores, fp32), a Varda forward pass takes 60 s with a 25 GB peak.
Design and methodology: `docs/design.md`.

## Layout

```
src/varda/      engine library: format (safetensors mmap), config, graph (CSR), model, rollout, downscale
                (the temporal downscaler's window loop), cpu/ (fp32 oracle: OpenMP + OpenBLAS), cuda/ (CUDA
                kernels + cuBLAS), metal/
apps/           varda-info, varda-run, varda-parity, varda-bench, varda-rollout, varda-downscale;
                apps/io/: varda-prepare, varda-grib-write, varda-grib-compare, varda-forecast
tests/          C++ unit tests (ctest)
python/         converter, golden export, reference drivers (tools only, never the engine)
tolerances/     per-checkpoint tolerances, keyed by uuid (calibrated on anemoi reference runs)
docs/           design.md (start here), format.md (contracts), results.md, cpu.md, cuda.md, metal.md,
                rollout-engine.md, rollout.md, forcings.md, tools.md, testing.md,
                downscaler.md (the temporal downscaler), downscaler-reference.md,
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
anemoi-inference does), with the temporal downscaler when given one (`--downscaler`); it needs ecCodes
(`-DVARDA_GRIB=ON`, `tools/env.sh`): `docs/forecast.md`. `build/varda-downscale` runs the downscaler on stored
states and judges it against reference dumps (`docs/downscaler.md`).

Apple Silicon: the CPU fp32 backend builds natively on macOS arm64 with OpenMP and OpenBLAS.
Build, GRIB dependencies and verification: [docs/apple-silicon.md](docs/apple-silicon.md).
The optional [Metal/MPS backend](docs/metal.md) runs the forecaster on the Apple GPU with
`--device metal --precision fp32` or `--precision fp16` (mixed precision);
CUDA requires an NVIDIA GPU.

## Licence and disclaimer

- The code is licensed under the BSD 3-Clause License (`LICENSE`), for every version of this repository.
- `third_party/nlohmann/json.hpp` is © Niels Lohmann, MIT License (notice in the file).
- The model weights are not part of this repository. Varda-single-1.0 is published by MeteoSwiss on
  [Hugging Face](https://huggingface.co/MeteoSwiss/Varda-single-1.0) under CC BY 4.0. Files taken from or derived
  from that repository by the tools here (the converted model, the assets bundle, the GRIB templates) remain under
  CC BY 4.0: credit MeteoSwiss when you share them.
- varda.cpp is not an official MeteoSwiss product. It is not developed, endorsed or supported by MeteoSwiss, and
  forecasts made with it are not MeteoSwiss forecasts. It comes without warranty (see `LICENSE`).
