# varda.cpp

A from-scratch C++/CUDA inference engine for the **forecaster** of
[MeteoSwiss Varda-single-1.0](https://huggingface.co/MeteoSwiss/Varda-single-1.0)
(`varda-forecaster-det-sgm/model.ckpt`): an anemoi GraphTransformer
encoder–processor–decoder on a stretched grid (1 688 650 data nodes: 1 km over the Alps,
31 km globally; 189 809 hidden nodes; 6-hourly steps).

It follows the methodology of the Python re-implementation in
the Python re-implementation (checkout:
`$HOME/projects/the Python project`), whose second checkpoint, *varda*, is
bit-identical to the Hugging Face file (sha256 `1ad7a7a6…a9c7d8`). Its specs, golden
activations, calibrated tolerances and reference environments are reused here; this
project replaces PyTorch with C++17, OpenMP, BLAS, CUDA and cuBLAS.

## Goals

1. **Convert once, load a clean format.** `python/convert.py` turns the pickled checkpoint
   into one safetensors file (weights + config JSON) without executing the pickle. The
   engine never sees Python.
2. **Write the forward pass from scratch in C++**: normalisation, encoder, 16 processor
   blocks, decoder, residual, bounding, de-normalisation. A CPU fp32 path is the
   readable oracle; a CUDA path is the fast one.
3. **Match the reference numerically**, activation by activation, against the golden
   tensors of the original anemoi model.
4. **Make it fast**: beat the original and the Python engine's optimised path on an A100.
5. **Run the autoregressive rollout** the way anemoi-inference does.

## Definition of done

1. `python/convert.py` converts the HF checkpoint; every tensor is bitwise equal to the
   Python engine's `weights.pt`, and loading is strict (`docs/format.md` §1).
2. The engine (`libvarda` + `apps/`) runs `predict_step` from the converted file alone:
   no Python, no libtorch.
3. CPU fp32: all 30 contract activations (`docs/format.md` §2) of Varda and of
   dev (a smaller model of the same family, for fast iteration) are within the
   `cuda-fp32` tolerances (≤ 1e-5 rel_absmax) of the `cpu-fp32-pyg` golden data.
4. CUDA fp32 within `cuda-fp32`; CUDA fp16 within `cuda-fp16` and
   `cuda-fp16-vs-ref-fp16` (`tolerances/*.json`).
5. Benchmarked against the original (anemoi, triton backend, fp16 autocast and fp32) and
   the Python engine's fast path, hook-free, warm, in the same job.
6. Autoregressive rollout with anemoi-inference semantics, verified per step against the
   original model driven by anemoi-inference's own loop, with stated tolerances.
7. `docs/` records the design, results, and what porting to C++/CUDA added to the
   methodology.

## Results

All seven items of the definition of done are met (details and job ids: `docs/results.md`).

| check | result |
|---|---|
| converter vs Python engine's weights | bitwise, both models (`docs/tools.md`) |
| engine dependencies | C++17, OpenMP, OpenBLAS, CUDA runtime, cuBLAS/cuBLASLt; no Python, no libtorch |
| CPU fp32, 30 activations vs anemoi CPU fp32 | Varda worst 0.18 of tolerance (max 1.8e-6 rel), dev 0.10; bitwise across thread counts |
| CUDA fp32 / fp16 vs anemoi | Varda worst 0.33 / 0.67 (and 0.62 vs anemoi's own fp16); dev 0.31 / 0.66 / 0.73 |
| generality: 6 synthetic models vs float64 reference | ≤ 9.2e-7 fp32 (tol 1e-5), ≤ 1.4e-3 fp16 (tol 3e-3), CPU and CUDA |
| computed forcings vs anemoi-inference | bitwise (0 of 194M values differ) |
| rollout, 24 × 6 h steps vs anemoi-inference's loop | CUDA fp32 0.50, fp16 0.73 / 0.84 of per-step tolerances; CPU 8 steps 0.29; device loop == host loop bitwise |

Speed, A100-SXM4-80GB, one warm forward pass, all implementations in the same job:

| model | precision | original anemoi | Python fast path | **varda.cpp** |
|---|---|---|---|---|
| Varda | fp16 | 1338 ms | 880 ms | **724 ms** (1.85x / 1.22x) |
| Varda | fp32 | 8068 ms | 7573 ms | **7439 ms** (1.08x / 1.02x) |
| dev | fp16 | 79.0 ms | 38.0 ms | **31.7 ms** (2.49x / 1.20x) |
| dev | fp32 | 379 ms | 334 ms | **330 ms** (1.15x / 1.01x) |

A 24-step (6-day) Varda rollout runs at 727 ms per step in fp16, which includes the state
update (2.2 ms). The first call takes 0.9 s, against 5–36 s for the Python implementations.
On CPU (64 cores, fp32), a Varda forward pass takes 60 s with a 25 GB peak.
Design and methodology additions are in `docs/design.md`.

## Layout

```
src/varda/      engine library: format (safetensors mmap), config, graph (CSR), model,
                cpu/ (fp32 oracle: OpenMP + OpenBLAS), cuda/ (CUDA kernels + cuBLAS)
apps/           varda-info, varda-run, varda-parity, varda-bench, varda-rollout
tests/          C++ unit tests (ctest)
python/         converter, golden export, reference drivers (tools only, never the engine)
tolerances/     per-checkpoint tolerances, keyed by uuid (from the Python project)
docs/           design.md (start here), format.md (contracts), results.md, cpu.md, cuda.md,
                rollout-engine.md, rollout.md, forcings.md, tools.md, testing.md
bench/          same-job benchmark driver (original, Python fast path, varda.cpp)
```

## Build

```
source tools/env.sh
cmake -S . -B build && cmake --build build -j
.venv/bin/python -I python/convert.py MODEL.ckpt model.safetensors     # once per checkpoint
build/varda-run --model model.safetensors --input golden.safetensors --device cuda --precision fp16
build/varda-rollout ...                                                  # docs/rollout-engine.md
```
