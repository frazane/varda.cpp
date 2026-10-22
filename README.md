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

## Layout

```
src/varda/      engine library: format (safetensors mmap), config, graph (CSR), model,
                cpu/ (fp32 oracle: OpenMP + OpenBLAS), cuda/ (CUDA kernels + cuBLAS)
apps/           varda-info, varda-run, varda-parity, varda-bench, ...
tests/          C++ unit tests (ctest)
python/         converter, golden export, reference drivers (tools only, never the engine)
tolerances/     per-checkpoint tolerances, keyed by uuid (from the Python project)
docs/           format.md (contracts), design.md, results.md
```

## Build

```
source tools/env.sh
cmake -S . -B build && cmake --build build -j
```
