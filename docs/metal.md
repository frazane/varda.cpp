# Apple Silicon Metal backend

`--device metal --precision fp32` runs the forecaster on a unified-memory Metal
GPU. It uses MPS matrix multiplication and native Metal compute shaders. There
is no PyTorch dependency and no silent CPU fallback. CPU and CUDA numerical
kernels, precision settings and tolerance files are unchanged.

## Build and run

Follow `apple-silicon.md` for OpenMP/OpenBLAS and the pinned GRIB dependencies.
`VARDA_METAL` defaults to ON on macOS and OFF elsewhere. Disable it explicitly
with `-DVARDA_METAL=OFF` for a CPU-only Mac build. Enabling it on Linux is an
error; Linux builds do not enable Objective-C++ or link any Apple framework.

```sh
cmake -S . -B build-metal -DVARDA_CUDA=OFF -DVARDA_METAL=ON \
  -DCMAKE_PREFIX_PATH="$(brew --prefix libomp);$(brew --prefix openblas)"
cmake --build build-metal -j 4
build-metal/varda-metal-info
build-metal/varda-run --model MODEL.safetensors --input INPUT.safetensors \
  --device metal --precision fp32 --steps 3
```

The build links Apple's Foundation, Metal and MetalPerformanceShaders frameworks.
Shader source is embedded at configure time and compiled by Metal at model load;
the executable does not need the source tree or an external `.metallib` at run
time. The normal parity app still needs its reference/tolerance files.

`varda-rollout --device metal --precision fp32` uses the existing host rollout
loop: predictions run on the GPU, while feedback and computed forcings run on
the host. `--engine device` remains CUDA-only. With an assets bundle,
`./varda forecast latest --device metal --precision fp32` runs locally and probes
the Metal GPU before downloading inputs; it does not invoke Slurm.

## Implementation and numerics

- Dense layers: `MPSMatrixMultiplication`, float32 inputs, weights and outputs,
  with a separate bias addition and one cached multiplication object per shape.
- Graph attention: destination-sorted CSR, explicit edge projection inside the
  score/value kernels, per-edge/head scores and softmax, and stored-order sums.
  No edge-by-channel tensor is materialised. Empty graphs and zero-in-degree
  destinations are supported.
- Normalisation, input assembly, residuals, bounding and inverse normalisation
  are GPU kernels. Fraction bounding stages the products before assignment,
  including when the total field is itself an output being bounded.
- Metal compilation uses safe math and precise functions, with contraction off
  in the shaders. Metal has no fp64: LayerNorm uses compensated fp32 moments.
  The erf form of GELU evaluates erf with A&S 7.1.26; the configured tanh form is
  also supported. This is a separate numerical implementation, validated by the
  existing fp32 tolerances, rather than a bitwise match to CPU or CUDA.
- Shared buffers hold the weights and reusable activations. Input/output copies
  and hook reads happen only after the relevant GPU work is complete. Hooks
  synchronize/read the same production calculations; they do not substitute CPU
  calculations. GPU command-buffer errors propagate to the caller.

`VARDA_METAL_CHUNK_ROWS` controls complete destination-row chunks (default 8192).
It bounds dense/attention workspace; it does not split a destination's softmax.
All buffers are planned at load. Allocations are checked against the device's
maximum buffer size and recommended working-set budget; dimensions must fit
32-bit GPU indexing. Bounding index lists must have unique output indices.

## Verification

```sh
BIN="$PWD/build-metal" SYN_DIR="$PWD/local/synthetic" \
  OPENBLAS_NUM_THREADS=1 bash tests/run_synthetic.sh metal
.venv/bin/python -I python/tests/test_metal.py --bin build-metal \
  --synthetic local/synthetic --work local/metal
```

On Apple M1, fp32: all 12 synthetic parity runs (six configurations, default
chunking and 37-row chunks) pass unchanged `synthetic-fp32` tolerances. Worst
activation `rel_absmax` is 2.10e-6, or 0.210 of its 1e-5 tolerance. Input
normalisation and assembly are bitwise equal to the reference; hook-free and
hooked outputs are bitwise equal. A small-chunk run also passes with the Metal
API validation layer enabled.

The integration suite checks repeated inference, a four-step host rollout with
computed forcings/feedback against CPU at the existing synthetic fp32 tolerance,
completely empty graphs, FP16 rejection and invalid chunk sizes. These are
synthetic correctness checks, not meteorological forecast qualification.

`varda-metal-info` exits 77 if no unified-memory Metal device is present. An
explicit synthetic `metal` run also exits 77 rather than reporting a GPU pass
when execution was skipped. CI compiles the backend on macOS arm64 and runs the
GPU checks when a device is exposed; a GPU-less hosted runner reports the skip.

## Limits

FP32 only; no mixed precision, Metal graph capture, or GPU-resident rollout
window yet. The first call includes MPS kernel setup. Large-model performance
has not been measured; synthetic timings must not be extrapolated to Varda.

The full Varda checkpoint and real-data GRIB forecasts have not been qualified
on Metal. Its activation storage exceeds the 16 GB test Mac's GPU working-set
budget, so a larger unified-memory machine and full reference assets are needed
for that qualification. Device limits are reported by `varda-metal-info`.
