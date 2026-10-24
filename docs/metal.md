# Apple Silicon Metal backend

`--device metal --precision fp32` or `--precision fp16` runs the forecaster on a
unified-memory Metal GPU. FP16 uses mixed precision: low precision storage with
FP32 accumulation and sensitive calculations. It uses MPS matrix multiplication
and native Metal compute shaders. There
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
build-metal/varda-run --model MODEL.safetensors --input INPUT.safetensors \
  --device metal --precision fp16 --steps 3
```

The build links Apple's Foundation, Metal and MetalPerformanceShaders frameworks.
Shader source is embedded at configure time and compiled by Metal at model load;
the executable does not need the source tree or an external `.metallib` at run
time. The normal parity app still needs its reference/tolerance files.

`varda-rollout --device metal --precision fp32|fp16` uses the existing host rollout
loop: predictions run on the GPU, while feedback and computed forcings run on
the host. `--engine device` remains CUDA-only. With an assets bundle,
`./varda forecast latest --device metal --precision fp16` runs locally and probes
the Metal GPU before downloading inputs; it does not invoke Slurm.

## Implementation and numerics

- FP32 dense layers: `MPSMatrixMultiplication`, float32 inputs, weights and outputs,
  with a separate bias addition and one cached multiplication object per shape.
- Graph attention: destination-sorted CSR, explicit edge projection inside the
  score/value kernels, per-edge/head scores and softmax, and stored-order sums.
  No edge-by-channel tensor is materialised. Empty graphs and zero-in-degree
  destinations are supported.
- Normalisation, input assembly, residuals, bounding and inverse normalisation
  are GPU kernels. Fraction bounding stages the products before assignment,
  including when the total field is itself an output being bounded.
- Metal compilation uses safe math and precise functions, with contraction off
  in the shaders. Metal has no fp64: LayerNorm uses one SIMD group per row,
  queries the SIMD width at runtime and computes compensated fp32 moments after
  subtracting a common row origin. Applying normalization to centered values
  avoids cancellation for nearly constant rows with large offsets.
  The erf form of GELU evaluates erf with A&S 7.1.26; the configured tanh form is
  also supported. This is a separate numerical implementation, validated by the
  existing fp32 tolerances, rather than a bitwise match to CPU or CUDA.
- Shared buffers hold the weights and reusable activations. Input/output copies
  and hook reads happen only after the relevant GPU work is complete. Hooks
  synchronize/read the same production calculations; they do not substitute CPU
  calculations. GPU command-buffer errors propagate to the caller. Submission
  batches are capped at 64 encoded operations (compute dispatches or MPS calls),
  retaining a final completion wait before reading output. This bounds queued
  operation count and resource retention, not elapsed GPU time.

### Mixed FP16 precision policy

| Stage | Storage and calculation |
|---|---|
| Normalization and feature assembly | FP32, including public inputs and hook values |
| Dense layers | FP16 operands/weights/bias; MPS **FP32 result**; FP32 bias addition, then one FP16 rounding |
| LayerNorm | FP32 affine parameters, centered FP32 moments and affine arithmetic; FP16 output (the next Linear's input cast) |
| Attention | FP16 q/k/v and edge operands upcast for FP32 dot products, softmax and weighted sums; FP16 attention output |
| Edge projection | FP16 attributes/weights/bias, FP32 sum, then FP16 projection rounding before use in attention |
| Residual stream and GELU | FP32 arithmetic with FP16 rounding at each stored operation |
| Physical output | Decoder's FP16 prediction upcast; residual, bounding and inverse normalization in FP32; public outputs FP32 |

MPS gets padded FP16 weight rows. Already compatible FP16 activations are used
directly; FP32 inputs and odd/unaligned rows use a reusable packing buffer.
GEMMs write a bounded FP32 scratch result, rather than asking MPS for an all-half
GEMM, which can overflow intermediate values before cancellation. Ordered,
tracked Metal accesses preserve scratch dependencies. Large embedding GEMMs
are also chunked in FP16 mode so scratch size stays bounded.

Activation hooks upcast stored FP16 values for inspection after synchronization;
they read the same production results. FP32 retains the original GEMM path; its
outputs are unchanged by the FP16 support (bitwise, all six synthetic models). CPU remains FP32-only;
CUDA precision choices are unchanged. FP32 accumulation does not prevent
overflow when a genuinely large result is stored in FP16, nor recover input
quantization losses. The policy follows autocast rounding boundaries but does
not promise bitwise equality with CUDA's fused/folded attention implementation.

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
activation is `proc.1.out` in model d: `rel_absmax` 2.72e-6 and `max_abs`
4.18e-5, or 0.272 of its 1e-5 relative tolerance (8 host threads). Input
normalisation and assembly are bitwise equal to the reference; hook-free and
hooked outputs are bitwise equal. A small-chunk run also passes with the Metal
API validation layer enabled.

All 12 FP16 synthetic runs also pass unchanged `synthetic-fp16` tolerances
(3e-3; 1e-6 for input stages). Worst activation in model e is `proc.3.out`:
`rel_absmax` 1.40e-3, `max_abs` 2.49e-2, ratio to tolerance 0.465 (8 host
threads). FP16 hooks are inert, input stages match bitwise, and repeated
inference is bitwise stable. No tolerance files were changed.

The integration suite checks repeated inference, a four-step host rollout with
computed forcings/feedback against CPU at existing FP32/FP16 tolerances,
completely empty graphs, CPU FP16 rejection and invalid chunk sizes. A dense
cancellation regression drives the real FP16 embedding GEMM with products
of +/-65536 and an exactly zero sum, catching intermediate overflow. These are
synthetic correctness checks, not meteorological forecast qualification.
It also generates three independent golden models with 7/35/96 channels,
including one hidden node, incomplete SIMD groups, 1/7/37-row chunks, and 8,201
data nodes crossing the default chunk boundary: ten further checks per precision,
including hook inertness. CI runs GPU parity and integration with Metal API
validation. The direct kernel test below also enables GPU Shader Validation.
Global shader instrumentation changes MPS GEMM results on the hosted Apple
Paravirtual GPU (an API-only control passes, while the first embedding GEMM
fails with instrumentation). Full inference therefore disables global shader
instrumentation in hosted CI; the full suite additionally passes with both
validation layers enabled on the physical M1.

The macOS-only `test_metal_kernels` CTest target directly checks the production
LayerNorm shader against double-precision mathematical references: 384 cases per
precision (768 total),
with widths 1 through 4097, random/constant/nearly constant/cancellation-prone
inputs, optional bias and output guards. FP32 cases pass at 1e-5; measured worst
`rel_absmax` 3.39e-7 and `max_abs` 1.62e-6 in FP32. FP16 inputs are quantized
before constructing the reference; worst `rel_absmax` is 4.81e-4 and `max_abs`
1.95e-3 at tolerance 3e-3. The old FP32 serial shader fails 90 of
these cases (worst relative error 1.07e-2). This test enables Metal API/shader
validation and returns skip code 77 when no GPU is available.

An Apple M1 fp32 microbenchmark of models e and d (two alternating baseline/new
rounds of 30 predictions, discard the first three calls, eight host threads)
measured mean warm forward times of 40.8 -> 25.9 ms and 53.2 -> 31.1 ms after
parallel LayerNorm and bounded batching. Repeats are bitwise stable; Metal has
no CUDA graph path. These synthetic timings do not establish full-model speed.

In a separate mixed-precision comparison on M1 (two alternating FP32/FP16 rounds
of 20 predictions, discard the first three, eight host threads), model e measured
27.1 -> 23.0 ms and model d 32.6 -> 27.8 ms: about 1.18x faster with FP16.
Repeats are bitwise stable. FP16 halves selected weights/activation storage;
indices, input features, scores and FP32 GEMM scratch remain full width, so total
memory usage does not halve. No full-model speed or memory qualification is implied.

`varda-metal-info` exits 77 if no unified-memory Metal device is present. An
explicit synthetic `metal` run also exits 77 rather than reporting a GPU pass
when execution was skipped. CI compiles the backend on macOS arm64 and runs the
GPU checks when a device is exposed; a GPU-less hosted runner reports the skip.

## Limits

No Metal graph capture or GPU-resident rollout window yet. The first call
includes MPS kernel setup. Large-model performance
has not been measured; synthetic timings must not be extrapolated to Varda.

The full Varda checkpoint and real-data GRIB forecasts have not been qualified
on Metal. Its activation storage exceeds the 16 GB test Mac's GPU working-set
budget, so a larger unified-memory machine and full reference assets are needed
for that qualification. Device limits are reported by `varda-metal-info`.
