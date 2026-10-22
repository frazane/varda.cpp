# Design

varda.cpp re-implements the inference of an anemoi GraphTransformer encoder–processor–decoder
(the Varda forecaster; dev as a smaller sibling) in C++17/CUDA, from the checkpoint
alone, following the methodology of the Python project (`$IE_ROOT/docs/methodology.md`). This
file is the map; each part has its own document with the details and numbers.

## 1. Pipeline

```
model.ckpt (pickled anemoi interface, HF)
   │  python/convert.py: restricted unpickler, strict accounting, no foreign code runs
   ▼
model.safetensors  ── weights (bitwise = Python engine's weights.pt) + config JSON
   │                  + grid.latitudes/longitudes (F64) + forcings section     docs/format.md
   │  Model::load: mmap, strict take() of every tensor, Config validation
   ▼
Backend ── CPU fp32 oracle (OpenMP + OpenBLAS)                               docs/cpu.md
       └─ CUDA fp32 / fp16 (cuBLASLt + custom kernels, CUDA graph)           docs/cuda.md
   │  predict_step: normalise → encoder → 16 processor blocks → latent skip →
   │                decoder → residual → bounding → denormalise
   ▼
Rollout ── host loop (any backend) / device loop (one CUDA graph per step)   docs/rollout-engine.md
       └─ computed forcings, bitwise equal to anemoi's                        docs/forcings.md
```

Library: `src/varda/` (`varda.h` is the public API). Apps: `varda-info` (inspect a file),
`varda-run` (forward + timing), `varda-parity` (one-step parity against golden data),
`varda-bench` (GPU timing), `varda-rollout` (rollout, per-step comparison and judging).
Tests: `docs/testing.md`.

## 2. Decisions

- **One file, a standard container.** safetensors, with the config embedded as metadata, so
  the file is readable by other tools and mmap-able with no parser beyond a JSON header.
  Tensor names, shapes and bytes equal the Python engine's converted weights, which made the
  converter verifiable by `sha256` and let the C++ work start before the converter existed.
- **Everything from the config.** Sizes, index lists, eps, GELU variant, bounding, residual
  step, forcing sets and timestep are read from the checkpoint by the converter; the engine
  holds no model literal and rejects unknown kinds at load. Strict loading (every tensor
  taken, shapes checked) on both backends. A synthetic suite (6 tiny models outside the real
  checkpoints' dimension envelope, against an independent float64 reference) guards this.
- **CPU path = readable oracle.** pyg-exact op order (max subtraction, `+1e-16`, CSR order of
  sums), double moments in LayerNorm, fixed GEMM tiles so the bits do not depend on the thread
  count. Varda takes 60 s on 64 cores, against ~200 s for the original as measured by the Python
  project (same node type, not the same job).
- **CUDA path = fast path, same contract.** Fused `[q|k|v|self]` GEMMs, the static edge
  projection folded into the attention (no per-edge tensor), fp32 online softmax, residuals
  fused into the next LayerNorm while keeping the reference's separate fp16 roundings,
  decoder destination chunks, all buffers planned at load, one cuBLASLt plan per shape, one
  CUDA graph. fp16 reproduces the reference's autocast cast placement op by op.
- **Debug activations on the production path.** Both backends report the 30 contract
  activations through a hook; the hook never changes the arithmetic, and every parity run
  checks that the hook-free output is bitwise equal.
- **Rollout as a decode loop.** Preallocated device window, update kernel inside the step's
  graph, double-buffered outputs/forcings with one graph per parity, forcings for the next step
  computed on the host while the GPU runs, callback one step behind. Update overhead 0.3%.

## 3. Verification (numbers: docs/results.md)

| what | reference | criterion |
|---|---|---|
| converter | Python engine's `weights.pt`/`config.json` | bitwise |
| one step, 30 activations | original anemoi model's golden dumps (CPU fp32 pyg, CUDA fp32/fp16 triton) | calibrated tolerances, `tolerances/<model>.json` |
| generality | float64 numpy reference on 6 synthetic models | 1e-5 (fp32), 3e-3 (fp16) |
| forcings | anemoi-inference's own `ComputedForcings` and the forcings recorded in reference rollouts | bitwise |
| rollout, 24 steps | the original driven by anemoi-inference's own loop (CPU fp32, CUDA fp32/fp16, fp16 perturbation ensemble) | per-step tolerances, `tolerances/rollout/<model>.json` |
| internal | hook vs no hook, graph vs eager, device loop vs host loop, computed vs replayed forcings, thread counts | bitwise |

## 4. Methodology additions (from porting to C++/CUDA)

Generic lessons this port added to the Python project's playbook. Details and measurements
are in the linked documents.

**Process**
- **Reuse the previous port's artefacts by identity, not by name.** Hashing the Hugging Face
  checkpoint showed it was byte-identical to a model already studied, which turned weeks of
  ground-truth work (golden dumps, tolerances, reference venvs) into a lookup. Pair every
  artefact by checkpoint uuid, and refuse unpaired comparisons.
- **Bootstrap the new format from the old engine's converted weights.** A 30-line repack gave
  both C++ backends real files on day one. The real converter then had a bitwise target.
- **Fix the contracts and write the harness before spawning parallel work.** The format, the
  public API, the activation keys and a parity harness that only uses the public API let the
  CPU and GPU backends be built and verified independently, against the same golden files.
- **Review with fresh eyes and synthetic models outside the envelope.** Two real checkpoints
  with identical dimensions can't reveal width, head-count, step-count or bias assumptions. A
  generator of tiny models plus an independent float64 implementation of the spec can. An
  independent review also found holes in the harness itself (NaN in the reference counted as a
  match, checks without a table passing). Test the test: feed it corrupted golden data and
  wrong shapes.

**CPU**
- **Know your BLAS.** The cluster's OpenBLAS was a serial build that silently ignores thread
  settings. All parallelism had to come from OpenMP over GEMM tiles.
- **A GEMM's bits depend on its call shape.** Fixed tiles make an oracle bitwise invariant to
  the thread count, which turns "is this difference a bug?" into a yes/no question.
- **Pre-fault mmapped weights and keep activation arenas.** On network filesystems, the first
  forward otherwise pays for page faults, and per-call allocations re-fault gigabytes.

**GPU**
- **Fixed arithmetic, not just fixed order.** The compiler contracted multiply-adds
  differently in two template instances of one kernel. Explicit `__fmaf_rn` and friends, and
  a test that runs one computation under two launch layouts, made the kernels bitwise stable.
- **One cuBLASLt plan per shape, created eagerly before capture,** makes graph, eager and
  hooked runs bitwise identical. Timing-based autotuning bought nothing and would cost
  cross-process determinism.
- **Pad GEMM dimensions to multiples of 8 with zeros.** It is exact and avoids slow
  `align1` kernels: 2.6x on the 67-wide output layer.
- **Read the limiter, not the time.** ncu moved the attention kernel from L1-bound (strided
  small weights → store structure-of-arrays) to latency-bound (registers → split each
  destination's channels over warps).
- **Fusing can keep the reference's roundings.** Applying the separate fp16 roundings inside
  fused kernels costs no bandwidth, and keeps fp16 errors at 0.6-0.7 of the reference-derived
  tolerances.
- **Know when to stop.** fp32 without TF32 is SGEMM-bound at 97.6% of peak. fp16 is GEMM-bound
  at 80% of the tensor-core peak, and the largest remaining item (exact-erf GELU, 48 ms) needs a
  GEMM epilogue cuBLASLt lacks. A hand-written mma.sync GEMM with that epilogue was slower
  overall.

**Rollout**
- **Make cross-implementation equality a design property.** Both rollout loops call the same
  kernels and the update only moves values, so device loop == host loop bitwise, checked per
  step on the window as well as the output.
- **Forcings: find the coordinates the runner actually uses.** These are the checkpoint's
  supporting arrays (F64 degrees), not the graph's node attributes. With them, and the source's
  operation order in double precision, glibc reproduces numpy bit for bit.
- **Double-buffer by parity and capture one graph per parity**, rather than patching graph
  parameters. Deliver results one step behind so the GPU never waits for the host.
