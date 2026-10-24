@README.md

# Notes for agents

## Docs map

- `docs/format.md`: the contracts (converted file, activation names, golden files,
  tolerances). Code that produces or consumes any of them follows it exactly.
- `docs/design.md`: the map: pipeline, decisions, verification, methodology.
- `docs/cpu.md`, `docs/cuda.md`: CPU and CUDA backends. `docs/rollout-engine.md`: the C++ rollout.
  `docs/forcings.md`: computed forcings. `docs/rollout.md`: reference rollouts and their
  tolerances. `docs/tools.md`: converter. `docs/testing.md`: every test suite and how to run it.
- `docs/metal.md`: the optional Apple Silicon Metal/MPS backend (fp32 and mixed fp16) and its
  qualification limits. `docs/apple-silicon.md`: building and verifying on macOS arm64.
- `docs/forecast.md`: running a forecast from real data (GRIB in, GRIB out), overlap, timings, limits.
  `docs/spec-io.md`: the I/O spec. `docs/realdata-reference.md`: the anemoi-inference reference run and dumps.
  `docs/io-input.md`: GRIB -> x0. `docs/io-output.md`: y -> GRIB, io config.
- `docs/results.md`: parity and benchmark numbers, each with its full configuration.
- `docs/quickstart.md`: the user entrypoint (`./varda setup|fetch|forecast`), the assets bundle
  (`tools/make_assets.py`) and the input fetch (`python/fetch_inputs.py`).
- The maths is anemoi's GraphTransformer (encoder, processor, decoder; normaliser, residual,
  bounding, index spaces). The CPU backend (`docs/cpu.md`, `src/varda/cpu/`) is its readable
  op sequence: read it before writing a kernel. Fold generic lessons into
  `docs/design.md` §"Methodology".

## Hard rules

- The engine (`src/`, `apps/`, `tests/*.cpp|cu`) uses C++17/CUDA (Objective-C++ for Metal): standard library,
  OpenMP, BLAS (cblas), CUDA runtime, cuBLAS/cuBLASLt, and the vendored
  `third_party/nlohmann/json.hpp`, plus Apple's Foundation/Metal/MetalPerformanceShaders on macOS.
  No Python, no libtorch, no other dependency without
  asking the integrator.
- `python/` holds tools only (converter, golden export, reference drivers). They run in
  `.venv` (torch 2.10 CPU, numpy, pyyaml). Never unpickle the checkpoint into real anemoi
  classes in `.venv`: use the restricted unpickler (stubs every non-torch global).
- The CPU fp32 path is the readable oracle. Optimisations go in the CUDA or Metal backend and are
  verified against golden data at `tolerances/<model>.json`. Never loosen a tolerance
  without saying so explicitly.
- A parity claim needs numbers: `rel_absmax` and `max_abs` per key, plus device, precision
  and thread count (metric: `docs/format.md` §2). A timing claim names device, precision,
  warm or first call, CUDA graph on/off, and compares like with like in the same job.
- Read-only, never write there: the checkpoint files (`$VARDA_CKPT`, `$VARDA_DEV_CKPT`) and
  their directories, `$REF_ROOT`, `$REF_SCRATCH` (set in `tools/env.local.sh`; the reference
  data made with anemoi: golden activations, reference rollouts, reference venvs, weight
  exports). Our large artefacts go under `$VARDA_SCRATCH` (converted models, exported
  golden, job logs), never in the repository.
- Machine-specific values (paths, toolchain, Slurm arguments) live only in the untracked
  `tools/env.local.sh` (template: `tools/env.local.example.sh`) and `CLAUDE.local.md`; tracked
  files use the variables, never the values.
- One owner per file. Your brief lists the files you may change; everything else is
  read-only to you. Shared headers (`src/varda/varda.h`, `config.h`, `format.h`,
  `backend.h`, `graph.h`) change only via the integrator: ask in your report.
- Agents do not commit. The integrator commits (short one-line messages, no
  `Co-authored-by` trailer).
- Never `rm -r` a directory you did not create in the same task, nor a parent such as
  `$VARDA_SCRATCH`. Write shared artefacts atomically (temporary name, then rename).

## Build

```
source tools/env.sh          # sources tools/env.local.sh: toolchain (gcc, CUDA, OpenBLAS), paths
cmake -S . -B build-<you> && cmake --build build-<you> -j 32
```
Use your own build directory (`build-cpu`, `build-cuda`, ...). A running job must never
execute a binary that someone may rebuild: jobs copy the binaries they run to a job-private
directory under `$VARDA_SCRATCH/jobs/<name>/` first. `-DVARDA_CUDA=OFF` builds the CPU
backend only. GPU target: sm_80 (A100).

## Compute

- Model runs go to compute nodes, not the login node: GPU jobs with `$VARDA_SLURM_GPU`, CPU
  jobs with `$VARDA_SLURM_CPU` (Slurm arguments, set in `tools/env.local.sh`). Varda on CPU
  needs ~100–150 GB. GPU only when needed.
- Long or important runs: `sbatch` (survives the orchestrator), logs and exit codes to files
  under `$VARDA_SCRATCH/jobs/`. Blocking `srun` dies with its parent.
- Check your own jobs with `squeue -u $USER` before and after; cancel only jobs you started.

## Models and data (all paths after `source tools/env.sh`)

| | Varda forecaster (`varda`) | dev model (`dev`: smaller, same architecture) |
|---|---|---|
| checkpoint | `$VARDA_CKPT` (the Hugging Face file `varda-forecaster-det-sgm/model.ckpt`) | `$VARDA_DEV_CKPT` |
| converted | `$VARDA_SCRATCH/converted/varda/model.safetensors` | `$VARDA_SCRATCH/converted/dev/model.safetensors` |
| golden (anemoi, read-only) | `$REF_GOLDEN/{cpu-fp32-pyg,cuda-fp32-triton,cuda-fp16-triton}/` (~40 GB each) | `$REF_GOLDEN_DEV/<same tags>/` (1.4 GB each) |
| exported golden | `$VARDA_SCRATCH/golden/varda/<tag>.safetensors` | `$VARDA_SCRATCH/golden/dev/<tag>.safetensors` |
| tolerances | `tolerances/varda.json` | `tolerances/dev.json` |
| sizes | G 1 688 650, H 189 809, E enc/proc/dec 2.85M / 1.52M / 5.07M, 79 in / 67 out | G 40 320, H 10 242, 84 in / 72 out |

Iterate on the dev model (seconds per forward), then confirm on Varda.
