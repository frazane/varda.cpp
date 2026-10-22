@README.md

# Notes for agents

## Docs map

- `docs/format.md`: the contracts (converted file, activation names, golden files,
  tolerances). Code that produces or consumes any of them follows it exactly.
- `docs/design.md`: the map: pipeline, decisions, verification, methodology additions.
- `docs/cpu.md`, `docs/cuda.md`: the two backends. `docs/rollout-engine.md`: the C++ rollout.
  `docs/forcings.md`: computed forcings. `docs/rollout.md`: reference rollouts and their
  tolerances. `docs/tools.md`: converter. `docs/testing.md`: every test suite and how to run it.
- `docs/results.md`: parity and benchmark numbers, each with its full configuration.
- The maths is specified in the Python project (read-only, `$IE_ROOT` =
  `$HOME/projects/the Python project`): `docs/spec-model.md` (encoder, processor,
  decoder), `docs/spec-interface.md` (normaliser, residual, bounding, index spaces),
  `docs/varda.md` (= Varda: delta, scale, GPU notes), `docs/rollout.md`,
  `docs/methodology.md` (the playbook; fold generic lessons from this port into our
  `docs/design.md` §"Methodology additions"). Its engine (`engine/model.py` plain path,
  `engine/fast.py` GPU path) is the closest prior art: read it before writing a kernel.

## Hard rules

- The engine (`src/`, `apps/`, `tests/*.cpp|cu`) is C++17/CUDA only: standard library,
  OpenMP, BLAS (cblas), CUDA runtime, cuBLAS/cuBLASLt, and the vendored
  `third_party/nlohmann/json.hpp`. No Python, no libtorch, no other dependency without
  asking the integrator.
- `python/` holds tools only (converter, golden export, reference drivers). They run in
  `.venv` (torch 2.10 CPU, numpy, pyyaml). Never unpickle the checkpoint into real anemoi
  classes in `.venv`: use the restricted unpickler (stubs every non-torch global).
- The CPU fp32 path is the readable oracle. Optimisations go in the CUDA backend and are
  verified against golden data at `tolerances/<model>.json`. Never loosen a tolerance
  without saying so explicitly.
- A parity claim needs numbers: `rel_absmax` and `max_abs` per key, plus device, precision
  and thread count (metric: `docs/format.md` §2). A timing claim names device, precision,
  warm or first call, CUDA graph on/off, and compares like with like in the same job.
- Read-only, never write there: `$MODELS_DIR/`, `$IE_ROOT`,
  `$IE_SCRATCH` (`$SCRATCH/the Python project`, the Python project's golden
  data, venvs and converted models). Our large artefacts go under
  `$VARDA_SCRATCH` = `$VARDA_SCRATCH/` (converted models, exported
  golden, job logs), never in the repository.
- One owner per file. Your brief lists the files you may change; everything else is
  read-only to you. Shared headers (`src/varda/varda.h`, `config.h`, `format.h`,
  `backend.h`, `graph.h`) change only via the integrator: ask in your report.
- Agents do not commit. The integrator commits (short one-line messages, no
  `Co-authored-by` trailer).
- Never `rm -r` a directory you did not create in the same task, nor a parent such as
  `$VARDA_SCRATCH`. Write shared artefacts atomically (temporary name, then rename).

## Build

```
source tools/env.sh          # gcc 12.3, CUDA 12.3 (nvcc, cuBLAS), OpenBLAS 0.3.26, paths
cmake -S . -B build-<you> && cmake --build build-<you> -j 32
```
Use your own build directory (`build-cpu`, `build-cuda`, ...). A running job must never
execute a binary that someone may rebuild: jobs copy the binaries they run to a job-private
directory under `$VARDA_SCRATCH/jobs/<name>/` first. `-DVARDA_CUDA=OFF` builds the CPU
backend only. GPU target: sm_80 (A100-SXM4 96 GB on the GPU nodes).

## Compute (Slurm, account `$ACCOUNT`)

- Login node: editing, building, reading metadata. No model runs.
- CPU work: `srun -p cpu-short -A $ACCOUNT --time=00:30:00 -c 64 --mem=200G ...` (256-core
  EPYC 7713 nodes, 456 GB, 1 h limit). Varda on CPU needs ~100–150 GB.
- GPU: `-p debug` (30 min) or `-p gpu-short` (1 h) or `-p gpu-normal`, with
  `-A $ACCOUNT --gres=gpu:1 -c 16 --mem=200G`. GPU only when needed.
- Long or important runs: `sbatch` (survives the orchestrator), logs and exit codes to files
  under `$VARDA_SCRATCH/jobs/`. Blocking `srun` dies with its parent.
- `/tmp` is node-local and invisible to jobs.
- Check your own jobs with `squeue -u $USER` before and after; cancel only jobs you started.

## Models and data (all paths after `source tools/env.sh`)

| | Varda forecaster (= varda) | dev (dev model) |
|---|---|---|
| checkpoint | `$VARDA_SCRATCH/hf/varda-forecaster-det-sgm/model.ckpt` (HF download; same sha256 as `$VARDA_CKPT`) | `$VARDA_DEV_CKPT` |
| converted | `$VARDA_SCRATCH/converted/varda/model.safetensors` | `$VARDA_SCRATCH/converted/dev/model.safetensors` |
| golden (Python project) | `$IE_SCRATCH/golden/varda/{cpu-fp32-pyg,cuda-fp32-triton,cuda-fp16-triton}/` (~40 GB each) | `$IE_SCRATCH/golden/{cpu-fp32-pyg,cuda-fp32-triton,cuda-fp16-triton}/` (1.4 GB each) |
| exported golden | `$VARDA_SCRATCH/golden/varda/<tag>.safetensors` | `$VARDA_SCRATCH/golden/dev/<tag>.safetensors` |
| tolerances | `tolerances/varda.json` | `tolerances/dev.json` |
| sizes | G 1 688 650, H 189 809, E enc/proc/dec 2.85M / 1.52M / 5.07M, 79 in / 67 out | G 40 320, H 10 242, 84 in / 72 out |

Iterate on dev (seconds per forward), then confirm on Varda.
