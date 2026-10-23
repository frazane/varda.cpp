# Rollout: reference runs of the original model (Varda), tolerances, exported files

What the C++ rollout loop is verified against: the original model driven by anemoi-inference's own loop. Semantics of
the loop: `docs/rollout-engine.md` §1; fp16 ensemble tolerance method (v2): §4 here and the `method` field of the
tolerance file. Varda = the Hugging Face forecaster checkpoint (1 688 650 data nodes, 79 inputs, 67 outputs, 66 prognostic,
6-hourly).

Scripts: `python/reference/run_rollout.py`, `run_reference.py`, `compare_golden.py`, `export_rollout.py`. Lock: `python/reference/requirements-ref-varda.lock.txt`.
Everything runs in `$VARDA_SCRATCH/venv-ref-varda`; outputs are under `$VARDA_SCRATCH/rollout/` (`ref/`, `ensemble/`,
`forcings/`, and the job scripts, logs and `exit_code` files).

## 1. Environment

Recipe (also in the header of `python/reference/run_reference.py`):

1. `uv venv --python <python3.12> $VARDA_SCRATCH/venv-ref-varda`.
2. `torch==2.10.0+cu130` and `torchvision==0.25.0+cu130` from `https://download.pytorch.org/whl/cu130` alone
   (`--index-url`, never combined with a PyPI extra index).
3. The rest of the Varda reference lock (`$REF_LOCK`, the lock of the reference venv `$REF_VENV`, minus
   the two torch lines) from PyPI and the lock's git pins, with a constraints file keeping torch fixed. `uv pip freeze`
   afterwards is **identical** to that lock (diff empty).
4. anemoi-inference at `9505dd0926ec3c5dea9308813cc7859bfdad755a` (the commit the Hugging Face release pins in
   its `pyproject.toml`; `git describe`: `0.12.0-16-g9505dd0`), with a constraints file that pins every installed
   package (`name==version`) except the three that had to move.

Freeze diff (before step 4 -> after), complete:

| | before | after | why |
|---|---|---|---|
| anemoi-inference | - | 0.12.1.dev16+g9505dd092 | the point |
| anemoi-utils | 0.5.1 | **0.5.14** | 9505dd09 requires `anemoi-utils[provenance]>=0.5.13` |
| anemoi-transform | 0.1.26 | **0.4.2** | with 0.1.26 the first run fails: `prepare_input_tensor` reads `typed_variables[name].units` and `VariableFromDict` has none (added in 0.3.0). The HF release asks `>=0.4.2`, so 0.4.2 (the lowest) |
| pandas | 3.0.6 | **2.3.3** | every anemoi-transform with `.units` requires `pandas<3` |
| added | - | anytree 2.13.0, astropy 8.0.2, astropy-iers-data, healpy 1.20.1, peekle 0.0.5, pydantic-settings 2.16.1, pyerfa 2.0.1.5, termcolor 3.3.0, tzdata 2026.5, wcwidth 0.9.3 | new dependencies of inference and transform |

torch (2.10.0+cu130), torchvision, triton 3.6.0, numpy, torch_geometric, anemoi-models/graphs/training/datasets
(the reference lock's pins `3c5fc21` / `00396bb4`) did not change. `uv pip check`: compatible. Attempt 1 (everything
pinned) fails to resolve on anemoi-utils; attempt 2 (only anemoi-utils free) installs but fails at the first call, hence
the third pin set above. The model forward does not import anything that moved: the 2-step CPU smoke run reproduces
the reference golden one-step output bitwise (§3).

## 2. anemoi-inference 9505dd09 against eaeb36fa (the version of the dev model's reference rollouts)

`eaeb36fa` is an ancestor of `9505dd09` (36 commits; clones in `$VARDA_SCRATCH/src/anemoi-inference` at 9505dd09
and `$VARDA_SCRATCH/src/ai-eaeb36fa`; `git diff eaeb36fa 9505dd09 -- src`). In the files that define the loop:

- **`runner.py` forecast loop.**
  - The lead time is now `ceil(lead_time / checkpoint.rollout_shift)` steps (before: `timestep * multi_step_output`).
    `rollout_shift` = `output_offsets[-1]` = `multi_step_output * timestep` = 6 h for Varda: same step count. The valid
    dates of output `i` are `start + s * rollout_shift + output_offsets[i]` (before: `start + s*horizon + timestep*(i+1)`):
    identical for Varda (one output per call). The state `step` is `step - rollout_shift + output_offsets[i]`.
  - A new `mid_processors` hook runs on each new output state; none configured, so it is a no-op (`applied` False).
  - `predict_step`, the `torch.autocast(device, dtype=self.autocast)` wrapped around only the model call, `inference_mode`,
    the output assembly (all outputs, `tp` included) and the "no update after the last step" rule are unchanged.
  - `run()` still takes the date of the state; `reference_date` and the input dates are now built from
    `reference_date + metadata.lagged` (`create_input_state(dates=...)`, was `date=`). Our script passes the state
    directly, so no input is created. The `DummyInput`/`DefaultRunner` registry names changed
    (`forecaster`, alias `default`); `RunConfiguration` is unchanged for our use.
- **`tensors.py`.** `copy_prognostic_fields_to_input_tensor` no longer does `roll(-keep_steps, dims=1)` into a new tensor:
  it advances **in place** with `metadata.advance_map`: `inin` = `[(i, i-m) for i in range(m, n)]` copies
  slot 1 into slot 0 (T=2, m=1), then `outin` = `[(m-i-1, n-i-1)]` writes the prognostic outputs into the newest slot.
  The dynamic forcings are written to the same `outin` slots. For Varda (T = 2, one output) the newest slot is
  still `y[..., prognostic]` at `prognostic_input_mask` plus the 5 dynamic forcings, and the oldest slot is still the old
  newest one. The only possible difference: the 8 constant-in-time inputs. `roll` gave the newest slot the oldest slot's
  values; in place the newest slot keeps its own. They are equal here (checked on the 24 recorded CPU windows: the 8 constants are bitwise equal in both slots at every step and
  never change), so the result is the same. The "check" mask logic is unchanged.
- **`forcings.py`.** No change (`ComputedForcings` and the other providers are byte-identical in the diff). The
  dynamic forcings are computed as before from the valid dates and the grid. Varda's five are
  `cos_julian_day, cos_local_time, insolation, sin_julian_day, sin_local_time` at input indices `[4, 6, 8, 23, 25]`.
- **Other:** `precisions.py` unchanged; `inputs/dummy.py` signature (`dates=`); `metadata.py` adds `output_offsets`,
  `rollout_shift`, `advance_map`, `lagged` from the checkpoint's `timesteps` block if present (absent in Varda's
  metadata: defaults apply); `state.py` gets `concat_states`; many I/O changes (netcdf, parallel output, inputs) that this
  work does not touch.
- **Our script adaptation that matters:** `runner.run(lead_time=...)` takes `int` as hours (`to_timedelta`), so the
  original script's `lead_time=n_steps` is right for a 1 h model only. Here `lead_time = checkpoint.timestep * n_steps`
  (6 h steps). Dates follow the runner: the forcings are for `start + 6 h, start + 12 h, ...` (23 dates for 24 steps).

No difference in the numerics of the loop was found for this model. The check is empirical too: step 0 of our CPU run
is bitwise the reference golden `output.pt` (§3).

## 3. Reference rollouts

Input: the reference golden one-step input (anemoi, read-only: `$REF_GOLDEN/cpu_float32_pyg/input.pt`, shape
`(1, 2, 1688650, 79)`), the newest slot dated 2024-06-01T00:00:00 UTC, 24 steps (6 days, outputs dated
06-01T06:00 .. 06-07T00:00). A step is one model call; step 0 sees the input.

**Forcings.** `$VARDA_SCRATCH/rollout/forcings/forcings.pt` (23 dates, 2024-06-01T06:00 .. 06-06T18:00) was recorded
first by running anemoi-inference's loop with a stub model (`--forcings-only`, zeros as the network output; the forcings
depend only on dates and grid; seconds, no GPU) instead of a 24-step GPU run, because the CPU run needs them before it
starts and takes 94 minutes. Every real run **replays** them and, at each call, also lets the runner compute its own
`ComputedForcings` and asserts it is bitwise equal to the replayed array (all runs: "replay bitwise equal to
ComputedForcings"; 23 calls on CPU, 46 with `--runs 2`). So the CPU run and every GPU run saw identical forcings
and these are the runner's own.

| tag (dir) | steps | device / precision / backend | runtime | node |
|---|---|---|---|---|
| `ref/cpu-fp32-pyg` | 24 | CPU (EPYC 7713), 64 threads, fp32 (runner precision "32": autocast disabled), pyg | 5362 s rollout (223 s/step), 5535 s with load; peak RSS 104 GB | CPU node, 300 GB |
| `ref/cuda-fp32-triton` | 24 x 2 runs | A100-SXM4-80G, fp32 (TF32 off), triton | 223.7 s and 204.2 s (9.3 / 8.5 s per step); peak RSS 98 GB; CUDA peak 36.2 GB | GPU node |
| `ref/cuda-fp16-triton` | 24 x 2 runs | A100, runner precision = checkpoint = "16-mixed" (fp16 autocast), triton | 48.1 s and 42.8 s (2.0 / 1.8 s per step); RSS 98 GB; CUDA 23.4 GB | GPU node |
| `ensemble/chunks-mapper-8` | 24 | A100, fp16, triton, `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=8` | 47.8 s | GPU node |
| `ensemble/chunks-processor-4` | 24 | same, `..._PROCESSOR=4` | 63.7 s | GPU node |
| `ensemble/ulp-seed1` | 24 | fp16 triton, input moved by 1 fp32 ulp, random signs, seed 1 | 46.6 s | GPU node |
| `ensemble/ulp-seed2` | 24 | same, seed 2 | 47.1 s | GPU node |
| `ensemble/pyg-backend` | 24 | fp16, **pyg** backend on the GPU (scatter atomics) | 76.0 s; CUDA 41.6 GB | GPU node |

The `ensemble/` rollouts (357 GB) were deleted on 2026-10-23 after calibration; the tolerances derived from them are
committed (`tolerances/rollout/varda.json`, method below), and the commands above regenerate them.

Each run is one job. GPU jobs: `--gres=gpu:1 -c 16 --mem=200G` (peak RSS 98 GB with `--runs 2`, 64 GB with one run: the hooks
keep the 24 windows, 25.6 GB, and the outputs, 10.9 GB, on the host). Smoke tests first: 2 steps CPU (252 s/step)
and 2 steps fp16 GPU with `--runs 2`. Every job exited 0.

**Checks.**
- Step 0 `y` of the CPU run is **bitwise** equal to the reference golden `cpu_float32_pyg/output.pt`
  (max_abs 0; smoke run, 2 steps) and `window[0]` equals the input bitwise: the new venv (anemoi-utils/transform/pandas
  moved, anemoi-inference 9505dd09) reproduces the old golden data.
- `--runs 2` bitwise: **yes** for `cuda-fp32-triton` (y and window, all 24 steps) and `cuda-fp16-triton` (y and window,
  all 24 steps). Asserted in the script.
- The two chunk-env fp16 runs (mapper 8, processor 4) are **bitwise identical** to the reference fp16 run, hence not
  perturbations; dropped from the ensemble (recorded in `fp16_ensemble_dropped_identical`), as for the dev model's
  processor-4 run. (On the dev model the mapper-8 run differed; on Varda it does not. No explanation was investigated.)
- All outputs finite.

### Per-step spreads (`rel_absmax`, contract metric `max|a-r|/max|r|` in float64, reference = `ref/cpu-fp32-pyg`)

Selected steps; the full 24-step table is in `tolerances/rollout/varda.json` (`reference`, `ensemble`).

| step | fp32 y | fp32 y_norm | fp32 window_norm | fp16 y | fp16 y_norm | fp16 window_norm |
|---|---|---|---|---|---|---|
| 0 | 1.52e-07 | 3.55e-06 | 0 | 2.61e-05 | 3.54e-03 | 0 |
| 1 | 2.27e-07 | 5.20e-06 | 5.05e-06 | 3.72e-05 | 3.98e-03 | 4.29e-03 |
| 4 | 3.76e-07 | 2.01e-05 | 1.68e-05 | 6.13e-05 | 2.67e-03 | 3.33e-03 |
| 8 | 5.24e-07 | 1.25e-05 | 2.77e-05 | 7.85e-05 | 3.75e-03 | 3.65e-03 |
| 12 | 5.60e-07 | 1.51e-05 | 1.62e-05 | 9.77e-05 | 3.03e-03 | 4.27e-03 |
| 16 | 6.36e-07 | 1.16e-05 | 1.29e-05 | 1.20e-04 | 4.44e-03 | 2.89e-03 |
| 20 | 6.37e-07 | 1.69e-05 | 1.51e-05 | 1.41e-04 | 6.75e-03 | 5.97e-03 |
| 23 | 5.63e-07 | 1.44e-05 | 1.76e-05 | 1.60e-04 | 8.70e-03 | 8.11e-03 |

fp32 GPU vs CPU: `y` <= 6.4e-7 at every step; `y_norm` peaks at 2.77e-5 (step 7), i.e. above the one-step
contract 1e-5 from step 2 on (the rollout amplifies the 1e-6-level one-step differences; no growth trend, 1e-5..3e-5 from
step 4 to 23). fp16 vs CPU: `y` grows from 2.6e-5 to 1.6e-4, `y_norm` stays between 2.7e-3 and 8.7e-3.
The reference fp16 run vs the fp32 GPU run is within 1% of its distance to CPU at every step (the fp16 error dominates).

fp16 ensemble members (3 perturbed runs; vs CPU `y_norm` at steps 0 / 4 / 8 / 12 / 16 / 23):
ulp-seed1 0.0039 / 0.0035 / 0.0052 / 0.0038 / 0.0029 / 0.0067; ulp-seed2 0.0029 / 0.0058 / 0.0039 / 0.0038 / 0.0035 /
0.0060; pyg-backend 0.0028 / 0.0031 / 0.0030 / 0.0039 / 0.0032 / 0.0071. `y` at step 23: 1.59e-4, 1.68e-4, 1.64e-4.
Against the reference fp16 run (`y_norm`): 0.0031-0.0080 at step 4..23. So independent fp16 runs decorrelate to the level of
the fp16-vs-CPU distance already at step 0: the fp16 trajectories do not track each other bit-closely, as found on
the dev model, but over 24 six-hourly steps Varda's growth is slow (y_norm roughly doubles, 3.5e-3 -> 8.7e-3 for the reference fp16 run).

## 4. Tolerances: `tolerances/rollout/varda.json`

Written by `run_rollout.py --calibrate ref/cpu-fp32-pyg ref/cuda-fp32-triton ref/cuda-fp16-triton --fp16-ensemble
ensemble/{chunks-mapper-8,chunks-processor-4,ulp-seed1,ulp-seed2,pyg-backend}` (one CPU job), keyed by the
checkpoint identity (uuid `e8c577ed-806e-47c5-b257-72c5edcf4cb5`, run id `98d4e297...`, sha256 `1ad7a7a6...a9c7d8`, taken from
the HF file). Method (the file's `method`, as for the dev model): 24 steps, keys
`y, y_norm, window, window_norm`, tables `fast-fp32-vs-cpu`, `fast-fp16-vs-cpu`, `fast-fp16-vs-ref-fp16`,
`fp16_allowance`, `fp16_mean_tolerance`, `fp16_leave_one_out`, `fp16_criterion`; floors 1e-6 (exact) and 1e-5 (fp32);
the fp16 ones are 1.75 x the envelope of {reference fp16 + members}; step 0 is the strict one-step value. Differences
from the dev model's file: (a) the ensemble has **3** members, not 4 (the two chunk-env runs are bitwise the reference);
(b) the leave-one-out worst ratio is 0.74-0.96 (< 1) for both families, so `fp16_allowance` is the floor **1.0** for both
(no leniency was needed; the dev model had 1.7 / 1.9) and `fp16_mean_tolerance` is the per-key mean bound
(vs CPU: y 1.8e-4, y_norm 8.1e-3, window 1.7e-4, window_norm 7.2e-3). Note the envelope rests on only 4 runs
(reference + 3) vs CPU and 3 vs the reference fp16, so it is thin: treat the factor 1.75 as the only safety margin.

Selected values (steps 0 / 1 / 4 / 8 / 12 / 16 / 20 / 23):

| table, key | tolerance |
|---|---|
| fp32-vs-cpu, y | 1.1e-5 at every step |
| fp32-vs-cpu, y_norm | 1.1e-5, 1.6e-5, 6.1e-5, 3.8e-5, 4.6e-5, 3.5e-5, 5.1e-5, 4.4e-5 (3 x the reference's own spread) |
| fp32-vs-cpu, window_norm | 1e-6, 1.6e-5, 5.1e-5, 8.4e-5, 4.9e-5, 3.9e-5, 4.6e-5, 5.3e-5 |
| fp16-vs-cpu, y | 4.6e-5, 6.6e-5, 1.2e-4, 1.5e-4, 1.8e-4, 2.2e-4, 2.7e-4, 3.0e-4 |
| fp16-vs-cpu, y_norm | 6.2e-3, 1.3e-2, 1.1e-2, 9.1e-3, 6.9e-3, 7.8e-3, 1.2e-2, 1.6e-2 |
| fp16-vs-ref-fp16, y | 5.3e-5, 7.5e-5, 1.3e-4, 1.6e-4, 2.0e-4, 2.4e-4, 2.9e-4, 3.3e-4 |

`--check`: `run_rollout.py --calibrate ... --check tolerances/rollout/varda.json` recomputes into a temporary file
(`$VARDA_SCRATCH/rollout/ref/tolerances-check-varda.json`) and exits 1 unless `steps, keys, fast-fp32-vs-cpu, fast-fp16-vs-cpu, fast-fp16-vs-ref-fp16,
fp16_allowance, fp16_mean_tolerance` are equal. Result (the calibration job): **reproduces the committed file exactly** (exit 0), all 24 steps. (An earlier version compared the file with itself when
`--check` was given without `--tolerances`; fixed.)

## 5. Files

Per reference rollout directory `D` (`ref/<tag>` or `ensemble/<tag>`): `rollout.pt` (torch),
`manifest.json` (command, job id, device, versions, determinism, timings), `rollout.safetensors` (below). `forcings/forcings.pt`
holds the 23 forcing arrays. All are float32 unless stated.

`D/rollout.safetensors` (`python/reference/export_rollout.py`; container of `docs/format.md` §1, 35.7 GiB; 8-byte dtypes
first, then by name; checked by reading back with `python/safetensors_io.py` and comparing each tensor bitwise with
`rollout.pt`: done for `cuda-fp32-triton` and `cuda-fp16-triton`; the others were written by the same code path, atomically):

| tensor | dtype, shape | content |
|---|---|---|
| `x0` | F32 [2, G, 79] | the input window (oldest slot first), physical units, model-input order |
| `y` | F32 [N, G, 67] | step `s` output (physical, after residual, bounding, de-normalisation), model-output order |
| `window` | F32 [N, 2, G, 79] | the window step `s` sees (`window[0] == x0`), physical |
| `forcings` | F32 [N-1, G, 5] | dynamic forcings for the valid date of output `s` (`dates[s]`), written into the newest slot after step `s`, in `forcing_vars` order (G = 1 688 650, N = 24) |
| `forcing_index` | I64 [5] | input-variable index of each forcing column: `[4, 6, 8, 23, 25]` = `cos_julian_day, cos_local_time, insolation, sin_julian_day, sin_local_time` |
| `prog_in` | I64 [66] | input indices that receive the prognostic outputs |
| `prog_out` | I64 [66] | output indices fed back (same order as `prog_in`); `tp` (output 18) is not |
| `kept_in` | I64 [8] | the constant-in-time inputs, never rewritten |
| `mul_in`, `add_in` | F32 [79] | normaliser (`x*mul + add`) of the inputs |
| `mul_out`, `add_out` | F32 [67] | same for the outputs (`y_norm = y*mul_out + add_out`) |

`__metadata__` (all strings): `varda_rollout` = "1", `tag`, `steps`, `start_date` (date of `x0[1]`), `timestep` ("6:00:00"),
`dates` (JSON list of N-1 ISO dates of the forcings = valid dates of outputs 0..N-2; the last step needs none),
`forcing_vars`, `in_vars` (79 names), `out_vars` (67), `kept_vars` (JSON lists), `run` (JSON: device, backend, precision,
autocast, perturbation seed, chunk env, job id, command, determinism, timings, checkpoint path).
Singleton dims of `rollout.pt` (`y` [N,1,1,1,G,67], `window` [N,1,2,G,79], `x0` [1,2,G,79], `forcings` [N-1,1,G,5]) are dropped.
The C++ loop is verified like this: start from `x0`, feed `forcings[s]` after each step, compare `y[s]` and the window
it sees per step with `rel_absmax` against these files, using the tables of §4. Do not compare a C++ fp16 run to
a single reference trajectory without the ensemble tolerance.

## 6. Surprising / to know

- 223 s per CPU step (the brief's ~200 s). The 24-step CPU reference fits in 94 min, so tolerances cover all 24 steps.
- `anemoi-inference` at the HF pin does not run with the Varda reference lock: anemoi-utils, anemoi-transform and pandas
  must move (§1). The anemoi-inference pin alone says `anemoi-transform>0.1.13`, which is wrong in practice.
- The fp32 GPU rollout's `y_norm` spread (up to 2.8e-5) is above the 1e-5 one-step number: the tolerance table has it
  (3 x spread), as for the dev model.
- The fp16 ensemble tolerance needed no leniency here (allowance 1.0); with 3 members its margin is statistical, not proven.
