# Rollout engine: the autoregressive loop of varda.cpp

How varda.cpp runs N forecast steps with anemoi-inference's semantics, the two implementations
(host-level oracle, device-resident CUDA loop), how they are verified against the original
model driven by anemoi-inference's own loop, and what they cost.

The semantics are those of anemoi-inference's `Runner.forecast` (§1); the dynamic forcings are
`docs/forcings.md`. The references are rollouts of the original model with anemoi-inference's own loop:
the dev model (anemoi-inference eaeb36fa, `$REF_SCRATCH/rollout/`, read-only) and Varda (this project's
reference driver `python/reference/run_rollout.py`, `$VARDA_SCRATCH/rollout/`).

| file | content |
|---|---|
| `src/varda/rollout.h/.cpp` | `StateUpdate` (the window update), `valid_date`, `Rollout` (host-level loop, any backend) |
| `src/varda/cuda/rollout_cuda.h/.cu` | `cuda::DeviceRollout` (device-resident loop, one CUDA graph per step), `cuda::rollout_update` (kernel) |
| `apps/varda-rollout.cpp` | run N steps, computed or replayed forcings, `--compare` with per-step metrics and the rollout tolerances, bitwise self-checks, benchmark, JSON report |
| `tests/test_rollout.cpp` | state update on tiny synthetic windows (T = 2..5, slot values that differ) vs the literal upstream definitions (9505dd0 `advance_map`, eaeb36fa `roll(-1)`), both constant sources, invalid index sets, dates, real configs |
| `tests/test_rollout_cuda.cu` | device update kernel bitwise equal to the host update, both constant sources (T = 2, 3, 4) |
| `python/export_rollout.py` | the dev model's reference rollouts (`rollout.pt`) to the rollout file read by the app |

## 1. Semantics

The window `(1, T, G, V_in)` holds **physical** fp32 values in model-input order, oldest slot first.
Step `s` (0-based):

1. `y = predict_step(window)`: `(1, 1, 1, G, V_out)` physical, after the residual, the bounding and
   de-normalisation. The whole window is re-normalised every step (`x*mul`, then `+add`), so a fed-back
   prognostic value makes the round trip `((y - add)/mul)*mul + add`.
2. Unless `s` is the last step (anemoi-inference prepares no input after the last step), update the window
   in place, as anemoi-inference **9505dd0** (the version the Hugging Face release pins; `docs/rollout.md` §2)
   does with `Metadata.advance_map` (`n` = T input steps, `m` = 1 output step):
   - `inin = [(i, i-m) for i in range(m, n)]`, executed in list order: slot `t` <- slot `t+1` for
     `t = 0 .. T-2`, ascending, so every source is read before it is overwritten: a shift down by one. The
     newest slot is not written by it. (T = 2: `(1->0)`; T = 3: `(1->0) (2->1)`; T = 5: `(1->0) .. (4->3)`;
     checked against the literal formula in `test_rollout` for T = 2..5.)
   - `outin = [(m-i-1, n-i-1)]` = `(0 -> T-1)`: newest slot, prognostic inputs (`indices.residual_input`) <- `y`'s
     prognostic outputs (`indices.residual_output`, same order): physical and bounded; diagnostic outputs (`tp`)
     are not fed back; dynamic forcings (config `forcings.dynamic_input_index`) <- `compute_forcings(valid date of
     output s)`, or the replayed array (also written to the `outin` slot);
   - newest slot, constant inputs (`forcings.constant_input_index`): never written, so they **keep the newest
     slot's own values** (`ConstantSource::Newest`, the default).

   `ConstantSource::Oldest` (`--constants oldest`) is the explicit option for anemoi-inference **eaeb36fa**
   (the dev model's references): there `copy_prognostic_fields_to_input_tensor` did `torch.roll(-1)` into a
   new tensor, so the constant inputs of the newest slot were the **oldest** slot's. The two semantics differ only
   if a constant input differs between slots; for real inputs they are equal in every slot and never change
   (checked for every window of both models' references and of our runs, §4), so both give bitwise the same
   rollout. `Metadata.advance_map` can also be overridden by a checkpoint's `timesteps.advance_map`; Varda's and
   the dev model's metadata have none, so the formula applies. Models with more than one output step are
   rejected (`check_rollout_model`).
3. The valid date of output `s` is `date0 + (s + 1) * timestep_hours`, `date0` the date of the newest input
   slot (`x0[T-1]`): Varda 6 h, the dev model 1 h.

`StateUpdate::from_config` builds one source per input channel (prognostic: an output index; forcing: a
column; constant: kept). Every input must be exactly one of the three, as anemoi-inference's `check` mask
requires (Varda: 79 = 66 + 5 + 8; the dev model: 84 = 71 + 5 + 8); anything else, or a config without the
`forcings` section, is rejected with an error naming the channel or the section. Coordinates for the
computed forcings are the file's `grid.latitudes` / `grid.longitudes` (F64, `Model::latitudes()`), so the
rollout needs a file that carries them (every file now under `converted/`); `check_rollout_model` says so.

## 2. API

```cpp
#include "varda/rollout.h"
varda::RolloutRequest q;
q.x0 = x0;                       // (1, T, G, V_in) physical, host
q.date0 = varda::parse_iso8601_utc("2024-06-01T00:00:00");   // newest input slot
q.n_steps = 24;
q.replay_forcings = nullptr;     // default: computed; or [n_steps-1, G, n_dynamic] in config forcings.dynamic order
q.on_step = [](int s, const varda::DateTime& valid, const float* y) { ... };   // y physical, host, valid during the call
q.on_window = nullptr;           // optional: the window step s sees (verification)

varda::Rollout(*model).run(q);                // host-level: Model::predict_step, any device
// optional last constructor argument: varda::ConstantSource::Oldest (eaeb36fa's roll; default Newest = 9505dd0)
varda::cuda::DeviceRollout(*model).run(q);    // CUDA, device-resident (rollout_cuda.h); same results bitwise
```

`Rollout` is the readable oracle: a window and an output buffer on the host, `Model::predict_step`, then
`compute_forcings` and `StateUpdate::apply` (OpenMP over grid points, pure copies). `DeviceRollout`
also exposes `step_ms()` (GPU time per step of the last run), `wall_s()`, `forcing_host_s()`,
`time_forward(iters)` (replays of the forward-only graph) and its memory use.

## 3. Device-resident loop (`DeviceRollout`)

Engineered like a decode loop, on the CUDA backend's device-level API (`src/varda/cuda/cuda_backend.h`:
`normalise`, `forward_normalised`, `denormalise`, `prepare`), which only enqueues work, so it is capturable.

**Buffers** (all allocated once, in the constructor): window `win` and normalised window `xn`
`(T, G, V_in)`, normalised output `yn`, two physical outputs `y[0..1]` and two forcing slots `f[0..1]`
`(G, F)`; pinned host `hy[0..1]` and `hf[0..1]`; the update's per-channel `kind` / `src` tables. Varda:
3.56 GB of device memory and 0.97 GB pinned on top of the backend's; the dev model 0.09 / 0.02 GB
(plus one pinned window copy when `on_window` is used).

**Graphs.** After `prepare()` (one eager forward: every cuBLASLt plan exists), four graphs are captured
on the rollout's own stream: `graph[p][u]` = `normalise(win -> xn)`, `forward_normalised(xn -> yn)`,
`denormalise(yn -> y[p])` and, with `u = 1`, `rollout_update(win, y[p], f[p])`. Parity `p = s % 2`
double-buffers the output and the forcing slot; `u = 0` is the last step (no update). One graph launch
per step.

**The update kernel** (`rollout_update`): one thread per `(g, c)` of a slot; it shifts the column's `T`
values down by one (`inin`) and writes the newest one from its source (`y[g, src[c]]` or `f[g, src[c]]`); a
constant input keeps the newest slot's value, or with `ConstantSource::Oldest` gets the old oldest value. Each
thread owns its column, so the in-place update is race free; there is no arithmetic, so it is bitwise the
host update (`test_rollout_cuda`, both constant sources).

**Per step `s`** (three streams: main, D2H, H2D; no host synchronisation except what the callback needs):

```
main:  wait(forcing s uploaded)  wait(D2H of step s-2 out of y[p] done)  launch graph[p][s<n-1]
D2H:   wait(step s)  y[p] -> hy[p]
host:  compute forcings of step s+1 into pinned hf[1-p]   <- overlaps step s on the GPU
H2D:   wait(step s-1, the last reader of f[1-p])  hf[1-p] -> f[1-p]
host:  wait(D2H of step s-1)  on_step(s-1, hy[1-p])       <- the callback also overlaps step s
```

The host never waits for the GPU except for the copy the callback consumes, one step behind the GPU,
so the next step is always queued before the callback runs. `on_window` (verification only) adds a
window copy and a stream sync per step.

**Why bitwise equal to the host loop.** Both run the same backend kernels on the same inputs: the
host loop calls `Model::predict_step`, which the backend guarantees bitwise equal to
`predict_step_device` = `denormalise(forward_normalised(normalise(x)))` (`cuda_backend.h`); the update and
the forcing upload only copy values; the forcings are the same `compute_forcings` call (or the same
replayed array). Checked on both models at every step (§4).

**Not done, on purpose.** A ring-indexed window (no roll copy) would need a gather or a forward that
takes the slot order: the update costs ~0.05 ms per step on the dev model (§5), not worth it. Forcings
computed on the device: the host computation is hidden behind the step (0.4 ms on the dev model, 10.4 ms on
Varda, vs a 32 ms / 727 ms fp16 step) and stays bitwise anemoi's (glibc libm in double, `docs/forcings.md` §7).

## 4. Verification

### Method

`varda-rollout --compare REF` (`apps/varda-rollout.cpp` header): per step `s`, the contract metric `rel_absmax` (float64,
`docs/format.md` §2) of `y`, `y_norm = y*mul_out + add_out`, `window` (the window step `s` sees) and
`window_norm`; tolerances from `tolerances/rollout/<model>.json` (found by checkpoint uuid; the run fails
without one), table by reference and precision:

| run | vs `cpu-fp32-pyg` | vs `cuda-fp16-triton` |
|---|---|---|
| CPU fp32, CUDA fp32 | `fast-fp32-vs-cpu`: every step within its tolerance | - |
| CUDA fp16 | `fast-fp16-vs-cpu` | `fast-fp16-vs-ref-fp16` |

fp16 (the file's `fp16_criterion`, tolerances v2): step 0 within the one-step tolerance; steps >= 1
within `fp16_allowance` x the per-step tolerance (the dev model 1.7 vs CPU, 1.9 vs the reference fp16);
per key, the mean over steps within `fp16_mean_tolerance`. Entries above the bare per-step tolerance are
counted. The C++ CPU fp32 path is held to the fp32 table, as for one step (`docs/format.md` §4).

Self-checks of the same run (exit 1 if any fails):
- dates: `date0 + (s+1)*timestep` equals the reference's recorded `dates[s]`, and `date0` its `start_date`;
- `--check-forcings`: `compute_forcings` at each valid date bitwise equal to the reference's recorded
  forcings (anemoi's `ComputedForcings` inside the real runner loop), and the forcing columns of the newest
  slot of every window the rollout feeds (steps >= 1) bitwise equal to them;
- `--verify-replay`: a second rollout with replayed forcings (the recorded array): `y` bitwise equal at every
  step, windows equal (64-bit hash of the bytes);
- `--verify-host`: the host-level `Rollout` on the same model and backend: `y` bitwise equal at every step,
  windows equal;
- `--verify-constants`: a second rollout with the other constant source: `y` bitwise equal at every step,
  windows equal; and, always, `constants_across_slots`: the number of windows (this run's, and each reference's
  recorded ones) whose constant inputs are not bitwise equal in every slot.

`varda-rollout --judge REPORT.json [--tolerances F]` re-applies the tolerances to the metrics recorded in an earlier
report (used for Varda, whose tolerance file arrived after the runs).

Input: the reference's `x0` (the golden one-step input `in.x`) with its `start_date` 2024-06-01T00:00, so
step 0 of the CPU run is the one-step parity case.

### Dev model (24 steps, 1 h)

References: `$REF_SCRATCH/rollout/ref/cpu-fp32-pyg`, `.../v2/ref/cuda-fp32-triton`, `.../v2/ref/cuda-fp16-triton`,
exported by `python/export_rollout.py` (run `dev-export`, `--verify`: every tensor bitwise equal to `rollout.pt`, and
`forcings.pt` equal to the rollout's forcings) to `$VARDA_SCRATCH/rollout-engine/ref/dev/<tag>/rollout.safetensors`.

**CPU fp32** vs `cpu-fp32-pyg`, table `fast-fp32-vs-cpu` (CPU node, 64 threads `OMP_PLACES=cores
OMP_PROC_BIND=spread` on `-c 128`, run `dev-cpu`): **PASS**, worst 0.56 x tolerance (`window_norm`, step 6),
0 of 96 entries above.

**CUDA fp32** vs `cpu-fp32-pyg`, `fast-fp32-vs-cpu` (A100-SXM4-80GB, graph on, run `dev-gpu`): **PASS**, worst
0.58 x (`window_norm`, step 15), 0 of 96 above.

| step | CPU y | CPU y_norm | CPU window_norm | CUDA fp32 y | CUDA fp32 y_norm | CUDA fp32 window_norm | tol y | tol y_norm | tol window_norm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.50e-07 | 4.46e-07 | 0 | 7.48e-08 | 1.45e-06 | 0 | 1.1e-05 | 1.1e-05 | 1e-06 |
| 1 | 2.24e-07 | 2.35e-06 | 1.71e-06 | 2.24e-07 | 2.85e-06 | 1.71e-06 | 1.1e-05 | 1.1e-05 | 1.1e-05 |
| 2 | 2.24e-07 | 4.07e-06 | 2.76e-06 | 2.24e-07 | 4.92e-06 | 3.07e-06 | 1.1e-05 | 2.5e-05 | 1.1e-05 |
| 4 | 2.98e-07 | 7.54e-06 | 4.14e-06 | 2.98e-07 | 6.47e-06 | 5.74e-06 | 1.1e-05 | 2.3e-05 | 1.7e-05 |
| 8 | 2.98e-07 | 1.10e-05 | 5.86e-05 | 3.72e-07 | 1.86e-05 | 6.68e-05 | 1.1e-05 | 4.5e-05 | 2.0e-04 |
| 12 | 3.72e-07 | 1.95e-05 | 3.38e-05 | 3.72e-07 | 3.94e-05 | 5.30e-05 | 1.1e-05 | 1.1e-04 | 1.2e-04 |
| 16 | 4.45e-07 | 2.42e-05 | 3.16e-05 | 5.93e-07 | 3.91e-05 | 6.18e-05 | 1.1e-05 | 2.4e-04 | 1.8e-04 |
| 20 | 4.44e-07 | 3.79e-05 | 4.69e-05 | 4.44e-07 | 3.47e-05 | 5.19e-05 | 1.1e-05 | 3.6e-04 | 3.6e-04 |
| 23 | 7.39e-07 | 8.27e-05 | 6.06e-05 | 5.18e-07 | 8.81e-05 | 8.54e-05 | 1.1e-05 | 1.2e-03 | 1.2e-03 |

(`window` is the previous step's `y` plus exact forcings and constants, so it follows `y`; all 24 rows of
all four keys are in the runs' `check-*.json` reports.) Step-0 `y_norm` is 4.46e-7 on the CPU (the one-step
`out.y` parity) and 1.45e-6 on CUDA fp32.

**CUDA fp16** (run `dev-gpu`): **PASS** against both references.

| step | y vs CPU | tol | y_norm vs CPU | tol | window_norm vs CPU | tol | y vs ref fp16 | tol | y_norm vs ref fp16 | tol |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.66e-06 | 1.2e-05 | 1.25e-03 | 1.9e-03 | 0 | 1e-06 | 5.98e-06 | 1.4e-05 | 8.69e-04 | 2.2e-03 |
| 1 | 1.13e-05 | 2.0e-05 | 1.07e-03 | 2.6e-03 | 1.64e-03 | 2.8e-03 | 1.08e-05 | 2.2e-05 | 1.31e-03 | 2.3e-03 |
| 4 | 1.53e-05 | 3.3e-05 | 2.04e-03 | 4.3e-03 | 3.24e-03 | 6.2e-03 | 1.56e-05 | 3.3e-05 | 2.13e-03 | 4.6e-03 |
| 8 | 4.07e-05 | 1.1e-04 | 6.36e-03 | 0.016 | 2.37e-02 | 0.055 | 3.11e-05 | 1.2e-04 | 1.00e-02 | 0.018 |
| 12 | 1.55e-04 | 2.1e-04 | 2.18e-02 | 0.041 | 6.61e-02 | 0.13 | 1.40e-04 | 2.2e-04 | 2.17e-02 | 0.041 |
| 14 | 2.22e-04 | 1.8e-04 | 1.48e-02 | 0.036 | 4.25e-02 | 0.08 | 1.89e-04 | 2.0e-04 | 1.48e-02 | 0.030 |
| 16 | 2.79e-04 | 4.2e-04 | 1.17e-02 | 0.057 | 2.51e-02 | 0.084 | 2.53e-04 | 4.1e-04 | 3.24e-02 | 0.064 |
| 20 | 3.76e-04 | 7.8e-04 | 4.39e-02 | 0.077 | 2.71e-02 | 0.084 | 3.46e-04 | 7.6e-04 | 4.39e-02 | 0.11 |
| 23 | 3.84e-04 | 6.6e-04 | 4.70e-02 | 0.083 | 2.41e-02 | 0.11 | 3.54e-04 | 6.5e-04 | 4.69e-02 | 0.11 |

- vs CPU fp32 (`fast-fp16-vs-cpu`): worst 1.23 x the per-step tolerance (`y`, step 14), 4 of 96 entries above it
  (allowance 1.7); mean over steps / mean tolerance: y 0.61, y_norm 0.56, window 0.59, window_norm 0.41.
- vs the reference's fp16 rollout (`fast-fp16-vs-ref-fp16`): worst 1.02 x (`y`, step 15), 2 of 96 above
  (allowance 1.9); means 0.55, 0.50, 0.53, 0.45.

**Forcings and self-checks** (all three runs): dates 23 of 23 equal; computed forcings bitwise equal to the
recorded ones, 0 of 23 x 40 320 x 5 elements differ; the forcing columns of every window fed to the model
bitwise equal to the reference's recorded forcings (steps 1-23, against all three references); computed vs
replayed forcings: `y` bitwise equal at all 24 steps, windows equal; CUDA fp32 and fp16: device loop vs
host-level `Rollout`: `y` bitwise equal at all 24 steps, windows equal.

### Varda (24 steps, 6 h)

References: the original Varda forecaster driven by anemoi-inference's own loop (`python/reference/run_rollout.py`,
24 steps from the golden input, forcings recorded once and replayed, `$VARDA_SCRATCH/rollout/ref/{cpu-fp32-pyg,
cuda-fp32-triton,cuda-fp16-triton}/rollout.safetensors`); tolerances `tolerances/rollout/varda.json` (sha256
`7b020cd8...c5f95` when judged; calibrated with the same method as the dev model's (`docs/rollout.md` §4); its
`fp16_allowance` is 1.0 for both fp16 tables, i.e. no extra factor). The GPU and CPU runs were made before the file
existed; their recorded per-step metrics were judged with `varda-rollout --judge REPORT.json` (same table choice and
criterion as a live run; `*.judged.json` next to each report).

**CPU fp32**, 24 steps (~60 s per forward, 73 s per step with the window copy and comparison; CPU node, 64 threads
`OMP_PLACES=cores OMP_PROC_BIND=spread`, `-c 128`; run `const-cpu`, constants in place) vs `cpu-fp32-pyg`,
`fast-fp32-vs-cpu`: **PASS**, worst 0.58 x (`y_norm`, step 23), 0 of 96 above. (The earlier 8-step run, `varda-cpu-roll`,
roll semantics: worst 0.29 x, 0 of 32 above; its 8 steps are bitwise the first 8 of this run's metrics.)

**CUDA fp32**, 24 steps (run `varda-gpu-vscpu`) vs `cpu-fp32-pyg`, `fast-fp32-vs-cpu`: **PASS**, worst 0.50 x (`y_norm`, step 8),
0 of 96 above.

| step | CPU y | CPU y_norm | CPU window_norm | CUDA fp32 y | CUDA fp32 y_norm | CUDA fp32 window_norm | tol y | tol y_norm | tol window_norm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.52e-07 | 1.64e-06 | 0 | 1.52e-07 | 3.13e-06 | 0 | 1.1e-05 | 1.1e-05 | 1e-06 |
| 1 | 2.27e-07 | 4.63e-06 | 2.23e-06 | 2.27e-07 | 7.40e-06 | 4.63e-06 | 1.1e-05 | 1.6e-05 | 1.6e-05 |
| 2 | 2.26e-07 | 4.85e-06 | 2.77e-06 | 2.26e-07 | 6.66e-06 | 4.63e-06 | 1.1e-05 | 3.9e-05 | 1.8e-05 |
| 4 | 3.76e-07 | 1.48e-05 | 3.17e-06 | 3.76e-07 | 2.96e-05 | 1.02e-05 | 1.1e-05 | 6.1e-05 | 5.1e-05 |
| 8 | 4.49e-07 | 1.10e-05 | 1.16e-05 | 7.49e-07 | 1.91e-05 | 1.74e-05 | 1.1e-05 | 3.8e-05 | 8.4e-05 |
| 12 | 5.23e-07 | 1.25e-05 | 1.17e-05 | 8.59e-07 | 1.65e-05 | 1.43e-05 | 1.1e-05 | 4.6e-05 | 4.9e-05 |
| 16 | 5.24e-07 | 1.06e-05 | 9.90e-06 | 7.85e-07 | 1.40e-05 | 1.40e-05 | 1.1e-05 | 3.5e-05 | 3.9e-05 |
| 20 | 5.25e-07 | 1.75e-05 | 1.57e-05 | 5.62e-07 | 1.66e-05 | 1.29e-05 | 1.1e-05 | 5.1e-05 | 4.6e-05 |
| 23 | 5.25e-07 | 2.53e-05 | 2.42e-05 | 5.63e-07 | 1.56e-05 | 2.37e-05 | 1.1e-05 | 4.4e-05 | 5.3e-05 |

Step 0 of the CPU run is the one-step parity case (`in.x` of the golden data): `y` 1.52e-7, `y_norm` 1.64e-6.
For information, CUDA fp32 vs the reference's own `cuda-fp32-triton` rollout: `y` <= 9.0e-7, `y_norm` <= 1.7e-5 over
24 steps (run `varda-gpu-fp32`).

**CUDA fp16**, 24 steps (run `varda-gpu-vscpu`): **PASS** against both references, with no step needing an allowance.

| step | y vs CPU | tol | y_norm vs CPU | tol | window_norm vs CPU | tol | y vs ref fp16 | tol | y_norm vs ref fp16 | tol | window_norm vs ref fp16 | tol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.96e-05 | 4.6e-05 | 3.54e-03 | 6.2e-03 | 0 | 1e-06 | 2.31e-05 | 5.3e-05 | 2.91e-03 | 7.1e-03 | 0 | 1e-06 |
| 1 | 3.48e-05 | 6.6e-05 | 6.51e-03 | 0.013 | 5.11e-03 | 0.011 | 3.51e-05 | 7.5e-05 | 4.61e-03 | 0.011 | 4.55e-03 | 8.9e-03 |
| 2 | 4.36e-05 | 8.8e-05 | 2.28e-03 | 0.016 | 5.11e-03 | 0.011 | 4.99e-05 | 9.7e-05 | 3.38e-03 | 0.012 | 4.55e-03 | 8.9e-03 |
| 4 | 5.60e-05 | 1.2e-04 | 2.74e-03 | 0.011 | 4.15e-03 | 0.011 | 6.26e-05 | 1.3e-04 | 3.98e-03 | 0.014 | 2.88e-03 | 6.7e-03 |
| 5 | 7.04e-05 | 1.2e-04 | 2.82e-03 | 6.8e-03 | 2.33e-03 | 4.6e-03 | 6.45e-05 | 1.4e-04 | 2.96e-03 | 7.4e-03 | 3.69e-03 | 4.4e-03 |
| 8 | 7.85e-05 | 1.5e-04 | 3.66e-03 | 9.1e-03 | 3.19e-03 | 7.7e-03 | 7.65e-05 | 1.6e-04 | 5.45e-03 | 9.1e-03 | 5.02e-03 | 8.7e-03 |
| 12 | 9.96e-05 | 1.8e-04 | 2.90e-03 | 6.9e-03 | 3.42e-03 | 0.011 | 8.66e-05 | 2.0e-04 | 3.40e-03 | 7.7e-03 | 3.79e-03 | 0.011 |
| 13 | 1.03e-04 | 1.9e-04 | 4.68e-03 | 6.4e-03 | 2.99e-03 | 7.5e-03 | 8.49e-05 | 2.2e-04 | 3.95e-03 | 8.2e-03 | 3.79e-03 | 0.011 |
| 16 | 1.20e-04 | 2.2e-04 | 3.46e-03 | 7.8e-03 | 3.01e-03 | 7.0e-03 | 8.77e-05 | 2.4e-04 | 4.16e-03 | 8.9e-03 | 3.43e-03 | 7.3e-03 |
| 20 | 1.47e-04 | 2.7e-04 | 5.20e-03 | 0.012 | 4.42e-03 | 0.011 | 1.04e-04 | 2.9e-04 | 4.81e-03 | 0.014 | 3.10e-03 | 0.012 |
| 23 | 1.63e-04 | 3.0e-04 | 6.51e-03 | 0.016 | 6.44e-03 | 0.015 | 1.15e-04 | 3.3e-04 | 3.87e-03 | 0.018 | 3.67e-03 | 0.017 |

- vs CPU fp32 (`fast-fp16-vs-cpu`): worst 0.73 x the per-step tolerance (`y_norm`, step 13), 0 of 96 above; mean over
  steps / mean tolerance: y 0.54, y_norm 0.49, window 0.54, window_norm 0.53.
- vs the reference's fp16 rollout (`fast-fp16-vs-ref-fp16`): worst 0.84 x (`window_norm`, step 5), 0 of 96 above; means
  0.53, 0.50, 0.54, 0.55.

Unlike the dev model (1 h steps, 24 h), the 6-hourly Varda fp16 rollout does not decorrelate from the references
within 6 days: the error vs CPU grows only from 3e-5 to 1.6e-4 in `y` and stays at 2-7e-3 in `y_norm`.

**Forcings and self-checks**: dates 23 of 23 equal (6 h steps from 2024-06-01T00 to 2024-06-07T00, across a month);
computed forcings bitwise equal to the recorded ones, 0 of 23 x 1 688 650 x 5 = 194 194 750 elements differ (checked in
every run against each reference's recorded array); the forcing columns of every window fed to the model bitwise equal
to them (steps 1-23); computed vs replayed forcings: `y` bitwise equal at every step and windows equal
(CUDA fp32 and fp16 24 steps, runs `varda-gpu-fp32`, `varda-gpu-fp16`; CPU 8 steps, run `varda-cpu-verify`; the constant-source change below
leaves all of these bitwise unchanged); device loop vs host-level `Rollout` on the
same backend: `y` bitwise equal at all 24 steps and windows equal (fp32 run `varda-gpu-fp32`, fp16 run `varda-gpu-fp16`).

### Constant inputs: in place (9505dd0) vs roll (eaeb36fa)

The rollout first used eaeb36fa's `roll(-1)` (committed as 83e6205); the default is now 9505dd0's in-place
`advance_map` (§1), with the roll kept as `ConstantSource::Oldest`. Verification of the change:

- `test_rollout` (CPU, run `const-cpu` and the login node): for T = 2, 3, 4, 5 and both sources, three consecutive updates
  bitwise equal to the literal upstream definitions executed on whole slots (the `inin` / `outin` lists built from the
  formula and applied in order, or `roll(-1)` then `outin`), on windows whose values differ between slots; explicit
  per-slot expectations (slot `t` <- old slot `t+1`; constants keep the newest slot's value in place, take the oldest
  slot's with the roll; after a second update in place they are still the newest slot's original values); with
  constants equal in all slots, both sources give the same window; the default is `Newest`.
- `test_rollout_cuda` (run `const-gpu`): the device kernel equals the host update for both sources (T = 2, 3, 4), and the
  two sources really differ on those windows (so the flag reaches the kernel).
- Are the references' constants equal across slots? Yes: in every recorded window of every reference (the dev model's
  `cpu-fp32-pyg`, `cuda-fp32-triton`, `cuda-fp16-triton`, made with eaeb36fa; Varda `cpu-fp32-pyg`,
  `cuda-fp16-triton`, made with 9505dd0) and of every one of our runs, the 8 constant inputs are bitwise equal in both
  slots, at all 24 steps (`constants_across_slots` in the reports). So both semantics give the same rollout here.
- Rollouts with the other source (`--verify-constants`): `y` bitwise equal at all 24 steps and windows equal, for
  the dev model CUDA fp16 and CPU fp32 and Varda CUDA fp16 (runs `const-gpu`, `const-cpu`); device vs host still bitwise
  equal (the dev model and Varda fp16, 24 steps, run `const-gpu`).
- Against the committed binary (83e6205 build, roll semantics, the backend before the later review fixes): the
  24-step `y` files of the dev model fp16 and Varda fp16 are byte-identical (`cmp`), and every per-step metric in the
  new reports equals the earlier one (runs `dev-gpu`, `varda-gpu-vscpu`). The verdicts above stand unchanged.

## 5. Speed

Varda, A100-SXM4-80GB, warm, 24 steps, same protocol (runs `varda-gpu-fp32` and `varda-gpu-fp16`; `bench-*.json`):

| precision | graph | forcings | GPU ms/step | wall ms/step | forward-only ms | step - forward | host-level `Rollout` ms/step |
|---|---|---|---|---|---|---|---|
| fp16 | on | replayed | 726.9 | 727.9 | 724.7 | 2.19 ms (0.30%) | 1021 |
| fp16 | on | computed | 726.9 | 728.1 | 724.7 | 2.14 ms (0.30%) | 1393 |
| fp16 | off | replayed | 727.2 | 728.8 | 725.0 | 2.16 ms (0.30%) | 1047 |
| fp32 | on | replayed | 7442.6 | 7443.5 | 7440.2 | 2.37 ms (0.03%) | 7621 |

- The update (a pass over the 1.07 GB window: read both slots, write both, gather `y`) costs 2.1-2.4 ms per step,
  0.3% of an fp16 step. Wall = GPU time + ~1 ms: the loop keeps the GPU busy.
- Host forcings for 1.69M points: 10.4 ms per step computed (16 OpenMP threads), 4.1 ms replayed (a copy from the
  memory-mapped file into pinned memory), both hidden behind the step.
- The host-level loop pays ~300 ms per fp16 step (pageable H2D of the window and D2H of `y`, synchronous; the
  host update; computed forcings in series): 1021-1393 ms/step, one rollout each, not repeated. The device loop is
  1.4-1.9x faster per step in fp16; in fp32 the forward dominates (2.4%).
- Device memory: backend weights 0.68 GB (fp16) / 1.18 GB (fp32), backend activations 20.1 / 37.1 GB, rollout buffers
  3.56 GB, pinned host 0.97 GB; `cudaMemGetInfo` used 24.8 / 42.3 GB including the CUDA context.
- One-step reference (`docs/cuda.md`): 718 ms fp16, 7.44 s fp32 device-resident, consistent with the
  forward-only graph here (724.7 ms in this job). The original anemoi-inference runner (the reference runs `cuda-*-triton` of
  `docs/rollout.md` §3, warm second run, 24 steps incl. its own forcings and output handling): 42.79 s (1.78 s/step)
  fp16 and 204.18 s (8.51 s/step) fp32, i.e. 2.45x / 1.14x slower per step than this loop; that is the reference
  driver's wall time, not a hook-free benchmark.

Dev model, A100-SXM4-80GB, warm, 24 steps, outputs copied to pinned host memory every step (run `dev-gpu`,
`bench-*.json`). GPU time per step = CUDA events around each step's graph, median over all steps of the
timed rollouts (5 in fp16, 3 in fp32, after one warm-up rollout); forward-only = 24 replays of the
`u = 0` graph (normalise, forward, denormalise: the one-step time on the same buffers).

| precision | graph | forcings | GPU ms/step | wall ms/step | forward-only ms | step - forward | host-level `Rollout` ms/step |
|---|---|---|---|---|---|---|---|
| fp16 | on | replayed | 31.92 | 31.94 | 31.87 | 0.045 ms (0.14%) | 40.75 |
| fp16 | on | computed | 31.92 | 31.98 | 31.81 | 0.110 ms (0.34%) | 36.51 |
| fp16 | off | replayed | 32.12 | 32.14 | 32.07 | 0.054 ms | 36.25 |
| fp32 | on | replayed | 329.08 | 329.25 | 329.05 | 0.035 ms (0.01%) | 334.75 |
| fp32 | on | computed | 329.15 | 329.19 | 329.09 | 0.057 ms (0.02%) | 334.13 |
| fp32 | off | replayed | 329.34 | 329.36 | 329.31 | 0.031 ms | 337.04 |

- The state update costs 0.03-0.11 ms per step (one 13.5 MB pass over the window); the wall time per step is the GPU
  time: the loop never stalls on the host. Host forcings: 0.4-0.5 ms per step, fully overlapped.
- The device loop saves 4-9 ms per fp16 step over the host-level loop (H2D of the window, D2H of `y`, host update,
  synchronous each step): 12-22%.
- For reference, the original anemoi-inference loop on the same model: 78.3 / 344.5 ms/step fp16 / fp32 (another
  job).

## 6. Files

Rollout file (`varda_rollout` = "1"), written by `python/reference/export_rollout.py` (Varda) and
`python/export_rollout.py` (the dev model), read by `apps/varda-rollout`: `x0` [2, G, V_in], `y` [N, G, V_out],
`window` [N, 2, G, V_in] (the window step s sees), `forcings` [N-1, G, F] (forcing s: valid date of output s),
`forcing_index`, `prog_in`, `prog_out`, `kept_in` (I64), `mul_in`, `add_in`, `mul_out`, `add_out`; metadata
`tag`, `steps`, `start_date`, `dates`, `forcing_vars`, `in_vars`, `out_vars`, `kept_vars`, `run` (JSON), and
(`python/export_rollout.py`) `checkpoint_uuid`. The app refuses a file whose variables, grid size or recorded
uuid differ from the model's, and reorders replayed forcings by name into the config's `forcings.dynamic` order.

Runs (one job each; each kept its job script, the exact binaries, the log, the exit code and the JSON reports). GPU
runs on an A100-SXM4-80GB, CPU runs on an EPYC 7713 node:

| run | what |
|---|---|
| `dev-export` | `python/export_rollout.py --verify` of the dev model's three references |
| `dev-smoke` (CPU and GPU) | first 3-4 step runs, `test_rollout`, `test_rollout_cuda` |
| `dev-cpu` | dev model CPU fp32, 24 steps, compare, forcings, replay |
| `dev-gpu` | dev model CUDA fp32 + fp16, 24 steps, all checks; benchmarks |
| `varda-gpu-fp32` | Varda CUDA fp32 vs cuda-fp32-triton, forcings, replay, host; benchmark |
| `varda-gpu-fp16` | Varda CUDA fp16 vs cuda-fp16/fp32-triton, forcings, replay, host; benchmarks |
| `varda-cpu-verify` | Varda CPU fp32, 8 steps, forcings, computed vs replayed |
| `varda-gpu-vscpu` | Varda CUDA fp16 + fp32 vs cpu-fp32-pyg (and the triton references) |
| `varda-cpu-roll` | Varda CPU fp32, 8 steps vs cpu-fp32-pyg (roll semantics) |
| `const-gpu` | constants in place (9505dd0): `test_rollout_cuda`; the dev model and Varda fp16 24 steps with `--verify-host --verify-constants`; byte comparison with the committed binary |
| `const-cpu` | `test_rollout`; the dev model CPU 24 steps `--verify-constants`; Varda CPU fp32 24 steps vs cpu-fp32-pyg |

Every binary was built from the committed CUDA backend (20f47b2) plus this work's files.

## 7. Lessons

- **Double-buffer by parity, capture one graph per parity.** A captured graph has fixed pointers. Instead of
  patching kernel parameters, capture `graph[p]` for the two output/forcing buffers (and a no-update variant
  for the last step): four captures, no extra memory beyond the second buffers, and the upload of the next
  forcing never races the step that reads the current one.
- **Deliver one step behind.** The callback for step s-1 runs after step s is queued: the GPU always has
  work and the host wait is only for a D2H copy that finished long ago.
- **Bitwise equality across implementations is a design property, not luck.** Both loops call the same
  device kernels; the update and every transfer only move values. Checking it per step, on the window as
  well as the output, turned a whole class of indexing mistakes (wrong slot, wrong forcing column, wrong
  parity) into a yes/no test.
