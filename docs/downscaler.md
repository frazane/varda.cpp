# Temporal downscaler: the loop of varda.cpp

How varda.cpp drives the second model of Varda-single-1.0, the **1-hourly temporal downscaler**
(`varda-td`; an anemoi interpolator, config `architecture.model.kind = enc_proc_dec_interpolator`),
over a forecaster run: given the forecaster's state at `t` and at `t + 6 h`, one `predict_step` emits the
six hourly states `t+1 .. t+6`. This page is the engine part (host-level loop, app, tests, what is verified).
The semantics are those of anemoi-inference's `temporal_downscaler` runner
(`anemoi/inference/runners/temporal_downscaler.py`, class `TemporalDownscalerMultiOutRunner`); the reference
runs and their dumps are described in `docs/downscaler-reference.md`; the model internals (T_out > 1
decoder, config, converter) in `docs/format.md`, `docs/cpu.md`, `docs/cuda.md`.

| file | content |
|---|---|
| `src/varda/downscale.h/.cpp` | `DownscaleAssembly` (per-channel source table), `DownscaleTimes`, `ForecasterInfo`, `Downscaler` (the loop) |
| `apps/varda-downscale.cpp` | run on reference inputs or on a forecaster run, compare with reference dumps, tolerances, JSON report, smoke test |
| `tests/test_downscale.cpp` | assembly table, slot fill, rejection of bad lists, dates (ctest, no model file) |

## 1. Semantics

A forecaster run gives `date0` (the analysis date: the date of the newest forecaster input slot), the
forecaster's input window `x0` `(1, T_f, G, V_fin)` in the **forecaster's** input order (Varda: 79 variables:
66 prognostics, 5 dynamic forcings, 8 constants incl. `sdor`, `slor`), and for each forecaster step
`s = 0 .. N-1` the output `y_s` `(G, V_fout = 67)` valid at `date0 + (s+1) * 6 h`.

The downscaler processes the windows `k = 0 .. N-1`, window `k` covering `[date0 + 6k h, date0 + 6(k+1) h]`
(`execute()`: `window_start_date = reference_date + window_idx * window`):

| input slot | state | source |
|---|---|---|
| 0 (window start) | k = 0 | the forecaster's newest input slot `x0[T_f-1]` (the analysis) |
| 0 (window start) | k >= 1 | the forecast `y_{k-1}` |
| 1 (window end) | any k | the forecast `y_k` |

Every downscaler input channel (77 for Varda) is filled **by variable name**, never by position:

- **prognostic** (66: `10u 10v 2d 2t msl q_* sp t_* u_* v_* z_*`; `tp` is output only): the same-named variable
  of the slot's state, looked up in the forecaster's *output* names for `y_s` and in its *input* names for `x0`;
- **constant forcing** (`forcings.constant`: `cos_latitude cos_longitude lsm sin_latitude sin_longitude z`):
  from `x0`'s newest slot by name, identical in both slots (anemoi: `constant_forcings_inputs`, state created once
  and concatenated twice);
- **dynamic forcing** (`forcings.dynamic`: `cos_julian_day cos_local_time insolation sin_julian_day sin_local_time`):
  `compute_forcings` (`docs/forcings.md`, bitwise anemoi's) at the slot's date, window start and window end, on the
  downscaler's own grid (anemoi: `dynamic_forcings_inputs.create_input_state(dates=[start, end])`).

`DownscaleAssembly::from_config` builds the table and **asserts** at construction that every channel is
covered exactly once (a channel in both forcing lists, a listed name at the wrong channel index, an unknown
forcing name, a prognostic that is no forecaster output or not in the forecaster input, a constant that is not
in the forecaster input are errors naming the channel), and `Downscaler` that the downscaler's grid equals the
forecaster's (same `G`, and, when the forecaster's coordinates are given, bitwise equal `grid.latitudes` /
`grid.longitudes`) and that the forecaster's time step equals the window.

One `predict_step` then gives `T_out = 6` states, `y (1, 6, 1, G, 67)`, valid at window start +
`(output_times[t] - input_times[0]) * frequency_hours` = start + `t+1` h (`temporal_downscaler_stepper`:
`step = window * (s+1) / (boundary[-1] - boundary[0])`). The callback gets
`(window k, output index t, step hours from date0 = 6k + t + 1, valid date, y_t)`. `tp` is the raw per-hour
precipitation (the GRIB accumulation is not done here). Boundary states are not re-emitted by the loop (anemoi
skips the state at `t` of windows >= 1 and writes the forecaster's own states at the 6-hourly times); the
downscaler's own prediction of hour `t+6` is the last output of each window.

The forecaster's dynamic forcings of the window start are those of the previous window's end (same date, same
function): the loop reuses them, bitwise equal to recomputing.

## 2. API (`src/varda/downscale.h`)

```cpp
auto fc = varda::ForecasterInfo::from_file("varda/model.safetensors");   // names + grid, no weights
auto td = varda::Model::load("varda-td/model.safetensors", opts);
varda::Downscaler ds(*td, fc);          // throws on any coverage / grid / time-step problem
varda::DownscaleRequest rq;
rq.x0 = x0; rq.forecaster_T = T_f; rq.date0 = date0; rq.n_windows = N;
rq.y_at = [&](int s) { return y[s]; };  // called s = 0, 1, .. in order; pointer valid until the call for s+1 returns
rq.on_output = [&](int k, int t, int step_h, const DateTime& valid, const float* y_t) { ... };
rq.on_input = [&](int k, const float* x) { ... };   // optional: the assembled (2, G, 77)
ds.run(rq);
// the model alone, with inputs assembled elsewhere (e.g. the reference's exact x):
varda::Downscaler only_model(*td);
only_model.run_window(k, date0, x /* (2, G, V_in) */, on_output);
```

`DownscaleAssembly::make` takes explicit name and index lists (tests). `DownscaleTimes` holds the window length
and the output offsets (`start`, `end`, `valid`, `step_hours`). Everything is host-side, any backend
(`Model::predict_step`), threaded with OpenMP; the result does not depend on the thread count (assembly is a
copy, forcings are thread-count independent).

For a streaming caller (`varda-forecast`, §7): `DownscaleRequest::out_buffer(k)` optionally supplies window k's
`(T_out, G, V_out)` output buffer, into which `on_output` then points (the caller owns it again after the window's
last `on_output` returns); `run_window(k, date0, x, on_output, y_out)` does the same for one window. `y_at(s)` may
block (the loop assembles the start slot and the forcings before calling it); the pointer it returns is read only
between that call's return and the call of `y_at(s + 1)`. `WindowTimes` also reports `wait_input` (blocked in
`y_at`) and `wait_output` (blocked in `out_buffer`). The buffers are allocated on first use.

## 3. The app (`varda-downscale`)

```
varda-downscale --model varda-td.safetensors --ref model_io.safetensors            # (a) reference inputs
varda-downscale --model varda-td.safetensors --forecast y.safetensors [--x0 F] [--forecaster-model varda.safetensors] \
                [--out hourly.safetensors] [--dump-x x.safetensors] [--ref model_io.safetensors]   # (b) the loop
varda-downscale --model varda-td.safetensors --forecaster-model varda.safetensors --smoke           # (c) smoke
        [--device cpu|cuda|metal] [--precision fp32|fp16] [--threads N] [--cuda-device I] [--windows N] [--date ISO]
        [--tolerances FILE] [--table NAME] [--json OUT.json]
```

- **(a)** feeds each window's `x.<k>` `(2, G, 77)` of the reference dump directly (the model is judged on the
  reference's exact inputs) and compares `y` against `y.<k>` `(6, G, 67)` per window and per output step with the
  contract metric (`docs/format.md` §2: float64, `rel_absmax = max|a-r| / max|r|` over the whole `(G, V_out)`
  tensor, and `max_abs`; also per variable; NaN or non-finite reference values make an entry fail).
- **(b)** assembles the inputs from a forecaster run: `--forecast` is the file `varda-forecast --dump-y` writes
  (`y.<s>` F32 `[G, V_out]`, s = 1..N, valid at `date0 + 6s h`; metadata `meta` JSON `{out_vars, date}`) plus
  `x0` `[1, T, G, V_in]` (in that file, or `--x0 FILE` with key `x0` or `in.x`) and the forecaster's input names
  (`meta.in_vars` or `--forecaster-model`, which also gives the grid for the bitwise grid check). It writes
  every hourly state (`--out`: `y.<hour>` `[G, V_out]`, hour = step hours from `date0`, metadata `meta` with
  `out_vars`, `date`) and the assembled inputs (`--dump-x`: `x.<k>` `[2, G, V_in]`). With `--ref` it also compares the
  assembled `x.<k>` per slot and per channel against the reference's (the difference is the GRIB packing of the
  forecaster's outputs that anemoi-inference reads back; `rel_absmax` per channel, channels bitwise equal) and `y`
  against `y.<k>` (the effect of that packing on the output).
- **(c)** `--smoke` needs only the downscaler: a synthetic forecaster (deterministic values, names from the
  downscaler's own, or from `--forecaster-model`) runs the loop and checks (1) the assembled inputs against an
  independent by-name implementation inside the app (bitwise), (2) `run_window` on the captured inputs gives
  bitwise the same outputs, (3) all outputs finite, (4) dates and step hours. Exit 1 on failure.
- **Tolerances** (`--tolerances FILE`, `tolerances/downscale/varda-td.json`): `{"<table>": {"y": T | [T_0..T_5] |
  [[per step] per window], "y_var": same}}`; `y` is the overall rel_absmax of a state, `y_var` the worst per-variable
  one (the informative metric: `y` is dominated by the geopotential). Tables `fast-fp32-vs-ref` (engine fp32 vs the
  original's fp32 run on the same inputs), `fast-fp16-vs-ref` (engine fp16 vs the original's fp32 run),
  `fast-fp16-vs-ref-fp16` (engine fp16 vs the original's fp16 run on the same inputs); default by precision, or
  `--table`. Calibrated per window and hour from the original's own spreads (`docs/downscaler-reference.md`) by
  `python/reference/calibrate_downscale_tolerances.py` with the one-step rules (fp32: max(1e-5, 3 x the original's
  cpu-vs-cuda fp32 spread); fp16: 1.75 x the original's fp16-vs-fp32 spread on the same inputs; vs the fp16 run: 2 x
  that spread; rounded up to 2 significant digits). An entry passes if both metrics are within their limits and nothing
  is NaN; no file: reported only. The JSON report holds per window and step every number, per variable.

The window counts, dates and `--date` come from the files (`date`/`date0` metadata) or default to
2026-10-22T15 (only used for the printed valid dates).

## 4. Verification

- `tests/test_downscale.cpp` (ctest): assembly table on tiny name lists whose orders differ (coverage, name
  mapping, constants from `x0`'s newest slot, forcing columns); `fill_slot` against the literal definition for
  window 0 start (x0, input names), window 1 start (`y_0`), window ends (`y_k`), constants in both slots; every
  rejected configuration; dates and step hours across a day, a leap day, a year, and a non-6-h configuration
  (input `[0, 3]`, output `[1, 2]`, 2 h).
- `varda-downscale --smoke` on the synthetic interpolator `g` (`python/tests/synthetic.py`) and on Varda-td with
  the real forecaster's names and grid: see §6.
- Reference comparison: §6 (judged: §7).

## 5. Memory inputs versus anemoi-inference's GRIB inputs

anemoi-inference runs the forecaster, writes GRIB files, and builds each downscaler window by reading them back:
the prognostic inputs are the packed GRIB values (`grid_simple` packing, a fixed number of bits), while
varda.cpp passes the forecaster's float32 outputs straight from memory. The two differ by the packing error of
each field (per-field relative to its range, about 2^-bits). The constants (`z`, `lsm`, ..) and forcings are not
affected (`x0`'s own values, computed forcings). `varda-downscale --forecast .. --ref ..` quantifies this: the
assembled-input difference per slot and channel, and its effect on `y`.

## 6. Results

Reference: anemoi-inference's `temporal_downscaler` runner on the real forecaster run of 2026-10-22T15 (4 windows =
24 h, `$VARDA_SCRATCH/downscale/ref/20261022T15/fp32/model_io.safetensors`: original model, CUDA, fp32, float32 matmuls
(tf32 off), `x.<k>` `(2, G, 77)` and `y.<k>` `(6, G, 67)` per window; the forecaster inputs it was fed are the GRIB
files of the forecaster's fp32 reference run, `docs/downscaler-reference.md`). Metric: `docs/format.md` §2, float64,
over the whole `(G, 67)` tensor of an output step; it is dominated by the geopotential fields (`|z| ~ 1e5`), so the
per-variable `rel_absmax` (each variable against its own maximum) is the informative number and is given as
"worst variable". This section reports the numbers; the verdicts against `tolerances/downscale/varda-td.json` are in §7.
Converted downscaler: `$VARDA_SCRATCH/converted/varda-td/model.safetensors`; forecaster grid/names from
`converted/varda/model.safetensors`. Timings in these jobs are on shared nodes and are not benchmarks.

**(a) the model alone, on the reference's exact inputs** (all 4 windows x 6 steps = 24 entries):

| device, precision, threads | overall rel_absmax (worst of 24) | max_abs (worst) | worst variable rel_absmax (variable) |
|---|---|---|---|
| CPU fp32, 64 | 1.5e-7 | 3.1e-2 (z, 1 ulp) | 4.8e-6 (`tp`; next `10u` 3.8e-6, `v_400` 3.7e-6) |
| CUDA fp32 (A100) | 1.5e-7 | 3.1e-2 | 4.3e-6 (`tp`; `v_1000` 3.3e-6, `10v` 3.1e-6) |
| CUDA fp16 (A100) | 7.4e-5 | 15.2 (z) | 7.7e-3 (`tp`; `v_50` 4.8e-3, `q_200` 4.6e-3, `q_150` 4.2e-3) |

Against the original model's CPU fp32 run (`ref/20261022T15/cpu-fp32`, the ground truth; (a), 24 entries): CPU fp32
engine, 64 threads: overall 1.5e-7, worst variable 3.4e-6 (`v_300`; `10u` 3.4e-6, `v_400` 3.4e-6, `tp` 3.3e-6); CUDA fp32:
overall 1.5e-7, worst variable 6.1e-6 (`tp`; `v_1000` 4.4e-6, `10u` 4.0e-6); CUDA fp16: 7.4e-5, worst variable 7.7e-3 (`tp`).

Against the original model's own fp16 run (`ref/20261022T15/fp16`, CUDA fp16 autocast, `--precision checkpoint`, fed
by the fp16 forecaster run's GRIB files), same 24 entries:

| device, precision | overall rel_absmax (worst of 24) | max_abs (worst) | worst variable rel_absmax (variable) |
|---|---|---|---|
| CUDA fp16 (A100), (a) reference inputs | 7.7e-5 | 15.7 (z) | 4.6e-3 (`q_100`; `v_250` 3.7e-3, `v_300` 3.3e-3, `tp` 3.3e-3) |
| CUDA fp32, (a) reference inputs | 9.8e-5 | 20.1 | 8.2e-3 (`tp`; `v_50` 5.0e-3) |

(the fp32 engine is 8.2e-3 from the fp16 reference and the fp16 engine 4.6e-3: the reference's own fp16 error is
the larger part of that distance, as for the forecaster, `docs/results.md`).

By output step the worst per-variable `rel_absmax` is flat in fp32 (CPU 4.1e-6 .. 4.8e-6, CUDA 2.4e-6 .. 4.3e-6 over
steps 1..6) and in fp16 (vs the fp32 reference) 4.0e-3 .. 7.7e-3 (no growth with the hour); by window fp32 2.4e-6 ..
4.8e-6, fp16 4.0e-3 .. 7.7e-3 (vs the fp16 reference: by step 2.6e-3 .. 4.6e-3, by window 2.6e-3 .. 4.6e-3). In the
first two tables fp16 is compared with the fp32 reference.

**(b) the full loop** (inputs assembled in memory from the forecaster's fp32 rollout `ref-fp32/rollout.safetensors`
of the real-data run, outputs of its steps 1..4; the reference reads the same states back from GRIB):

| | CPU fp32 | CUDA fp32 | CUDA fp16 |
|---|---|---|---|
| assembled `x` vs reference `x.<k>`: worst channel rel_absmax (8 slots) | 1.1e-7 | 1.1e-7 | 1.1e-7 |
| channels bitwise equal per slot (of 77) | 36 .. 40 | 36 .. 40 | 36 .. 40 |
| `y` vs `y.<k>`: overall rel_absmax (worst of 24) | 1.5e-7 | 1.5e-7 | 7.7e-5 |
| `y`: worst variable rel_absmax | 4.9e-6 (`tp`) | 3.8e-6 (`tp`) | 7.2e-3 (`tp`) |

CUDA fp16 against the fp16 reference with the fp16 forecaster run's states (`ref-fp16`): `x` worst channel 1.2e-7, `y`
overall 9.9e-5, worst variable 5.2e-3 (`q_500`; `tp` 5.1e-3), against 4.6e-3 for the same model on the reference's exact
inputs ((a)): input differences of 1e-7 are amplified by fp16 rounding to the size of the fp16 noise itself.

The dynamic forcings (`cos/sin_julian_day`, `cos/sin_local_time`, `insolation`) and the constants `lsm` and
`cos/sin_latitude/longitude` are bitwise equal to the reference's in every window and slot; the constant `z`
differs by 1.95e-3 (3.4e-8 relative, identical in all 8 slots: the source of the constants, the forecaster's `x0`
as exported, versus the reference's constant-forcing input); the prognostic channels differ by at most 1.1e-7
relative to the channel's maximum (a few float32 ulps: the round trip of the forecaster states through the GRIB
files and the reference's dtype conversions), 36 to 40 of the 77 channels per slot being bitwise equal. The
effect on `y` is not above the (a) numbers (CPU fp32 4.9e-6 against 4.8e-6; CUDA fp32 3.8e-6 against 4.3e-6):
memory versus GRIB inputs is not measurable at the output precision.

Self-checks (`--smoke`, deterministic synthetic forecaster; assembly bitwise equal to the app's independent by-name
implementation, `run_window` bitwise equal to the loop, outputs finite, dates): synthetic interpolator `g` on CPU
fp32, CUDA fp32 and CUDA fp16, Varda-td with the real forecaster's names and grid on CPU fp32 (2 windows, 64
threads, 93.5 s per window): all PASS.

## 7. The real-data path: `varda-forecast` with the downscaler

`varda-forecast --downscaler TD.safetensors --downscaler-io-config varda-td-io.json` (or an assets bundle that holds
`downscaler.safetensors` + `varda-td-io.json`; `--no-downscaler` skips it) runs the HF notebook's two commands in one
process (`docs/forecast.md`): after the 6-hourly rollout, the downscaler weights (loaded after the forecaster's set-up,
overlapping the input preparation) run the loop of §1 on the rollout's outputs kept in memory, and a second GRIB writer
thread encodes the hourly LAM files `outputs/varda-single_<dateTime>_<h>.grib`, h = 0 .. lead time, as anemoi-inference's
`temporal_downscaler` runner writes them (`docs/downscaler-reference.md`; io config `docs/io-output.md`):

- hour 0: the 66 prognostics of the analysis (the forecaster's newest input slot), in the downscaler's input order;
- hours 6k+1 .. 6k+5: the downscaler's outputs; hours 6k (k >= 1): every field but `tp` is the forecaster's state of
  that step (anemoi's `copy-prognostic-from-forecaster`), taken from memory, not re-read from the 6-hourly files;
- `tp`: the downscaler's hourly precipitation, clamped and accumulated from the start of the forecast across windows
  (`accumulate_from_start_of_forecast`), x1000 (`rescale`), `startStep 0`, `endStep h`; never the forecaster's;
- message order: the initial-state fields first, `tp` last (`step_field_order`), 24-bit CCSDS, the ICON templates.

`--dump-y` also writes `hourly.safetensors` (`y.<hour>` raw outputs) next to `y.safetensors` (which now holds `x0`
and the input names, so `varda-downscale --forecast y.safetensors` can replay the loop).

**Results** (2026-10-22T15, 24 h, A100-SXM4-80GB, 32 threads, graph on; `$VARDA_SCRATCH/jobs/td-int-forecast`):

| check | fp16 | fp32 |
|---|---|---|
| hourly files vs anemoi-inference's (`varda-grib-compare`, per message, 25 files, 1674 messages) | worst `rel_absmax` 1.5e-2 (to the field max; the forecasters' own fp16 difference is 5.6e-3 and the original's whole-chain fp16-vs-fp32 difference 1.04e-2), 62 messages byte-identical (hour 0, as the original's own re-encode of its step 0) | worst 1.4e-5, 62 byte-identical; the compare's "FAIL" lines are the value-derived keys only (the statistics `average`, `kurtosis`, .. and the data-section `md5Section5/7`), which follow the values |
| two runs | byte-identical, 35 files / 2350 messages | same |
| wall clock, process (the binary before the overlap below; forecaster + downscaler + 35 GRIB files) | 23.3-24.3 s (forecaster part 7.9-9.1 s; downscaler 15.1 s: loop 10.8 s = 4 x 0.94-1.38 s predict incl. the 2.7 GB output copy + 0.1 s assembly per window + 3 s copying into the writer queue, + 2.3 s writing after the last window; hourly GRIB encode 6.2 s and write 5.0 s on the writer thread) | 73-94 s (loop 33 s: 6.8 s per window) |

Same job, same GPU, the original (anemoi-inference, the notebook's two commands; `$VARDA_SCRATCH/jobs/td-bench`):
fp16 autocast, forecaster 215-247 s + downscaler 192-194 s process wall clock (`execute` 162-173 s + 128-137 s),
against **23.7 s** for varda.cpp (two runs each): about 17x; fp32, 250 s + 217 s against 75.3 s: 6.2x. The node was
shared with another job of ours during this measurement; the forecaster-only comparison of `docs/results.md` section 5
(136 s vs 8.2 s) was taken on an idle node.

**Judged loop** (`varda-downscale --tolerances tolerances/downscale/varda-td.json`, the model on the reference's exact
inputs, 4 windows x 6 hours, both metrics `y` and `y_var`; `$VARDA_SCRATCH/jobs/td-judge`):

| engine | reference, table | worst x tolerance | verdict |
|---|---|---|---|
| CPU fp32, 64 threads | original cpu fp32, `fast-fp32-vs-ref` | 0.34 | PASS 24/24 |
| CUDA fp32 | original cuda fp32 / cpu fp32, `fast-fp32-vs-ref` | 0.33 / 0.47 | PASS 24/24 |
| CUDA fp16 | original fp16 on the same inputs, `fast-fp16-vs-ref-fp16` | 0.85 | PASS 24/24 |
| CUDA fp16 | original cuda fp32, `fast-fp16-vs-ref` | 1.02 (window 2, hour 1, metric `y`: 4.41e-5 vs 4.3e-5) | 23/24 within; one entry 2% above |

The fp16-vs-fp32 table is 1.75 x one fp16 trajectory of the original, which `docs/rollout.md` §4 showed is not a noise
floor (equally valid fp16 runs do not track each other closely on single entries); against the original's own fp16 run
on the same inputs every entry is within 0.85. Reported, not loosened. The full loop fed by varda.cpp's own forecaster run
(`--forecast`) is not a same-inputs comparison (the two forecasters differ by their model error: fp32 1.1e-5, fp16
8e-3 per `docs/results.md` section 5), so its distance from the reference (fp32 worst variable 4e-5, 3.3 x the
same-inputs tolerance; fp16 1.4e-4 overall) is reported only.

**Overlap with the rollout and hand-off to the writer** (since the measurements above). With the device rollout (CUDA,
not `--host-loop`) the downscaler no longer waits for the whole rollout:

- *Window k as soon as step k is out.* The loop runs on its own thread, started when the rollout starts; `y_at(k)` waits
  until the rollout callback (one step behind, `docs/rollout-engine.md`) has published step k. `Downscaler::run`
  computes the window's forcings and fills its start slot before it asks for `y_k`, so only the end slot waits. Both models
  are loaded, both validated and the rollout's graphs captured before that thread starts, so no cuBLASLt set-up or
  capture of one backend overlaps a capture of the other (this thread only replays graphs and copies during the rollout;
  the downscaler's first window runs eagerly and its second captures its graph, on its own thread). The downscaler's
  backend uses a stream of the device's highest priority (`CudaBackend::use_high_priority_stream`), so its windows go
  ahead of the rollout's steps on the GPU: the hourly writer is the critical path, the 6-hourly files have time to spare.
- *No copy into the writer queue.* `DownscaleRequest::out_buffer` lets the caller give each window's output buffer
  `(6, G, 67)`; `varda-forecast` hands the six states to the hourly writer by pointer (`StepWriter::push_ref`) and gets
  the buffer back once the writer no longer reads any of the six states (2 buffers by default, `VARDA_TD_BUFFERS`); the
  loop waits only when both are in flight. The 6-hourly writer also takes the forecaster's states by pointer when they are kept for the
  downscaler.
- *The hourly writer starts with the rollout* (the analysis, hour 0, first), and the 6-hourly writer encodes with 8
  OpenMP threads while the downscaler runs (`VARDA_FC_WRITER_THREADS`), leaving the cores to the hourly one.
- `VARDA_TD_OVERLAP=0` runs the downscaler after the rollout (as the other engines do), still without the copies.
  `VARDA_TD_PINNED=1` page-locks the window buffers. Nothing changes a byte of the files.

Measured in one job (A100-SXM4 96 GB "PG506-232", 32 threads, CUDA graph on, 2026-10-22T15, 24 h, `--assets`, process
wall clock, warm file caches; `$VARDA_SCRATCH/jobs/td-perf-5`), previous binary (`td-int-forecast/bin`) against this one,
two runs each in alternation (seconds):

| | fp16 before | fp16 after | fp32 before | fp32 after |
|---|---|---|---|---|
| total | 23.52 / 22.45 | **19.15 / 19.25** | 71.11 | **64.32** |
| ready (input and both loads, overlapped) | 5.19 / 4.05 | 3.91 / 4.15 | 9.56 | 8.63 |
| rollout, 4 steps (GPU shared with the downscaler after) | 2.98 / 2.98 | 4.48 / 4.42 | 26.17 | 45.85 |
| rollout `step_ms` | 649-655 | 647-650, 1311-1313 (shared) | 6472-6475 | 6474, then 12985-12987 (shared) |
| downscaler starts (from process start) | after the rollout + 6-hourly tail | 4.07 / 4.30 | after | 8.78 |
| downscaler weights load | 1.45 / 1.21 | 1.16 / 1.32 | 0.50 | 0.50 |
| loop (4 windows) | 11.06 / 11.00 | 10.92 / 10.77 | 31.22 | 52.81 (23.2 s waiting for the rollout) |
| per-window `predict_step` (ms) | 1107-1663 | 1287-1460, 2585-2714 (2nd window: graph capture on a shared GPU), 1401-1657 | 6820-6957 | 6896-7120 |
| loop: copying states into the writer queue | 3.37 / 4.06 | 0 | 3.05 | 0 |
| loop: blocked on the writer (queue full / both buffers in flight) | 0.81 / 1.19 | 2.17 / 2.37 | 0 | 0 |
| writing after the last window | 2.29 / 2.31 | 4.16 / 4.18 | 2.09 | 2.73 |
| hourly GRIB encode / write (writer thread, summed) | 6.44 / 4.64, 6.37 / 4.65 | 6.53 / 5.04, 6.55 / 5.12 | 6.08 / 4.70 | 5.93 / 4.49 |
| forecaster only (`--no-downscaler`, 10 files) | 7.31 / 7.32 | 7.37 / 7.42 | | |

Files: every run byte-identical to the reference run of the table above (`varda-grib-compare --require-bytes`, 35 files;
10 for `--no-downscaler`). The fp16 total is now set by the hourly GRIB writer: it starts on window 0's output about
1.5 s after the rollout starts and then needs ~0.5 s per hour (24 x 0.44-0.55 s encode + write, slower while the
6-hourly writer runs), so the GPU waits for it (2.2-2.4 s blocked). fp32 is GPU-bound (two models sharing one GPU:
4 x 6.5 s + 4 x 6.9 s); the gain is the writers' work hidden under the GPU and the removed copies.

Also measured, not kept or not default (same node type, `$VARDA_SCRATCH/jobs/td-perf-{1,2,4}`, fp16): page-locked window
buffers (`VARDA_TD_PINNED=1`) make `predict_step` 0.3-0.5 s faster per window but cost 2.7-3.0 s of `cudaHostAlloc`
for 2 x 2.7 GB on the downscaler's thread, during which the other threads stall (total 21.9 s against 19.4-19.7 s); the
default-priority stream (`VARDA_TD_PRIORITY=0`) 20.0 s (window 0 finishes 0.7 s later); the downscaler after the
rollout without the copies (`VARDA_TD_OVERLAP=0`) 21.7 s; the 6-hourly writer's full OpenMP team instead of 8
threads 19.5-19.7 s against 19.2-19.3 s. An out-of-tree experiment in which `GribOutput::write_state` does not wait for
a state's file writes before returning (they then overlap the next state's encoding; `close()` waits) gave 17.8 s against
19.4 s in the same job, byte-identical files (`$VARDA_SCRATCH/jobs/td-perf-exp`): the hourly writes slow down when they
overlap the encoding (5.3 s -> 10.3 s summed), so the per-hour cost goes from ~0.55 to ~0.45 s, not to the encode time
alone. That change is now in `GribOutput` (`write_state` leaves the last output's writes pending; `close()`, `stats()`
and `files()` wait). **Final binary** (`$VARDA_SCRATCH/jobs/td-final`, A100 96 GB, same settings): fp16 **18.75 s for the
first run in the allocation and 17.29 s warm** (process wall 23.2 / 17.9 s; hourly encode 6.9-7.1 s, writes 9.9-10.0 s
summed on the writer thread, 3.2 s after the last window), fp32 63.9 s; every run byte-identical to the reference runs
(35 files); `--no-downscaler` 7.40 s (10 files, unchanged).

**Concurrent hourly encoding** (since the final binary above). The hourly writer encodes several hours at once
(`GribOutput::set_concurrency`, `docs/io-output.md` §2): the writer thread runs the top-level post-processors in hour
order (the `tp` accumulation, the only state carried from hour to hour) and queues the state; 3 worker threads
(`VARDA_TD_WRITER_STATES`), each with an OpenMP team of threads / 3 (`VARDA_TD_WRITER_THREADS`), transpose, encode and
write whole hours, each hour's file created in hour order and written by its own worker. A worker releases the window
buffer's state once it has transposed it; the buffer returns to the pool when all six states of the window are released
(they may be released in any order). The files are byte-identical. The 6-hourly writer is unchanged (one state at a time).

Measured in one job (A100-SXM4 96 GB "PG506-232", 32 CPUs, CUDA graph on, 2026-10-22T15, 24 h, `--assets`, warm file
caches; `$VARDA_SCRATCH/jobs/perf2-final2`), the final binary above (`td-final/bin`) against this one, three runs each in
alternation after one discarded first run (seconds; `forecast.json` unless "process"):

| | fp16 before | fp16 after |
|---|---|---|
| process wall clock | 18.00 / 18.03 / 18.14 | **17.47 / 17.20 / 17.12** |
| total | 17.37 / 17.35 / 17.44 | **16.68 / 16.50 / 16.43** |
| ready | 3.91-3.96 | 3.98-4.21 |
| rollout (4 steps, GPU shared with the downscaler) | 4.56-4.58 | 4.30-4.69 |
| downscaler loop (4 windows) | 10.03-10.10 | 10.34-10.43 |
| loop blocked on the writer | 0.47-0.95 | **0** |
| writing after the last window | 3.21-3.25 | **1.92-1.94** |
| per-window `predict_step` (ms) | 1151-1223, 2950-3375, 1675-1784, 1516-1534 | 1183-1398, 3295-3482, 2033-2207, 1987-2033 |
| hourly GRIB encode / write (summed over hours) | 7.2-7.4 / 10.1-10.4 | 14.2-15.8 / 6.7-7.5 (3 hours at once) |

Every run byte-identical to the reference runs (`cmp` of all 35 files; `varda-grib-compare --require-bytes` on one fp16
and one fp32 run); `--no-downscaler` byte-identical, 7.35-7.80 s (`$VARDA_SCRATCH/jobs/perf2-final{,2}`, unchanged path);
fp32 63.71 s against 63.87 s (`perf2-final`; GPU-bound, the two models need the 96 GB GPU in fp32); `--no-graph` fp16
16.35-16.90 s. The writer no longer holds the loop back (0 s blocked, 1.9 s after the last window: the last window's six
hours, two rounds of three), but the gain is only ~0.9 s because the loop itself became slower: windows 2 and 3 take
~2.0 s instead of ~1.6 s. The hourly encoding costs ~140 CPU seconds per 24 h (gather 23, keys 37-47, CCSDS packing
70-76, check 3), so with 32 CPUs it occupies the machine while the loop runs, and the window's pageable copies (1 GB
host-to-device, 2.7 GB device-to-host, staged by the driver on the calling thread) slow down. Two states (instead of 3)
16.9 s, four 16.6 s, one (sequential) 17.6 s; 8 or 6 threads per state 16.9-17.0 s (same job type, `perf2-s{1,2}`).

What the second window costs, measured (`perf2-s2`, `perf2-s3`): the second window takes ~3.3 s against ~1.2 s for the
first, also with `--no-graph`, so it is not the graph capture. A warm-up that ran an eager forward on zeros and captured
the downscaler's graph before its first window (on the downscaler's thread or before the rollout; result discarded)
measured the capture, instantiation and upload at 1-2 ms and made the run slower (17.1-17.2 s against 16.5 s: one
extra forward on the shared GPU), so it is not kept. With page-locked window buffers the spike disappears (windows
0.9-1.7 s): it is the device-to-host copy into pageable memory while both writers load every CPU (the hourly writer
encoding window 0, the 6-hourly one steps 1-2). Page-locking the 2 x 2.7 GB costs 3.0-3.9 s of kernel time that stalls the
other threads wherever it is done: on the downscaler's thread before window 0 (`VARDA_TD_PINNED=1`) 18.3-20.3 s; in the
background from the start, or after the rollout set-up, 18.0-19.1 s (ready 6.6-7.2 s instead of 4.0 s), and the process
exit then spends 1.7-4 s unpinning; none is kept.

**The downscaler on a second GPU** (`varda-forecast --downscaler-device cuda:1`, `./varda forecast --gpus 2`). The
downscaler's backend gets its own CUDA device; everything else is unchanged (the rollout's steps reach it through host
memory, the start-up order stays as above, the files are byte-identical). Every CUDA call of the downscaler runs with
its device current: the backend's entry points set it, `varda-forecast` sets it around the downscaler's load and loop
(window buffers included) and restores the forecaster's on the main thread, and a CUDA engine's destructor sets its
own device. Measured in one job, 2 GPUs and 32 CPUs, one GPU and two GPUs in alternation after a discarded first run
(A100-SXM4 96 GB "PG506-232", 2026-10-22T15 and, for 120 h, 2026-10-30T15; `$VARDA_SCRATCH/jobs/gpu2/a`; seconds, process
wall clock / `forecast.json` total):

| | one GPU | two GPUs |
|---|---|---|
| fp16, 24 h | 16.89 / 16.18, 17.12 / 16.40 | 17.13 / 16.33, 17.22 / 16.41 |
| fp16, 120 h | 59.57 / 58.85 | 59.52 / 58.70 |
| fp32, 24 h | 68.36 / 67.44 | **48.25 / 47.26** |
| fp32, 24 h, A100-SXM4-80GB (`gpu2/b`) | cudaMalloc out of memory | **54.44 / 53.61** |

In fp16 a second GPU changes nothing: the rollout is no longer slowed (`step_ms` 647-659 instead of 648 / 1310
alternating, rollout 2.9-3.2 s instead of 4.5-4.9 s), but the downscaler loop, which sets the total, stays at 10.0 s
for 24 h (51.8 s for 120 h): its windows take 1.7-2.7 s for 0.7 s of GPU work, the rest being the pageable host copies
(1 GB in, 2.7 GB out) while the hourly writer loads every CPU (above); window 0 is even 0.3-0.6 s slower on the second
GPU (1.5-1.8 s against 1.1-1.3 s; not lazy kernel loading: an eager warm-up forward on that GPU before window 0 left it
at 1.8 s and changed the total by -0.2 s at 24 h and +1.5 s at 120 h, `gpu2/c`, not kept). In fp32 the two models no longer
share one GPU (rollout 26.3 s instead of 45.8 s, 6.0 s instead of 23.3 s of the loop waiting for the forecaster): 30%
faster, and both models fit on 80 GB GPUs. Every run byte-identical: 24 h runs to the reference runs above (35 files),
120 h two GPUs to one GPU (163 files), also through `./varda forecast --gpus 2` (`gpu2/v`, 80 GB node).

## 8. Open points

- With the concurrent hourly writer, the fp16 critical path is the downscaler loop on a CPU-saturated node: ~4 s ready,
  window 0 at ~0.7 s after the rollout starts, 4 windows of 1.2-3.5 s (forward 0.73 s; the rest is the pageable copies of
  1 GB in and 2.7 GB out per window, slowed by the encoders' CPU load), then 1.9 s of hourly writing. Ways forward, not
  done: (a) a device-resident loop (inputs assembled on the GPU from the rollout's device outputs, as `DeviceRollout`)
  and outputs copied through a small page-locked staging ring with a parallel host copy, which removes both the 1 GB
  upload and the driver's single-threaded staging without locking 5.4 GB; (b) cheaper hourly encoding: the keys part
  (~40 CPU s: template clone and `set_key_vals` per message) and the column gathers (~23 CPU s) are repeated identically
  every hour; (c) more CPUs per GPU (the job uses 32 of the node's 128).
- The overlap is used with the device rollout only; `--host-loop`, CPU and Metal run the downscaler after the rollout.
- Only downscalers with two input states (window start and end) are supported.
- Two dates are verified at 24 h. On 2026-10-30T15 every judged table holds without change, 24/24 each:
  CPU fp32 0.42, CUDA fp32 0.43 / 0.78 (vs the original's cuda / cpu fp32), CUDA fp16 0.89 vs the original's fp32 and
  0.85 vs its fp16 on the same inputs, and 0.68-0.81 for the loop assembled in memory from the original's rollouts.
  The hourly files match within 2.2e-5 (fp32) and 2.35e-2 (fp16). The fp16 figure is set by the forecaster's copied
  fields at hour 24; the original's own fp16 chain differs from its fp32 chain by 3.13e-2 there. Hour 0 is 66/66
  messages byte-identical (`docs/results.md` §6, `docs/downscaler-reference.md` §6.4). The first date's one entry above
  tolerance (1.02, `fast-fp16-vs-ref`, §7) is window 2 (0-based), hour 1, metric `y` (overall rel_absmax 4.41e-5
  against 4.3e-5).
- Two dates are verified at 24 h; at 120 h (20 windows) only self-consistency is checked (`docs/results.md` §6: the
  first 24 h byte-identical to a 24 h run, the copy rule at every 6 h, `tp` monotone), as no reference exists there.
- The tolerances are still calibrated on one date (2026-10-22T15). The original's own spreads differ by up to 1.9x
  between the two dates. The same rules applied to the 10-30 spreads give larger limits on 13 of 24 fp16 `y_var`
  entries (up to 1.76x) and on 7 of 24 fp32 ones (up to 1.90x), and smaller ones elsewhere. The 10-22 runs would
  exceed such a table (1.31) while the 10-30 runs pass the current one. Calibrating on the element-wise maximum of several
  dates is not done (tolerances are not loosened without the integrator).
- Metal: the backend is not yet generalised to T_out > 1 (`docs/metal.md`).
