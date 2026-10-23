# Forecast from real data: `varda-forecast`

`varda-forecast` runs Varda as anemoi-inference runs the Hugging Face config `inference_opendata.yaml`: KENDA-CH1
analyses and ECMWF open data in (GRIB), the autoregressive rollout, the `lam` (ICON templates) and `global` (IFS
templates) GRIB files out, initial state included. The pieces are documented where they live:

- what is reproduced and how, with the open questions: `docs/spec-io.md`;
- the reference run and its dumps: `docs/realdata-reference.md`;
- input side (GRIB -> x0): `docs/io-input.md`; output side (y -> GRIB): `docs/io-output.md`;
- the rollout: `docs/rollout-engine.md`.

This file is the end-to-end view: how to run it, how the work overlaps, where the time goes and what is not covered.
Numbers: `docs/results.md` section 5. The user entrypoint around it (`./varda setup`, `./varda forecast latest`: build,
assets bundle, input download, srun) is `docs/quickstart.md`.

## 1. Inputs

The first five rows are derived once per model release and live together in the **assets bundle**
`$VARDA_SCRATCH/assets/varda-single-1.0/` (`--assets DIR`; built by `tools/make_assets.py`, `docs/quickstart.md`
section 2): `model.safetensors`, `varda-io.json`, `regrid-ifs0p25-n320.safetensors`, `templates/`, `definitions/`,
`manifest.json`. The bundle's files are hard links of (sha256-equal to) the paths in the table.

| input | what | where (verified date) |
|---|---|---|
| model | converted file with the cutout masks (`docs/format.md` section 1.1; the converter adds `grid.<source>.cutout_mask`) | `$VARDA_SCRATCH/converted/varda/model.safetensors` |
| io config | `varda-io.json` (schema `varda-io/1`): variables, orders, namer, output paths, lead time, patches. Exported from the HF config by `python/export_io_config.py` | `$VARDA_SCRATCH/realdata/assets/varda-io.json` |
| regrid matrix | MIR's IFS 0.25 deg -> N320 `linear` interpolation as a sparse matrix (542 080 x 1 038 240, nnz 1 505 824), exported by `python/export_regrid_matrix.py` | `$VARDA_SCRATCH/realdata/assets/regrid-ifs0p25-n320.safetensors` |
| templates | `templates/` of the HF repo (`ICON` and `IFS` samples; the N320 `ifs-levtype=sfc.grib` also serves as encoding template for the regrid) | `$VARDA_SCRATCH/hf/templates` |
| COSMO definitions | ecCodes definitions that give KENDA-CH1 messages their COSMO short names; `--definitions` or `VARDA_COSMO_DEFINITIONS` (set by `tools/env.sh`) | `$VARDA_SCRATCH/deps/eccodes-cosmo-resources-2.44.0.1/definitions` |
| KENDA-CH1 | `--kenda DIR` with `DIR/raw/<YYYYmmddTHHMMSS>/<VAR>.grib2` for t-6h and t (11 variables: P PMSL PS QV T T_2M TD_2M U U_10M V V_10M) and `DIR/constants/{horizontal,vertical}_constants_icon-ch1-eps.grib2` | `.../20261022T15/cache/kenda` |
| IFS open data | `--ifs-prev F` (file valid at t-6h), `--ifs F` (valid at t), `--ifs-constants F`: full IFS open-data files (`<run>-<step>h-oper-fc.grib2`). Each option can be repeated if the fields are split over files | `.../20261022T15/cache/ecmwf/files` |

The ecCodes build is the one the reference uses (`ECCODES_HOME` in `tools/env.sh`); the build option is
`-DVARDA_GRIB=ON` (default; it needs ecCodes, and the engine core stays free of it).

**The constants run is explicit.** The reference takes the "latest" open-data run at start time for the global
constant forcings (`constant: true`); for 2026-10-22T15 that was the 18z+0h file, chosen at download time and recorded
in `pins.json`. A re-run later would pick another run, so the C++ never decides: `--ifs-constants` names the file.
The same holds for the IFS runs behind t-6h and t (06z+3h and 12z+3h here).

**`--realdata DIR`** reads an input directory: the frozen reference cache `$VARDA_SCRATCH/realdata/20261022T15`, or
one written by `python/fetch_inputs.py` (`$VARDA_SCRATCH/inputs/<YYYYmmddTHH>`, same layout, with only the IFS
messages used: `docs/quickstart.md` section 2). It reads
`DIR/pins.json` (date, `input_dates`, `ecmwf_files`, `constant_forcings_run`), takes the KENDA cache from
`DIR/cache/kenda` and the three IFS files from `DIR/cache/ecmwf/files`. `--date` given explicitly replaces the pinned date; the file lists are always taken from the pins. Any other date uses
the explicit options.

## 2. Run

```
source tools/env.sh
cmake -S . -B build-out && cmake --build build-out -j 32
build-out/varda-forecast --assets $VARDA_SCRATCH/assets/varda-single-1.0 \
    --realdata $VARDA_SCRATCH/realdata/20261022T15 --out OUT \
    --device cuda --precision fp16 [--lead-time-hours H] [--write-queue N] [--allow-nans] [--dump-y] [--quiet]
# any date:  --date 2026-10-22T15:00:00 --kenda DIR --ifs-prev F --ifs F --ifs-constants F
# without the bundle: --model M --io-config J --regrid-matrix R --templates DIR [--definitions D] (table above)
```

`./varda forecast DATE` (`docs/quickstart.md`) runs exactly this, with `--assets`, `--realdata` of the date's input
directory and `--quiet` (the input log goes to `forecast.json` only). `--assets` gives the same bytes as the
explicit options (10 / 10 files byte-identical, `docs/quickstart.md` section 6).

It needs a GPU node (or `--device cpu`, ~60 s per step in fp32, `docs/results.md` section 3). `--lead-time-hours`
defaults to the io config's (24 h, 4 steps of 6 h) and must be a multiple of the model's step. `--no-graph` and
`--host-loop` select the rollout variants of `docs/rollout-engine.md`. `--dump-y` also writes `OUT/y.safetensors`
(`y.<step>`, F32 `[G, V_out]`, before post-processing) for comparisons with the reference dumps.

**Outputs** under `OUT`, named as anemoi-inference names them (unpadded step in hours):

```
OUT/outputs/202610221500_{0,6,12,18,24}.grib          lam, ICON templates (70 messages at step 0, 67 per step)
OUT/outputs/global-202610221500_{0,6,12,18,24}.grib   global, IFS templates (same counts; NaN bitmap at the 1 410 masked N320 points)
OUT/forecast.json                                     provenance and timings
```

The 4-step forecast is 10 files, 676 messages, 1 276 890 049 bytes. `forecast.json` holds: model, assets, io config, regrid matrix, templates, definitions, realdata dir, checkpoint uuid, date,
device, precision, graph, engine (`device`/`host`), threads, steps, the files written, the input log (every file and
message used), the write queue length, `step_ms` (per step on the GPU) and `seconds`: `load`, `input` (and
`input_stages`: `model_grid`, `kenda`, `ifs`, `kenda_read_wait`, `ifs_read_wait`, `kenda_ifs`, `merge`, `x0`; `kenda` and `ifs` run
concurrently, so they overlap and do not sum to `total`; `docs/io-input.md` section 7), `rollout_setup`, `ready`, `initial_state_write`, `rollout`,
`rollout_blocked_on_writer`, `rollout_callback_copy`, `write_after_rollout`, `step_write` (per step), `grib`
(`post`, `encode`, `write`, `messages`, `bytes`) and `total`.

## 3. How the work overlaps

```
main thread     : load weights -> set up rollout (eager pass + graph) ---------- | join input | rollout, callback copies y | finish
input thread    : InputModel::load -> build_input (KENDA || IFS+regrid, merge, x0) |
writer thread   :                                                                 | initial state | step 1 | step 2 | ... | step n
```

- **Input and model load in parallel.** The input thread needs only the converted file's config, grid and cutout
  masks (`InputModel`), not the weights. The main thread loads the weights onto the device and runs the rollout set-up.
  The rollout starts when both are done ("ready").
- **Writer thread, bounded queue.** The rollout callback only copies `y` (G x V_out floats, 452 MB) into a queue of
  `--write-queue` steps (default min(steps, 8)) and returns; the writer thread encodes and writes the initial state
  (as soon as the input exists) and then each step in order. If the queue is full the callback blocks, and that wait is
  measured separately (`rollout_blocked_on_writer`) from the copy (`rollout_callback_copy`). The first error in the
  writer is rethrown at `finish()`.
- **Encoding in parallel, order kept.** Inside a step the fields are encoded in parallel with OpenMP (one ecCodes handle per message), and the messages are then written in field order, so the files are byte-identical to the serial writer
  and to anemoi-inference's. `y` is transposed once per step (`[G, V]` -> per-variable contiguous) and the lam and
  global file writes overlap.
- **State is not shared** between the rollout and the writer except through the queue; `tp` is accumulated and rescaled
  on the writer thread in step order.

## 4. Timings

Start-up (model load, input) was sped up after the tables below were measured (commits 5538fa1, eb7f17b). Current
expectation (A100-SXM4-80GB, 32 CPUs, fp16, graph on, 4 steps, process wall clock):
- **About 13 s for the first run in a fresh allocation** (12.9 and 13.6 s in two jobs, `docs/results.md` section 5). That
  first process loses 4.6-5.4 s before its own timer starts; shared libraries (CUDA, cuBLAS, ecCodes) not yet in the
  page cache are the likely cause (not proven).
- **8.3-8.7 s once those libraries are cached**, with the file caches of the model and the inputs dropped (three runs,
  2026-10-23); 7.8-7.9 s warm. The binary of the tables below, alternated with it in the same job: 20.8-26.3 s cold,
  10.5-10.6 s warm. Cold weights load 1.9-2.7 s (was 15-17 s). Table and method: `docs/results.md` section 5, last
  subsection.

The tables below, the "10.4 s warm" and the 14 s cold load are historical (commit a9dcc66).

Environment: `VARDA_PREFETCH_THREADS` (weights prefetch threads, default 32, 0 disables; the cold mapping on Lustre was
faulted in as one serial stream at ~100 MB/s), `VARDA_INPUT_IO_THREADS` (input GRIB reader threads, default 16),
`VARDA_INPUT_IFS_THREADS` (OpenMP team of the IFS branch, default a quarter of the threads clamped to 4..16),
`VARDA_INPUT_TRACE` (per-file and per-branch timelines in the input log). None changes a result.

One GPU job, A100-SXM4-80GB, 32 CPUs, 2026-10-23, `varda-forecast --realdata ... --precision fp16`
(device rollout, CUDA graph on, 4 steps, write queue 4, 32 OpenMP threads), the same command run three times in the
job. Run 1 starts with cold file caches and cold GPU code; runs 2 and 3 are warm.

| seconds | cold (run 1) | warm (run 2) | warm (run 3) |
|---|---|---|---|
| total | 20.07 | 10.53 | 10.41 |
| ready (input and model, overlapped) | 16.19 | 6.46 | 6.35 |
| input thread (critical path when warm) | 16.13 | 6.46 | 6.34 |
| weights load | 14.02 | 2.00 | 2.09 |
| rollout set-up | 1.46 | 1.33 | 1.30 |
| rollout, 4 steps | 3.21 | 3.41 | 3.40 |
| rollout callback copying y | 0.58 | 0.55 | 0.56 |
| rollout blocked on the writer | 0.00 | 0.00 | 0.00 |
| writing after the last step | 0.66 | 0.66 | 0.66 |
| GRIB encode / write (writer thread, summed) | 2.36 / 1.85 | 2.14 / 1.94 | 2.12 / 1.99 |

Run 3 in detail (`forecast.json`): `step_ms` 727.1, 722.0, 723.8, 721.1; input stages KENDA 3.86, IFS + regrid 1.33,
merge 0.25, x0 0.77 (6.32 in all); initial state written in 0.70 s; steps written in 0.73, 0.67, 0.65, 0.65 s;
GRIB post-processing 0.01 s.

Reading it:
- The GPU is not the limit. A warm forecast is 6.4 s of preparation (input: KENDA reading and vertical interpolation
  first), 3.4 s of rollout (4 x 0.72 s, as `docs/results.md` section 4) and 0.66 s to finish the last file.
- The weights load (2.0 s warm) and the rollout set-up (1.3 s) fit under the input time; the input is the critical path.
- The writer keeps up: the rollout never waits for it. The 0.56 s the callback spends copying `y` (4 x 0.14 s,
  device -> host -> queue) is on the rollout's critical path and is the remaining cost of the overlap.
- Before the writer thread and the transpose (separate run, same node and data, 16 threads): fp16 forecast 14.3 s;
  the per-step write went from 0.86 s to 0.63-0.69 s (separate runs before and after, `docs/results.md` section 5). The
  files are byte-identical to the earlier version's.
- Cold start (old binary): 20.1 s, 14.0 s of it loading the weights from the file system. Fixed by the prefetch (cold
  load 1.9-2.7 s, `docs/results.md` section 5, last subsection).

Against anemoi-inference, current binary (eb7f17b), same jobs and node, same frozen inputs, fp16, every GRIB file
written, 2026-10-23 (`docs/results.md` section 5): 24 h warm, varda.cpp 8.18 s against 135.8 s (16.6x; anemoi's process
wall clock 136.6 s minus the 0.8 s sha256 of the outputs that its driver computes); first run in a job 12.9 / 13.6 s
against 252.7 / 248.2 s (18-20x); 240 h warm 34.2 s against 310.7 s (9.1x). An earlier run with the old binary
(a9dcc66), same job for both engines: anemoi-inference 191 s warm (277 s first run), varda.cpp 10.4 s warm (23.4 s first
run); fp32 163 s against 41 s (4.0x), the only fp32 comparison. Details and configuration: `docs/results.md` section 5.

## 5. Limits

- **One verified date.** Everything is verified on 2026-10-22T15 only (`docs/realdata-reference.md`). Other dates run, but
  carry no parity claim (`docs/quickstart.md` section 6 records a run of another date).
- **Tolerances come from another date.** The rollout tolerances (`tolerances/rollout/varda.json`) were calibrated
  on the 2024-06-01 golden input. They are applied unchanged to this date (fp16: 0.49 of the per-step tolerance); they
  were not recalibrated and were not loosened.
- **AVX-512 numpy log path unverified.** The input side reproduces numpy's float32 `log` (`logf_np`), which is the
  AVX2/FMA3 routine of the reference host (EPYC 7713, dispatch `X86_V3`). A reference produced on an AVX-512 host
  would use another routine; that case has not been checked (`docs/io-input.md` section 6).
- **Constants run must be given.** See section 1. `--realdata` reads it from `pins.json`; the C++ has no "latest"
  logic. `python/fetch_inputs.py` makes the choice at download time as anemoi-inference makes it at start time (the
  latest run) and records it.
- **NaN policy.** The 2026-10-22 inputs give a NaN-free `x0`. A NaN in any input stops the run, naming the variable and
  the count, unless `--allow-nans` (the reference's `allow_nans: true`) is passed. In the output, NaN occurs only in the
  global files at the masked points (as a bitmap); `tp` follows the reference (negative clamp, accumulation).
- **Regrid matrix tied to the source layout.** The matrix is exact for IFS 0.25 deg messages on the
  -180..179.75 longitude layout, N320 target, no bitmap. A message with another geometry is refused.
- **Model and file coupling.** The io config's step (6 h) must equal the model's; the lam point count and order are
  those of the checkpoint's `source0` coordinates (`docs/realdata-reference.md` section 1).
- **Step files are not byte-identical to the reference's.** They differ from anemoi-inference's by the model error only;
  step 0 is identical (`docs/results.md` section 5).
