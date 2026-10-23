# Results

Every number names its configuration. Parity metric: `rel_absmax` (docs/format.md §2);
"ratio" is rel_absmax / tolerance from `tolerances/<model>.json`. Varda = varda
(bit-identical checkpoint). Full per-key tables: docs/cpu.md, docs/cuda.md, and the JSON
reports in the job directories named below.

## 1. Converter

`python/convert.py` on the HF checkpoint and on dev: every `weights.pt` tensor of
the Python engine is bitwise equal (names, dtypes, shapes, bytes); config equal except
`source.checkpoint`/`mtime` and the added `forcings` section; plus `grid.latitudes/longitudes`
(F64, bitwise the checkpoint's supporting arrays). Independently re-checked by the integrator
(data-section sha256 and tensor tables). Details: docs/tools.md, docs/forcings.md.

## 2. One-step parity (integrator re-runs from a clean build)

| model | device / precision | vs | table | worst ratio (key) | hook inert | job |
|---|---|---|---|---|---|---|
| dev | CPU fp32, 64 threads | cpu-fp32-pyg | cuda-fp32 | 0.101 (proc.1.out) | bitwise | 283438 |
| Varda | CPU fp32, 64 threads | cpu-fp32-pyg | cuda-fp32 | 0.181 (dec.block_out) | bitwise | 283438 |
| dev | CUDA fp32 | cpu-fp32-pyg | cuda-fp32 | 0.314 (dec.block_out) | bitwise | 283534 |
| dev | CUDA fp16 | cpu-fp32-pyg | cuda-fp16 | 0.656 (dec.out) | bitwise | 283534 |
| dev | CUDA fp16 | cuda-fp16-triton | cuda-fp16-vs-ref-fp16 | 0.733 (proc.8.out) | bitwise | 283534 |
| Varda | CUDA fp32 | cpu-fp32-pyg | cuda-fp32 | 0.329 (dec.block_out) | bitwise | 283534 |
| Varda | CUDA fp16 | cpu-fp32-pyg | cuda-fp16 | 0.670 (proc.2.out) | bitwise | 283534 |
| Varda | CUDA fp16 | cuda-fp16-triton | cuda-fp16-vs-ref-fp16 | 0.620 (proc.7.out) | bitwise | 283534 |

All 30 contract keys present and NaN-free in every run; input-assembly keys (`in.*`,
`enc.x_*`) are bitwise everywhere, and in fp32 so are the embeddings. CPU outputs are bitwise
independent of the thread count; CUDA graph replay, eager execution and hooked runs are bitwise
equal. Jobs: `$VARDA_SCRATCH/jobs/int-cpu-verify/`, `$VARDA_SCRATCH/jobs/int-gpu-verify/`.

## 3. Speed: one forward pass, same job

NVIDIA A100-SXM4-80GB, driver 595.91.07, node <node>; hook-free, warm (median of 20 after 5
warm-up calls, CUDA events), input on the device, TF32 off; `bench/run_bench.sh`, jobs 283535
(Varda) and 283536 (dev). varda.cpp replays its CUDA graph; the Python fast path
uses torch.compile + CUDA graph; the original is anemoi with the triton attention backend
under fp16 autocast (or plain fp32).

| model | precision | original anemoi | Python fast path | **varda.cpp** | vs original | vs Python |
|---|---|---|---|---|---|---|
| Varda | fp16 | 1338.0 ms | 880.4 ms | **724.3 ms** | 1.85x | 1.22x |
| Varda | fp32 | 8067.5 ms | 7572.9 ms | **7438.5 ms** | 1.08x | 1.02x |
| dev | fp16 | 79.0 ms | 38.0 ms | **31.7 ms** | 2.49x | 1.20x |
| dev | fp32 | 379.1 ms | 333.6 ms | **329.8 ms** | 1.15x | 1.01x |

First call (load excluded): varda.cpp 0.9 s / 7.7 s (Varda fp16 / fp32), against 4.9 s / 12.7 s
for the original and 16-36 s for the Python fast path (torch.compile, warm cache). Peak device
memory (cudaMemGetInfo, incl. CUDA context and ~3.5 GB of the bench's own buffers on Varda):
varda.cpp 24.7 / 42.2 GB, original 24.2 / 38.0 GB, Python 19.7 / 34.8 GB.

Where the fp16 time goes (nsys, docs/cuda.md): GEMMs 541 ms at 249 TFLOP/s (80% of the
312 TFLOP/s peak, 135 TFLOP per forward), attention 70 ms, GELU 48 ms, add+LayerNorm 46 ms.
fp32 is SGEMM-bound at 97.6% of the FMA peak: only TF32 could move it, which the 1e-5 fp32
tolerances exclude.

CPU (cpu-short, AMD EPYC 7713, 64 threads on 64 cores, fp32): Varda forward 60 s warm, peak RSS
24.7 GB; dev 5.3 s. For comparison (from the Python project): anemoi CPU 196-213 s,
Python plain path 242 s, both at 64 threads.

## 4. Rollout (24 steps × 6 h, from the golden input, newest slot 2024-06-01T00 UTC)

References: the original model driven by anemoi-inference 9505dd0's own loop (docs/rollout.md),
replaying identical forcings; per-step tolerances `tolerances/rollout/varda.json`
(sha256 `7b020cd8…`; fp32 tables and the fp16 ensemble criterion, docs/rollout-engine.md §4).
GPU rows: integrator re-run from a clean build, job 284075 (A100-SXM4-80GB),
`$VARDA_SCRATCH/jobs/int-rollout-verify/`; CPU row: job 284016 (cpu-long, 64 threads).

| engine | vs reference | table | worst ratio (key, step) | entries above per-step tol | means / mean tol (y, y_norm, window, window_norm) |
|---|---|---|---|---|---|
| CPU fp32 | cpu-fp32-pyg | fast-fp32-vs-cpu | 0.58 (y_norm, 23) | 0/96 | – |
| CUDA fp32 | cpu-fp32-pyg | fast-fp32-vs-cpu | 0.50 (y_norm, 8) | 0/96 | – |
| CUDA fp16 | cpu-fp32-pyg | fast-fp16-vs-cpu | 0.73 (y_norm, 13) | 0/96 | 0.54 / 0.49 / 0.54 / 0.53 |
| CUDA fp16 | cuda-fp16-triton | fast-fp16-vs-ref-fp16 | 0.84 (window_norm, 5) | 0/96 | 0.53 / 0.50 / 0.54 / 0.55 |

Bitwise checks (all pass): valid dates (23/23); computed forcings vs the reference's recorded
forcings (0 of 23 × 1 688 650 × 5 differ), and the forcing columns of every window the model
sees; device loop vs host loop (y and windows, every step); computed vs replayed forcings;
in-place (9505dd0) vs roll (eaeb36fa) constants (equal here: the constants are equal in both
slots of every window). dev (1 h steps, eaeb36fa references): CPU 0.56, CUDA fp32
0.58, fp16 1.23 / 1.02 within the fp16 allowance (as the Python fast path: 1.22 / 1.00), all PASS.

Speed (CUDA fp16, graph on, forcings computed and overlapped): 727.7 ms per step wall,
726.1 ms GPU, against 724–725 ms for the forward pass alone: the state update costs ~2 ms
(0.3%). fp32: 7442.6 ms per step. Host-level loop (host copies every step): 1007 ms per step.

## 5. Real-data forecast (2026-10-22T15)

KENDA-CH1 analyses (t-6h, t) and ECMWF open data frozen on 2026-10-23 (`docs/realdata-reference.md`), the Hugging
Face config `inference_opendata.yaml`, 4 steps of 6 h, initial state written. References: the original model run by
anemoi-inference 9505dd0 (`ref-fp16`: checkpoint precision, CUDA fp16 autocast; `ref-fp32`), dumps under
`$VARDA_SCRATCH/realdata/20261022T15/<run>/dumps`. varda.cpp: converted file `converted/varda-v2`, A100-SXM4-80GB
(GPU rows) or EPYC 7713 (input rows), details in the documents named in the last column.

| check | configuration | result | job / document |
|---|---|---|---|
| x0 against the reference's `input_tensor`, 79 columns incl. 9 computed forcings | CPU, numpy dispatch X86_V3, 1, 16 and 64 threads | **bitwise** (0 of 266 806 700 values differ); hashes equal across thread counts; the same against all three reference runs | 285351, 285354; `docs/io-input.md` section 5 |
| input stages | same | bitwise: source states lam_0 and global (prognostics and constants), vertical interpolation, gh -> z, 140 regridded N320 fields (75 891 200 values), cutout, merged coordinates, step-0 state (70 fields) | same |
| regrid matrix vs MIR's cache file | 542 080 x 1 038 240, nnz 1 505 824 | identical (sha256 `1e5fac3b...aed533`) | same |
| `logf_np` vs numpy float32 `np.log`, every positive finite float32 | numpy 2.5.4, X86_V3 | 0 of 2 139 095 039 differ | 285295 |
| GRIB writer alone, fed the reference's own states (encoder-only) | `varda-grib-write`, 10 files | **676 / 676 messages byte-identical** to anemoi-inference, fp16 and fp32 references; negative control (fp16 states against the fp32 outputs) fails as it must | 285346, 285350 (`docs/io-output.md` section 4); 285381 and 285692 with the parallel encoder |
| `varda-forecast` step 0 (initial state, 70 + 70 messages) | CUDA fp16 and fp32 | 140 / 140 messages byte-identical to anemoi-inference's | 285381 |
| `varda-forecast` steps 1-4, per GRIB message | CUDA fp16 / fp32, zero-tolerance compare | differ by model error: worst message `rel_absmax` (to the field max) 7.9e-3 fp16, 1.07e-5 fp32 (536 messages each); "FAIL" is by construction, the compare has tolerance 0 | 285381 |
| judged rollout, fp16 | `varda-rollout --compare` on the reference's `ref-fp16` stages (`export_rollout_realdata.py`), CUDA fp16, graph on, table `fast-fp16-vs-ref-fp16` | **PASS, worst 0.49 x per-step tolerance** (y, step 0); 0 / 16 entries above; y rel_absmax 2.6e-5 .. 5.4e-5, y_norm 1.9e-3 .. 3.3e-3; means over steps / mean tolerance y 0.26, y_norm 0.33, window 0.18, window_norm 0.16 | 285378 |
| rollout, fp32 | same, CUDA fp32 against anemoi CUDA fp32 | no table for this pair, reported only: y rel_absmax <= 3.0e-7 (max_abs <= 0.0625), y_norm <= 2.9e-6 | 285378 |
| computed forcings, dates, windows | both precisions | forcings 0 of 3 x 1 688 650 x 5 differ; forcing columns of the windows bitwise; dates equal; device loop == host loop bitwise (y every step, windows) | 285378 |
| `varda-forecast` y == `varda-rollout` y (from the reference x0) | fp16 and fp32 | bitwise | 285378 |
| reproducibility | two `varda-forecast` runs per precision | bytes equal | 285381 |
| output after the writer changes (parallel encoding, one transpose of y per step, overlapped file writes) | fp16 and fp32 | 10 / 10 files byte-identical to the earlier version's; encoder-only 676 / 676 again at 16 and 32 threads | 285692, 285715 |

Notes on the table:
- The rollout tolerances are those calibrated on the 2024-06-01 golden input (`tolerances/rollout/varda.json`),
  applied unchanged to this date. Not recalibrated, not loosened.
- The first end-to-end job (285378) also ran `varda-grib-compare` on the written files; it found no matching files, because of a
  bug: `varda-forecast` passed `OUT/outputs` as the base for the configured relative path, which already starts with
  `outputs/`, so the files went to `OUT/outputs/outputs/` (found there, fixed afterwards; those lines are not a result). Jobs 285381 and later write `OUT/outputs/<dateTime>_<step>.grib`.
- The fp16 and fp32 reference runs differ from each other by rel_absmax <= 3.7e-3 at step 0 and <= 5.6e-3 at step 3
  (`docs/realdata-reference.md`), which is the scale to read the 7.9e-3 fp16 figure against.

### Timing of `varda-forecast` (job 285715, <node>, A100-SXM4-80GB, 32 threads, fp16, device rollout, CUDA graph on)

| | cold, run 1 | warm, runs 2 / 3 |
|---|---|---|
| total | 20.07 s | 10.53 / 10.41 s |
| ready (model load and input in parallel) | 16.19 s | 6.46 / 6.35 s |
| rollout, 4 steps (`step_ms` run 3: 727, 722, 724, 721) | 3.21 s | 3.41 / 3.40 s |
| rollout blocked on the writer / callback copying y | 0.00 / 0.58 s | 0.00 / 0.55 and 0.56 s |
| writing after the last step | 0.66 s | 0.66 / 0.66 s |

Before the writer thread, the transpose and the overlapped file writes: fp16 forecast 14.3 s warm (job 285381, 16
threads), per-step write 0.86 s at 16 and 32 threads (job 285651) against 0.63-0.69 s after (job 285692). Breakdown
and reading: `docs/forecast.md` section 4. These are varda.cpp numbers only, runs on one node; the comparison with
anemoi-inference is the next section.

### Speed vs anemoi-inference, same job

Job 285717 (`$VARDA_SCRATCH/jobs/bench-forecast/`), node <node>, A100-SXM4-80GB, driver 595.91.07, 32 CPUs, the same
frozen 2026-10-22T15 inputs, 1-day (4-step) forecast, all 10 GRIB files written including step 0.
- anemoi-inference: `python/reference/realdata/run_reference_realdata.py --no-dumps` (plain runner, HF config, offline
  replay, `python -O`, mapper chunks 16, triton attention, fp16 autocast = the checkpoint's precision), with a shared
  earthkit/MIR cache (run a builds MIR's weight matrix, run b reuses it).
- varda.cpp: `varda-forecast`, CUDA graph on, device rollout, write queue 4.
- Times are the wall clock of the whole process (Python start and imports included for anemoi; the C++ process includes
  its weights load).

| precision | run | anemoi-inference | varda.cpp | ratio |
|---|---|---|---|---|
| fp16 | first (cold page cache; MIR matrix built by anemoi) | 277.1 s (execute 184.0 s) | 23.4 s (forecast 18.7 s: load 13.2, input 14.0) | 11.8x |
| fp16 | second (warm) | 191.1 s (execute 133.3 s) | **10.4 s** (forecast 9.74 s: ready 6.01, rollout 3.23, +0.50 s tail) | **18x** |
| fp32 | warm caches, after the fp16 runs | 163.4 s (execute 155.9 s) | 41.2 s (rollout 30.3 s = 4 GPU steps of ~7.4 s) | 4.0x |

- anemoi's no-dumps outputs are byte-identical to the dumped reference runs (10 / 10 files for fp16 a, fp16 b and fp32):
  the dump hooks do not change its results, and it is run-to-run deterministic here.
- The 244 s / 362 s of the dumped reference runs are not a speed baseline (they wrote ~91 GB of dumps).
- Single job, one run per cell; warm and cold refer to file and MIR caches, not to the GPU alone. Most of anemoi's time
  is outside the model: its four `predict_step` calls took 10.9 s in fp16 (5.77 + 3 x ~1.71 s, measured with CUDA
  synchronisation in the dumped run, job 285290) of a 133 s `execute`; the rest is input pre-processing, regridding and
  output encoding. So the ratio is mostly an I/O-pipeline ratio; like-for-like GPU forward times are in section 3.
