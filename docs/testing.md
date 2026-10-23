# Testing

Every suite in the repository, how to run it, and what it covers. Build first:
`source tools/env.sh; cmake -S . -B build-<you> && cmake --build build-<you> -j 32`. Model runs
need a compute node (CLAUDE.md, "Compute"); a job copies the binaries it runs to
`$VARDA_SCRATCH/jobs/<name>/` first.

| suite | what it checks | needs | run |
|---|---|---|---|
| ctest (`tests/*.cpp\|cu`) | kernels, GEMM wrapper, forcings, rollout state | CPU; GPU for the `.cu` tests; data for part of `test_forcings` | `cd build-<you> && ctest` |
| Python tests (`python/tests/test_convert.py`) | converted-file contract, restricted unpickler | `.venv`; converted files (env vars) | see below |
| `varda-parity` on real golden data | engine vs the anemoi golden data, all contract keys | converted model + golden + `tolerances/<model>.json` | see below |
| ctest, GRIB layer (`tests/io/*.cpp`) | numpy log, vertical interpolation, input and output building blocks, ecCodes link | ecCodes (`VARDA_GRIB=ON`); data for the real-data parts | `cd build-<you> && ctest` |
| real-data checks | x0, GRIB bytes and the rollout against anemoi-inference's run of 2026-10-22T15 | frozen reference data, converted model, GPU for the rollout | section 5 |
| synthetic suite (`tests/run_synthetic.sh`) | generality: models outside the real envelope vs an independent float64 reference | `.venv`, `build-*/varda-parity`; GPU for the CUDA part | `BIN=build-<you> tests/run_synthetic.sh` |

## 1. ctest

`CMakeLists.txt` globs `tests/*.cpp` and `tests/*.cu` (CUDA tests only when nvcc is present),
one executable and one ctest test each. All must exit 0. Some skip a part (message printed,
exit 0) when their data or device is missing.

- `test_cpu_kernels`: CPU fp32 kernels (Linear, LayerNorm, GELU erf/tanh, graph attention incl.
  zero in-degree and destination sub-ranges, thread-count invariance) against naive double references.
- `test_cuda_gemm`: cuBLASLt Linear wrapper, fp32 and fp16, odd rows, padded output width, tiny and wide K, repeatability.
- `test_cuda_kernels`: CUDA kernels (attention incl. C = 768, LayerNorm, fused residual + LayerNorm,
  GELU, normaliser, input/output assembly, every bounding type), fp32 and fp16 storage.
- `test_forcings`: dynamic forcings bitwise against anemoi's `ComputedForcings` fixtures
  (`$VARDA_SCRATCH/forcings/`), date helpers, thread invariance. Data part skipped if fixtures are missing.
- `test_rollout`, `test_rollout_cuda`: rollout state update (roll, feedback, forcings, constants) on synthetic
  windows, host and device bitwise equal; dates; config validation. The device test skips without a GPU.

The GRIB layer's tests (`tests/io/*.cpp`, built only when ecCodes is found) are listed in section 5.1.

Measured (build-synth, node with one A100, 2026-10-23): 6/6 passed, 18.5 s total
(`test_forcings` 12.7 s, others at most 2.7 s).

## 2. Python tests

```
VARDA_CONVERTED=a.safetensors:b.safetensors VARDA_WEIGHTS_PT=a/weights.pt:b/weights.pt \
  .venv/bin/python -I python/tests/test_convert.py
```
Details and the variables: `docs/tools.md`, docstring of `python/tests/test_convert.py`. Without
`VARDA_CONVERTED` the file-based tests skip. `python/tests/synthetic.py` is a generator, not a unittest
module; it is exercised by the synthetic suite (section 4).

## 3. varda-parity on real golden data

```
build-<you>/varda-parity --model $VARDA_SCRATCH/converted/<model>/model.safetensors \
    --check $VARDA_SCRATCH/golden/<model>/cpu-fp32-pyg.safetensors=cuda-fp32 \
    --device cuda --precision fp32 [--threads N] [--report out.json]
```
`--check GOLDEN=TABLE` compares every contract key (`docs/format.md` section 2, metric `rel_absmax`) against table
TABLE of `tolerances/<model>.json` (looked up by the model's `source.uuid`, or `--tolerances FILE`);
`--report-only GOLDEN` compares for information. A second, hook-free `predict_step` must reproduce the hooked
output bitwise. Exit 0 only if all keys are present, finite and within tolerance. Tables: `cuda-fp32` (CPU and CUDA
fp32 vs `cpu-fp32-pyg`), `cuda-fp16`, `cuda-fp16-vs-ref-fp16`. Recorded results: `docs/results.md` section 2.

## 4. Synthetic generality suite

Why: the two real checkpoints share one set of dimensions (C = 1024, 16 heads of 64, 16 layers, T = 2, A = 4, E = 3,
mlp = 4 C, exact GELU, all biases present, relu/hardtanh/fraction on fixed indices), so passing parity on them cannot
show that the engine has no model-specific assumption. The suite builds tiny models outside that envelope together
with a golden reference from an independent implementation of the spec, and runs the normal parity tool on them.

Files:

- `python/tests/synthetic.py` (numpy + stdlib only; `.venv/bin/python -I`): `make()` builds a model, `reference()` is
  the float64 implementation, `generate()` writes `<name>.model.safetensors` (`docs/format.md` section 1,
  `source.uuid` = `synthetic-<name>-v1`) and `<name>.golden.safetensors` (section 3: the float64 reference rounded to
  F32, tag `synthetic/<name>`, `checkpoint_uuid`, a manifest with generator, seed and the full config; all contract keys
  for that `n_proc`). Deterministic (`numpy.random.default_rng(seed)`). Options: `--out DIR`, `--only a,b`,
  `--write-tolerances DIR`, `--check-tolerances DIR`, `--hash`.
- `tolerances/synthetic-<name>.json`, one file per model: `load_tolerances` (`apps/common.h`) matches one
  `checkpoint.uuid` per file, so the suite has one file per uuid; the runner passes it with `--tolerances`. The tables
  are generated by `synthetic.py --write-tolerances tolerances` (constants `FP32_TOL`, `FP16_TOL` there), and the runner
  fails if a committed file differs from the generator (`--check-tolerances`).
- `tests/run_synthetic.sh`: generates into `$VARDA_SCRATCH/synthetic/` when files are missing or the SHA-256 of
  `synthetic.py` differs from `generator.sha256`; then for every model runs `$BIN/varda-parity`:
  CPU fp32 with 1 and 8 threads, CPU fp32 with `VARDA_CPU_CHUNK_ROWS=37` (8 threads), CUDA fp32, CUDA fp32 with
  `VARDA_DEC_CHUNK_ROWS=128`, CUDA fp16. Prints a summary with the worst `rel_absmax` and its key per run and exits
  non-zero on any failure or error. CUDA runs are skipped with a note when no GPU is visible (the final line then
  warns that this is not a full pass). Arguments `cpu` / `cuda` select a part; env: `BIN`, `SYN_DIR`, `MODELS`,
  `OUT`, `SKIP_CUDA`. Logs and JSON reports: `$OUT` (default `$SYN_DIR/runs/<timestamp>`).

```
source tools/env.sh
cmake -S . -B build-synth && cmake --build build-synth -j 32
# a job copies the binary first:  mkdir -p $VARDA_SCRATCH/jobs/synth/run/bin; cp build-synth/varda-parity $VARDA_SCRATCH/jobs/synth/run/bin
BIN=$VARDA_SCRATCH/jobs/synth/run/bin srun $VARDA_SLURM_CPU tests/run_synthetic.sh cpu
BIN=$VARDA_SCRATCH/jobs/synth/run/bin srun $VARDA_SLURM_GPU tests/run_synthetic.sh
```
16 CPUs and 32 GB are enough for either part. Wall time: generation about 10 s (the largest file, e, is 315 MB); the
whole suite about 1 minute on a GPU node.

### Configurations

All have 300-ish data nodes and 41-70 hidden nodes, destinations with zero in-degree (every 7th), random edge order,
and edge counts per destination from 1 to 9 (encoder), 6 (processor), 4 (decoder).

| name | C | heads x dim | mlp | n_proc | T | A | E | GELU | latent skip | residual step | absent biases | variables (in/out), bounding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 256 | 32 x 8 | 520 | 1 | 3 | 3 | 2 | tanh | yes | -2 | all q,k,v,self,edge, dec.out_norm, enc.ln_mlp | 10/8; relu, hardtanh(-0.5, 0.75), fraction(total = 2 vars) |
| b | 512 | 2 x 256 | 1024 | 3 | 1 | 5 | 1 | erf | no | 0 | none | same as a |
| c | 768 | 12 x 64 | 1536 | 2 | 2 | 4 | 3 | erf | yes | -1 | none | same as a |
| d | 2048 | 8 x 256 | 512 | 2 | 2 | 4 | 3 | tanh | yes | -1 | as a | same as a |
| e | 1024 | 16 x 64 | 4096 | 4 | 2 | 4 | 3 | erf | yes | -1 | none | 84/72 with the real index spaces: tp (relu, out 22), tcc (hardtanh 0..1, out 23), fraction on out 4,5,6 with total [23] |
| f | 256 | 4 x 64 | 640 | 2 | 2 | 2 | 1 | tanh | yes | -1 | none | 5/3: one output each: relu, hardtanh(-0.4, 0.6), fraction(0.05, 0.9; total = the hardtanh output) |

a-d are the reviewer's, e and f were added. Residual index lists are permuted in a-d (input index != output index), `mappers`
is present in the config of every model.

### What the reference does (independent of the engine; follows anemoi's GraphTransformer)

Checked against anemoi's maths, point by point: normalisation is two fp32 roundings, `x*mul` then `+add` (as anemoi's normaliser;
the reviewer's reference used float64 here, so `in.x_norm` differed from the engine at the rounding level; now it is
bitwise and so are all derived inputs); `x_data = [t0 vars | t1 vars | node attr]`; per block: `x_r = lin_self(LN(xd))`
added before the projection, residual with the un-normalised `xd`, edge term in both k and v, scale 1/sqrt(head_dim),
softmax with max subtraction and `+1e-16`, zero in-degree gives 0; mapper blocks use `ln_src` for the source and
`ln_dst` for the destination, processor blocks `ln_dst` for both; encoder source = `emb_src(x_data)`, destination =
`emb_dst(hidden attr)`; decoder source = latent (not embedded), destination = `emb_dst(raw x_data)`, then its own
LayerNorm and Linear; `latent = proc(z) + z` only if `latent_skip`; residual `y[out] += x_norm[step][in]` in fp32
normalised space, then the boundings in config order with a fraction bounding using the already bounded total (single
total broadcast, or one per index), then `(y - add) / mul`. Changes made to the reviewer's version: the fp32
normalisation above, the missing keys `in.x`, `in.x_norm` (shape `[1,T,1,G,V]`), `enc.x_dst`, output keys with shape
`[1,1,1,G,V]`, config `mappers`, uuid, layouts e/f, and I64 tensors written first (the engine's reader requires aligned
tensors, the safetensors format itself does not).

### Tolerances

Tables `synthetic-fp32` and `synthetic-fp16` in each `tolerances/synthetic-<name>.json`:

- `synthetic-fp32`: 1e-5 for every key, 1e-6 for `in.x`, `in.x_norm`, `enc.x_src`, `enc.x_dst` (copies and input
  assembly; bitwise in every run). The same table is used for CPU and CUDA. Measured worst: 9.2e-7 (CPU) and 7.9e-7
  (CUDA), i.e. at most 0.09 of the tolerance. The tolerance is about 10x the measured worst, the floor used for the
  real models (`docs/format.md` section 4); an error of a few 1e-5 would already point at a real defect.
- `synthetic-fp16`: 3e-3 for every non-exact key, 1e-6 for the exact ones. Measured worst over all models and keys:
  1.35e-3 (`proc.N.out` of e); per key it is 5.8e-4 (`enc.emb_dst`) to 1.35e-3 (processor outputs); a single fp16
  rounding is already 2^-11 = 4.9e-4 relative to the element, and the max over the elements is of the order of that, so
  the error is flat in depth. 3e-3 = 2.2x the worst, the headroom asked for; the measured spread between models is
  1.02e-3 to 1.35e-3. fp16 is compared against the float64 reference rounded to fp32.

Never loosen these to make a run pass without saying so.

### Measured results

Worst `rel_absmax` over all contract keys (the key of the worst is in the run's `summary.tsv`); one job on a GPU node:
CPU on its host CPU (EPYC 7713), CUDA on its A100; build-synth, with the CUDA fixes as of 2026-10-23. Every run passed, hook-free output bitwise equal to the hooked output.

| model | CPU fp32 1 thr | CPU fp32 8 thr | CPU fp32 chunk 37 | CUDA fp32 | CUDA fp32 dec chunk 128 | CUDA fp16 |
|---|---|---|---|---|---|---|
| a | 6.39e-07 | 6.39e-07 | 7.75e-07 | 3.23e-07 | 3.23e-07 | 1.02e-03 |
| b | 8.15e-07 | 8.15e-07 | 9.15e-07 | 6.47e-07 | 6.47e-07 | 1.15e-03 |
| c | 6.81e-07 | 6.81e-07 | 5.33e-07 | 4.70e-07 | 4.70e-07 | 1.16e-03 |
| d | 5.89e-07 | 5.89e-07 | 7.44e-07 | 7.91e-07 | 7.44e-07 | 1.13e-03 |
| e | 7.22e-07 | 7.22e-07 | 6.88e-07 | 6.95e-07 | 6.95e-07 | 1.35e-03 |
| f | 6.43e-07 | 6.43e-07 | 6.43e-07 | 2.99e-07 | 2.99e-07 | 1.23e-03 |

CPU output is bitwise independent of the thread count (the 1 and 8 thread columns agree to the printed digits).
Note that C = 768 (c) passes on CUDA in this build; on an earlier build it failed.
The CUDA decoder chunking (`VARDA_DEC_CHUNK_ROWS=128`, three chunks at 300 nodes) is exercised by every model (a decoder
chunk of the real size is at least 2^18 rows, so without the variable it never splits here).

Runs: a CPU-only pass on a CPU node (the first full pass after the alignment fix), a CUDA-only pass, then the whole
suite in one GPU job (the numbers above). An earlier run failed on I64 alignment of the first generator version.

### Adding a configuration

Add an entry to `CONFIGS` in `synthetic.py` (and a layout to `LAYOUTS` if the variable spaces differ), run
`.venv/bin/python -I python/tests/synthetic.py --write-tolerances tolerances`, add the name to `MODELS` in
`tests/run_synthetic.sh`, commit the new tolerance file. The generator hash changes, so the next run regenerates.

## 5. Real-data I/O

Reference: anemoi-inference's own run of the Hugging Face config on 2026-10-22T15 with every stage dumped
(`docs/realdata-reference.md`, runs `ref-fp16` and `ref-fp32` under `$VARDA_SCRATCH/realdata/20261022T15/`). Results:
`docs/results.md` section 5. Build with ecCodes (`source tools/env.sh` sets `ECCODES_HOME` and
`VARDA_COSMO_DEFINITIONS`); the apps are in `apps/io/`.

### 5.1 ctest targets (`tests/io/`, one executable and one ctest test each)

Parts that need data skip with a message and exit 0 when their variables are unset.

- `test_io_link`: the GRIB layer links against the ecCodes of the reference, and the COSMO definitions give KENDA-CH1
  messages their COSMO short names. Real-data part: `VARDA_KENDA_SAMPLE` = a KENDA-CH1 `T` file, plus
  `VARDA_COSMO_DEFINITIONS`.
- `test_io_numpy_log`: `logf_np` against numpy's float32 `np.log`: special values and a table recorded with numpy
  2.5.4 on an EPYC 7713 (dispatch X86_V3). With `--stdin` it is the exhaustive check over all 2 139 095 039 positive
  finite float32 (numpy's output streamed in, command in the file header; 0 differ, CPU node, EPYC 7713).
- `test_io_vertical`: the vertical interpolation and destaggering (`interpolate_extrapolate`, `destagger_z`) bitwise on 6
  synthetic columns whose values were produced by the plugin in the reference env; every extrapolation branch and the
  top clamp. No data needed.
- `test_io_input`: dates, state handling, regrid matrix files, `y = W x` in stored order. Real-data part, needs all of
  `VARDA_REGRID_MATRIX`, `VARDA_N320_TEMPLATE` (HF `templates/ifs-levtype=sfc.grib`), `VARDA_REGRID_REF_DUMPS` (a
  reference run's `dumps/`) and `VARDA_COSMO_DEFINITIONS`: MIR's own `regrid.1` input messages through our regridder
  must decode bitwise like MIR's output messages.
- `test_io_output`: 45 checks of the output side: Python-style parsing, `tp` accumulate and rescale, `assign_mask`, NaN
  and inf encoding, path rendering, the io config, `build_keys`, patch names, template matching, and a synthetic
  end-to-end write on the real templates. Needs `VARDA_IO_CONFIG` (default
  `$VARDA_SCRATCH/realdata/assets/varda-io.json`, produced by `python/export_io_config.py`), and for the end-to-end part
  `VARDA_TEMPLATES` (default `$VARDA_SCRATCH/hf/templates`) and `VARDA_COSMO_DEFINITIONS`.

### 5.2 Input side: `varda-prepare --compare`

```
R=$VARDA_SCRATCH/realdata/20261022T15
build-out/varda-prepare --model $VARDA_SCRATCH/converted/varda/model.safetensors \
    --io-config $VARDA_SCRATCH/realdata/assets/varda-io.json \
    --regrid-matrix $VARDA_SCRATCH/realdata/assets/regrid-ifs0p25-n320.safetensors \
    --n320-template $VARDA_SCRATCH/hf/templates/ifs-levtype=sfc.grib --realdata $R --out OUT \
    --compare $R/ref-fp16/dumps [--mir-mat <MIR .mat file>] --threads 64
```
Compares every stage with the reference's dumps (sources, regrid outputs, cutout, `x0` per column, step-0 state) and exits
1 on any difference; expected: all bitwise (`docs/io-input.md` section 4). Run on a compute node (`srun $VARDA_SLURM_CPU`, 64 cores;
about 8 s).

### 5.3 Output side: encoder-only byte identity

`varda-grib-write` feeds the reference's own states (`state0.pre`, `step<s>.y`) to the writer, so the model is out of the
loop, then `varda-grib-compare` checks the files against the reference's:
```
build-out/varda-grib-write --io-config $VARDA_SCRATCH/realdata/assets/varda-io.json \
    --templates $VARDA_SCRATCH/hf/templates --model $VARDA_SCRATCH/converted/varda/model.safetensors \
    --dumps $R/ref-fp16/dumps --out OUT --date 2026-10-22T15:00:00
build-out/varda-grib-compare --require-bytes OUT/outputs $R/ref-fp16/outputs     # exit 0: 676/676 messages byte-identical
```
`varda-grib-compare TEST REF` compares messages in file order, every header key, bitmap and values (default
tolerance 0); `--rel-tol`, `--abs-tol`, `--ignore-key`, `--max-report`, `-q` as in `docs/io-output.md` section 3. Use
`$R/ref-fp32` for the fp32 states. Results: `docs/io-output.md` section 4.

### 5.4 Rollout judged against the reference

`export_rollout_realdata.py` turns a reference run's stages into a rollout file (needs a rollout file of the same
checkpoint as template for the index tensors), and `varda-rollout` judges the real-data rollout per step with the
rollout tolerances (`docs/rollout-engine.md` section 4):
```
.venv/bin/python -I python/reference/realdata/export_rollout_realdata.py $R/ref-fp16 \
    $VARDA_SCRATCH/rollout/ref/cpu-fp32-pyg/rollout.safetensors          # -> $R/ref-fp16/rollout.safetensors
build-out/varda-rollout --model $VARDA_SCRATCH/converted/varda/model.safetensors \
    --input $R/ref-fp16/rollout.safetensors --compare $R/ref-fp16/rollout.safetensors \
    --device cuda --precision fp16 --check-forcings --verify-host --json rollout.json
```
For fp32 the pair has no tolerance table; the numbers are reported only.

### 5.5 End to end: `varda-forecast`

```
build-out/varda-forecast --model ... --io-config ... --regrid-matrix ... --templates $VARDA_SCRATCH/hf/templates \
    --realdata $R --out OUT --precision fp16 --dump-y
build-out/varda-grib-compare --require-bytes -q OUT/outputs $R/ref-fp16/outputs
```
Expected: step 0 files (`*_0.grib`) byte-identical (140 messages); step files differ by the model error, so the compare
reports failures at tolerance 0 (by construction; with `--rel-tol` set to the scale of interest it is a report, not a
verdict). The verdict on the model output is 5.4; `OUT/y.safetensors` (`--dump-y`) must equal the `y` of
`varda-rollout --out` bitwise (`docs/results.md` section 5). Rerun with the same arguments: the bytes must equal. Jobs copy the
binaries first (`$VARDA_SCRATCH/jobs/<name>/bin`).

## Entrypoint check

`./varda forecast 2026-10-22T15` with `VARDA_INPUTS` pointing at a directory holding the frozen date (symlink
`20261022T15 -> $VARDA_SCRATCH/realdata/20261022T15`) must give step-0 files byte-identical to
`$VARDA_SCRATCH/realdata/20261022T15/ref-fp16/outputs/*_0.grib` and all 10 files byte-identical to `varda-forecast` run
with explicit flags (10/10 in two separate runs, `docs/quickstart.md` section 6). `tools/make_assets.py check` verifies the assets bundle.

