# Real-data reference: Varda on KENDA-CH1 + IFS open data, GRIB in, GRIB out

This is the reference that the C++ real-data I/O (`docs/spec-io.md`) is checked against. It is the original
model, run by anemoi-inference itself with the Hugging Face config `inference_opendata.yaml`, from a frozen
copy of the open data. Every intermediate stage is dumped. All of it can be rerun offline.

Code (tools only, never the engine): `python/reference/realdata/` and `python/export_regrid_matrix.py`.
Artefacts: `$VARDA_SCRATCH/realdata/` (`$VARDA_SCRATCH`: `tools/env.local.sh`).

## 1. Results at a glance

| check | result |
|---|---|
| initial date | **t = 2026-10-22T15:00Z**, the notebook's rule (the latest KENDA-CH1 analysis with t.hour % 3 == 0 and t−6h published), picked at 07:25Z on 2026-10-23 |
| offline replay vs real downloads | online stub run == offline stub run, **bitwise**, for all 56 dumps and all 10 GRIB files (`compare-stub-offline-vs-online.json`) |
| pre-processing, CPU node vs GPU-node host | stub run (CPU node) == fp16 run (GPU node), bitwise, for all 25 input-side dumps (`compare-stub-vs-fp16-inputs.json`) |
| fp16 vs fp32 runs | input side and step-0 GRIB files identical; outputs differ by rel_absmax ≤ 3.7e-3 at step 0 and ≤ 5.6e-3 at step 3 (worst variables q_925, v_300) |
| precision the runner picks | checkpoint `precision: 16-mixed` → `torch.autocast(cuda, float16)`. The model output `y` is **float32** |
| MIR regrid as a matrix | `y = W x` in double, rows summed in stored order: **bitwise equal** to MIR's own output before the 16-bit encode, on all 140 regridded real fields. After the 16-bit re-encode through ecCodes, bitwise equal to the run's values (140/140) |
| MIR matrix of the run | `1e5fac3b…aed533` (sha256 of the `.mat`), source layout −180..179.75, MIR 1.29.0 `linear`. A fresh MIR in another process rebuilds it byte-identically |
| lam_0 points vs ICON-CH1 native cells | same count (1 147 980) and same order. The checkpoint's `source0/*` differ from the published CLAT/CLON by ≤ 1.2e-4° lat and ≤ 2.4e-4° lon (GRIB packing of CLAT/CLON) |
| NaN | **none** in any input, source state, merged state or `x0`. NaN occurs only in the global output, at the 1 410 masked N320 points, from `assign_mask` |
| x0 vs engine layout | `step0.window` [1, 2, G, 79] is the golden `in.x` layout. Its variable order equals the engine's `variables.input`, and `variables.output` equals the runner's output order. It equals the runner's `input_tensor` transposed, and its 70 retrieved columns equal `lam.<var>` ++ `global.<var>[:, global/cutout_mask]` bitwise |

## 2. Data

**Frozen cache** `$VARDA_SCRATCH/realdata/20261022T15/` (4.31 GB, 133 files), downloaded on the login node with
`fetch_realdata.py` at 2026-10-23T07:25–07:28Z:

```
cache/kenda/raw/20261022T090000/<VAR>.grib2      all 61 STAC items valid at t-6h (the run uses 11 of them:
cache/kenda/raw/20261022T150000/<VAR>.grib2        P PMSL PS QV T T_2M TD_2M U U_10M V V_10M)
cache/kenda/constants/{horizontal,vertical}_constants_icon-ch1-eps.grib2   ICON-CH1 collection assets
cache/kenda/stac/<YYYYmmddTHHMMSS>.json, collection_icon-ch1.json          the STAC documents, replayed offline
cache/ecmwf/files/<run>-<step>h-oper-fc.grib2 (+ .index)                   3 full IFS open-data files (google mirror)
pins.json                                                                  the ECMWF run choice at download time
manifest.json                                                              sha256 + size of every file
```

**Pins** (`pins.json`). At 07:27Z, `ecmwf.opendata.Client(source="google").latest(type="fc")` returned
**2026-10-22T18**. This is the value `oper-ecmwf-opendata` would have used. With it:

| use | target | run, step | file |
|---|---|---|---|
| global prognostics, t−6h | 2026-10-22T09 | 06z + 3h | `20261022060000-3h-oper-fc.grib2` |
| global prognostics, t | 2026-10-22T15 | 12z + 3h | `20261022120000-3h-oper-fc.grib2` |
| global constants (`constant: true` → latest run) | both dates | 18z + 0h | `20261022180000-0h-oper-fc.grib2` |

KENDA-CH1 needs both analyses (t−6h, t) plus the static constants. KENDA open data are kept about 25 h, IFS open
data about 12 runs.

**Manifest** (`20261022T15/manifest.json`): the date, input dates, pins, sha256 and size of every cached file, the
HF templates/config/metadata.yaml/pyproject sha256, the checkpoint sha256 (`1ad7a7a6…a9c7d8`, the HF file), and the
env lock sha256. Templates: `$VARDA_SCRATCH/hf/templates/` (only `templates/` of HF `MeteoSwiss/Varda-single-1.0`,
revision `89fc5a6b…` in `templates/REVISION.json`; the temporal downscaler was not downloaded).

## 3. Environment

`$VARDA_SCRATCH/ref-oper/.venv`, built from the HF `pyproject.toml`: anemoi-inference 9505dd0,
anemoi-plugins-meteoswiss a1488e1, anemoi-plugins-ecmwf-inference 0.7.0, eccodes 2.47.0 + eccodeslib 2.47.3,
eccodes-cosmo-resources 2.44.0.1, mir-python/mirlib 1.29.0, earthkit-data 0.19.2, torch 2.6.0+cu124.
`uv pip freeze`: `python/reference/realdata/requirements-ref-oper.lock.txt` (sha256 `ed99b85b…62c7fd`).

The driver sets the notebook's environment before anything imports ecCodes:
- `ECCODES_DEFINITION_PATH` = `eccodes_cosmo_resources.get_definitions_path()`;
- `ECCODES_VERSION_CHECK_OFF=1`;
- `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- `python -O` (= the notebook's `PYTHONOPTIMIZE=1`).

The driver differs from the notebook in three deliberate ways:
- It uses the config's default runner (`auto` → forecaster, one device), not `runner.parallel.cluster.manual=N`.
- `cache_dir` and `precision` are overrides.
- The network is replaced by the frozen cache (§4).

## 4. How the run works, and how to rerun it offline

`run_reference_realdata.py` builds a run directory that stands in for the HF repo root: symlinks
`varda-forecaster-det-sgm/` and `templates/`, so the config's relative paths resolve as in the repo. It then calls
`create_runner(RunConfiguration.load(<HF yaml>, overrides))` and `runner.execute()`. This is the same call as
`anemoi-inference run inference_opendata.yaml date=2026-10-22T15:00:00`, with `lead_time: 1d` (4 steps of 6 h)
and `write_initial_state: true`, both from the config.

**Offline replay** (the default mode):
- `oper_ecmwf_opendata._latest_published_run` returns the pinned 18z.
- `ecmwf.opendata.Client`:
  - The `.index` of each data file is served from the cache. The real `get_parts()` then selects the byte
    ranges, in request order (`preserve_request_order=True`, as earthkit sets it).
  - `_download` is redirected to `file://<frozen file>` with the same parts (multiurl's `PartFileDownloader`).
    So earthkit's `ecmwf-open-data` source gets the same bytes.
- `oper_kenda_opendata._stac_items` / `_collection_static_assets` return the saved STAC documents. Every asset
  is already in `cache_dir`.
- `socket.connect` raises, so any other network access fails.

`--online` keeps only the ECMWF pin and uses fresh caches.

**Rerun**:
```
PY=$VARDA_SCRATCH/ref-oper/.venv/bin/python; R=$VARDA_SCRATCH/realdata/20261022T15
# reference runs (GPU; job_reference.sh copies the code to $VARDA_SCRATCH/jobs/<NAME>/code-<jobid> first)
NAME=realdata-ref-gpu sbatch $VARDA_SLURM_GPU --time=03:00:00 \
  -o $VARDA_SCRATCH/jobs/realdata-ref-gpu/slurm-%j.out python/reference/realdata/job_reference.sh \
  "--device cuda --precision checkpoint --out $R/ref-fp16" "--device cuda --precision 32 --out $R/ref-fp32"
# replay check: stub model (zeros), CPU node, offline and online, then compare (expects bitwise)
NAME=realdata-stub-offline sbatch $VARDA_SLURM_CPU --time=01:00:00 ... \
  python/reference/realdata/job_reference.sh "--device cpu --stub-model --out $R/stub-offline"
$PY -I python/reference/realdata/compare_runs.py $R/stub-offline $R/stub-online --expect-equal
# post-processing (compute node)
$PY -I python/reference/realdata/export_stages.py $R/ref-fp16             # -> $R/ref-fp16/stages.safetensors
$PY -I python/export_regrid_matrix.py --run $R/ref-fp16 \
    --grib $R/cache/ecmwf/files/20261022120000-3h-oper-fc.grib2 --verify
$PY -I python/reference/realdata/analyse_grids.py $R/ref-fp16 $R/grids.json
$PY -I python/reference/realdata/make_manifest.py --date 2026-10-22T15    # frozen-cache manifest
```
A new date: `fetch_realdata.py pick`, then `kenda --date D` and `ecmwf --date D` (and `templates` once) on the
login node within ~24 h. The order matters: KENDA first, it expires first.

**Runs** (2026-10-23, one job each; the job's logs and exit codes go to `$VARDA_SCRATCH/jobs/<NAME>/`):

| name | what | result |
|---|---|---|
| realdata-ref-gpu | run 1: fp16 (checkpoint precision); run 2: fp32. GPU node, A100-SXM4-80GB, host CPU EPYC 7713 | exit 0, 0 |
| realdata-stub-offline | stub model, offline, CPU node (EPYC 7713) | exit 0 |
| realdata-stub-online | stub model, online (real downloads, ECMWF pin only), the same CPU node | exit 0 |

## 5. Runs

`$VARDA_SCRATCH/realdata/20261022T15/`:

| dir | precision | autocast | predict_step (s), steps 0..3 | execute (s) |
|---|---|---|---|---|
| `ref-fp16` | checkpoint (`16-mixed`) | float16 | 5.77, 1.72, 1.71, 1.71 | 244 |
| `ref-fp32` | `32` | float32 (autocast to fp32 = plain fp32) | 20.7, 9.69, 8.41, 8.41 | 363 |
| `stub-offline`, `stub-online` | stub (zeros) | – | – | 215, 355 |

The stub runs (`stub-offline`, `stub-online`, 34 GB) were deleted on 2026-10-23 once the replay check was recorded; its
result is `compare-stub-offline-vs-online.json` (kept), and the commands above regenerate the runs.

Settings shared by every run:
- The default `Runner` (forecaster), not the parallel runner.
- `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16`; `_NUM_CHUNKS_PROCESSOR` unset.
- The pickled attention backend (`triton`).
- TF32: matmul off, cudnn on (torch defaults, untouched).
- `python -O`.

Each run dir holds `outputs/` (the GRIB files), `dumps/`, `stages.safetensors` (the fp16 and fp32 runs only),
`manifest.json`, `config.yaml` (resolved config + overrides), and `cache/ecmwf/` (earthkit cache + MIR weights).

The fp16-vs-fp32 numbers (post-processing step, from `ref-fp16/dumps` and `ref-fp32/dumps`) are per variable, `rel_absmax = max|y16 − y32| /
max|y32|`. Worst per step: 3.7e-3 (q_925), 3.8e-3 (q_100), 4.1e-3 (q_925), 5.6e-3 (v_300). Median: 1.4e-3 to
1.8e-3.

## 6. Dumps

`<run>/dumps/*.safetensors` (readable with `python/safetensors_io.py` and the engine's `SafeTensorsFile`). Each
file has metadata `meta` (JSON: names, dates, NaN counts, provenance). `<run>/manifest.json["dumps"]` indexes
them all with shapes. Float data stay F32 as the runner holds them; coordinates are F64.

| dump | content |
|---|---|
| `source.prognostics.lam_0` | KENDA after `model_to_pressure_level`: `field/<var>` [2, 1147980] for 66 prognostics, dates (t−6h, t). Its `latitudes`/`longitudes` are the checkpoint's **full 1 688 650** arrays (see §8); the cutout then replaces them |
| `source.constant_forcings.lam_0` | KENDA constants after `geopotential_from_height`: `z lsm slor sdor` [2, 1147980] |
| `source.prognostics.global` | IFS after MIR and gh→z: 66 prognostics [2, 542080], N320 coordinates |
| `source.constant_forcings.global` | IFS constants (18z+0h, date overridden): `z lsm slor sdor` [2, 542080] |
| `regrid.<k>` (k = 0..9) | each MIR call: `out/<shortName>@<level>` F64 [542080] (decoded MIR output), `out_latitudes/out_longitudes`, metadata with the input and output message keys. `regrid.<k>.in.grib2` / `.out.grib2` hold the exact bytes MIR read and wrote. Calls: 0 = pl t−6h (60 fields: gh q t u v × 12 levels), 1 = sfc t−6h (6), 2/3 = same for t, 4..9 = constants (z, lsm, sdor+slor per date) |
| `cutout.prognostics`, `cutout.constant_forcings` | the merged state: `field/<var>` [2, G], `latitudes`, `longitudes`, `mask/lam_0`, `mask/global` (BOOL [G], the cutout's `_mask`) |
| `input_state` | the state given to `prepare_input_tensor`, after the initial computed forcings were added: 79 fields [2, G] |
| `input_tensor` | `prepare_input_tensor`'s result, [2, 79, G] (numpy layout of the runner) |
| `forcings.constant.0` | ComputedForcings `cos/sin_latitude/longitude` [4, 2, G], dates (t−6h, t) |
| `forcings.dynamic.<i>` | i = 0: `cos_julian_day cos_local_time insolation sin_julian_day sin_local_time` [5, 2, G] for the initial window. i = 1..3: [5, 1, G] for the window of step i (valid date of output i−1: 21Z, 03Z, 09Z) |
| `step<s>.window` | the tensor passed to `model.predict_step` at step s, [1, 2, G, 79] F32 |
| `step<s>.y` | its output, [1, 1, 1, G, 67] (dtype recorded: float32 in both runs) |
| `state<k>.pre` | the state before the top-level post-processors (k = 0: initial state, 70 fields, no tp; k = s+1: output of step s, 67 fields) |
| `state<k>.<i>.<Processor>` | fields that processor i changed: `0.Accumulate` (`accumulate_from_start_of_forecast`, tp; unchanged at k = 1), `1.ForwardTransformFilter` (`rescale` ×1000 on TOT_PREC = tp) |
| `grib.lam.<k>`, `grib.global.<k>` | the state each GRIB output encodes, after its own post-processors: lam [1147980] (`extract_from_state lam_0`); global [542080] (`extract_mask` inverse + `assign_mask` → NaN at the 1 410 masked points) |

**Stage file** `<run>/stages.safetensors` (`export_stages.py`, 233 tensors, 9.5 GB). It holds the same data under
the names the C++ checks use (`docs/spec-io.md` §5.2):

| name | shape, dtype | stage |
|---|---|---|
| `lam.<var>` | [2, 1147980] F32 | C1: KENDA after pre-processing, 66 prognostics + `z lsm slor sdor` |
| `lam.latitudes`, `lam.longitudes` | [1147980] F64 | the coordinates the cutout used (`source0/*`) |
| `global.<var>` | [2, 542080] F32 | C1: IFS after regrid (+ gh→z), before the mask, prognostics + constants |
| `global.latitudes`, `global.longitudes` | [542080] F64 | N320 (= `source1/*`) |
| `mask.lam_0`, `mask.global` | [G] U8 | the cutout's private masks (where each source sits in G) |
| `latitudes`, `longitudes` | [G] F64 | merged (= checkpoint `latitudes/longitudes`) |
| `x0` | [1, 2, G, 79] F32 | C2: the step-0 input of `predict_step` (= `in.x`) |
| `window.<s>` | [1, 2, G, 79] F32 | input of step s = 1..3 |
| `forcings.<s>` | [5, G] F32 | dynamic forcings inserted for step s = 1..3 (rows in `forcing_vars` order) |
| `y.<s>` | [G, 67] F32 | C3: model output of step s, before the post-processors (out_vars order) |
| `y.<s>.post` | [G, 67] F32 | after accumulate + rescale (only `tp` changes) |
| `init.<var>` | [G] F32 | C6: the initial (step-0) state after the post-processors, 70 fields |

Metadata `stages` (JSON) has:
- `in_vars`, `out_vars`, `forcing_vars`, `dates`, `forcing_dates`, `dates_out`;
- the checks: `x0_equals_input_tensor`, `in_vars_equal_engine_input_order`, `out_vars_equal_engine_output_order`,
  `x0_columns_equal_merged_sources` (70/70 bitwise, using `grid.{lam_0,global}.cutout_mask` of
  `converted/varda`), `y.<s>_state_equals_model_output`. All are true in both runs.

C4 and C5 data are `dumps/grib.{lam,global}.<k>` and `outputs/*.grib`.

## 7. Regridding (IFS 0.25° → N320)

**What runs.** `oper-ecmwf-opendata` → `anemoi.plugins.ecmwf.inference.opendata.retrieve` →
`MIRRegrid(grid="N320", area=[89.785, 0, -89.785, 359.719])` → `_mir_regrid_grib` → **mir-python 1.29.0**
(`mir.Job(grid, area, edition=2, packing="ccsds", accuracy=16, truncation="auto")`). This is the GRIB round trip:
each request's raw open-data messages in, a GRIB2 N320 file out. It is not earthkit-regrid. No interpolation key
is set, so MIR's default **`linear`** applies (finite-element triangulation of the source).

The weights are cached at `<run>/cache/ecmwf/mir/weights/18/linear/LL-0.25x0.25-90:180:-90:539.75/N320-89.7849:0:-89.7849:359.719-<hash>.mat`.
The run wrote two keys (`86a3addc…`, `88a91407…`) with identical bytes, sha256 `1e5fac3b…aed533`. The source
layout is the real files' −180..179.75: Ni 1440, Nj 721, first point 90°N / 180°E, last −90 / 179.75,
iScansNegatively 0, jScansPositively 0. The output is written as GRIB2 `grid_ccsds`, 16 bits, so the regridded
values the model sees are 16-bit quantised.

**Exported matrix** `$VARDA_SCRATCH/realdata/assets/regrid-ifs0p25-n320.safetensors` (22.4 MB, sha256
`ba2f0333…e83c29a`):
- tensors `regrid.indptr` I64 [542081], `regrid.indices` I32 [1 505 824] (MIR's stored column order, not
  sorted), `regrid.weights` F64 [1 505 824];
- metadata `regrid` (JSON): method, versions, n_source 1 038 240, n_target 542 080, row lengths (120 416 rows of
  2, 421 664 rows of 3), the source geometry read from a real message, the target grid and area, the MIR job, the
  `.mat` sha256, the verification;
- per-field verification: `regrid-ifs0p25-n320.verify.json`.

The tool reads the `.mat` raw: an eckit Stream with uint32 indices and float64 weights. It does not go through
scipy and does not sort.

**Verification** (`export_regrid_matrix.py --verify`, all 10 MIR calls of the fp16 run, 140 fields: pl and sfc
for both dates, plus the constants):
1. `y = W x`, with `x` = `codes_get_double_array` of the input message, each row summed in double in stored
   order (`s = 0; s += w·x`, separate multiply and add), against MIR on the same input bytes with 64-bit IEEE
   output (MIR's values before the encode): **0 differing points, max_abs 0** on all 140 fields.
2. `y` set (`codes_set_double_array`) into a clone of the run's output message (ccsds, 16 bits) and decoded,
   against the run's decoded MIR output: **bitwise on 140/140** (max_abs 0).
3. MIR rerun on the same bytes with the run's job, in a process with `ECCODES_DEFINITION_PATH` unset: its output
   bytes equal the run's on 10/10 calls. In the run, MIR shared an ecCodes context first initialised with the
   COSMO definitions, so those definitions change neither the values nor the bytes of MIR's output.
4. The `.mat` that fresh process built is byte-identical to the run's (`mir_cache_identical_to_run: true`).

**gh → z** is done after the regrid by `OrographyProcessor` (`docs/spec-io.md` §2.4). Surface `z` is requested
as `z` and is not touched.

## 8. Grids and cutout

The numbers are from `20261022T15/grids.json` (`analyse_grids.py`).
- **lam_0 = ICON-CH1 native cells, 1:1.**
  - The KENDA fields have 1 147 980 values, the ICON-CH1 unstructured grid.
  - The checkpoint's `source0/latitudes/longitudes` have the same count and order as CLAT/CLON of
    `horizontal_constants_icon-ch1-eps.grib2`. They differ by at most 1.22e-4° in latitude and 2.44e-4° in
    longitude, which is the GRIB packing of CLAT/CLON.
  - The KENDA messages carry no coordinates. earthkit falls back to the checkpoint's full `latitudes` (1 688 650),
    the cutout logs "Mismatch between latitudes and field points for source lam_0" and loads `source0/*`. The C++
    takes lam coordinates from the model file, never from the GRIB.
- **`lam_0/cutout_mask`**: 1 147 980 entries, all True, so every KENDA cell is kept, in GRIB order, as G[0 : 1 147 980].
- **`global/cutout_mask`**: 542 080 entries, 540 670 True. The 1 410 False points are N320 indices
  62 219…90 879, at 42.01–50.73°N, covering the LAM.
  - The kept points are appended after lam_0 in ascending N320 order, so G = 1 147 980 + 540 670 = 1 688 650.
  - The merged coordinates equal the checkpoint's `latitudes/longitudes` bitwise.
  - MIR's output coordinates equal `source1/*` bitwise.
- The cutout's private `_mask[lam_0]` is the leading 1 147 980 positions and `_mask[global]` the rest
  (disjoint, together all of G).
- Outputs: the lam file encodes G[0 : 1 147 980]; the global file encodes the full N320 (542 080) with NaN at
  the 1 410 masked points.

## 9. GRIB outputs

`outputs/` of each run (sha256 in `manifest.json["grib_outputs"]`):
```
202610221500_0.grib  _6  _12  _18  _24           lam_0, ICON templates, COSMO shortNames
global-202610221500_0.grib  _6  _12  _18  _24    global, IFS templates, ECMWF shortNames (patched param ids)
```
- The names are **unpadded** (`_6`, not `_006`), although the config says `{step:03}`.
- Step 0 has 70 messages. Their order is the prognostics in retrieval order (`z_*`, msl, sp, `q_*`, 2d, `t_*`,
  2t, `u_*`, 10u, `v_*`, 10v), then the constants `z lsm slor sdor`. There is no tp.
- Steps 6 to 24 h have 67 messages, in output-tensor order (10u 10v 2d 2t msl q… sp tp t… u… v… z…).
- Every message is `grid_ccsds` with **bitsPerValue 24**.
- lam: no bitmap. global: a bitmap on every message, 1 410 missing.
- Global quirks, from the IFS templates and patches:
  - every surface field gets `level 10`;
  - z, slor and sdor get `typeOfLevel heightAboveGround`, level 10;
  - 2t gets heightAboveGround 2;
  - lsm and sp get surface;
  - msl gets meanSea;
  - tp is paramId 228, `stepRange 0-6` … `0-24`.
- lam: tp is TOT_PREC (500041), accumulated from the start of the forecast, ×1000 (kg m-2).
- The fp16 and fp32 runs give identical step-0 files. All later files differ, as the model outputs differ.

## 10. Answers to the open questions of `docs/spec-io.md` §6

1. **CPU of the pre-processing.**
   - Both nodes were AMD EPYC 7713: a CPU node (stub runs) and a GPU node (fp16/fp32 runs).
   - numpy reports the baseline X86_V2 and dispatch targets X86_V3, X86_V4, AVX512_ICL and AVX512_SPR. Features
     on: AVX, AVX2, FMA3, F16C, BMI/BMI2, … X86_V3. No AVX-512, so the X86_V3 (AVX2+FMA3) float32 log path is
     taken.
   - The pre-processing outputs of the two nodes are bitwise equal. Recorded in `manifest.json["cpu"]`.
2. **MIR matrix of the run.**
   - sha256 `1e5fac3bf6e82d314b2c39add5962afb078b60903cdf3dd153baf19149aed533`, for the −180..179.75 source
     (cache dir `LL-0.25x0.25-90:180:-90:539.75`).
   - Exported and verified bitwise (§7).
   - The COSMO definitions in MIR's ecCodes context change neither MIR's output values nor its bytes (§7, item 3).
3. **Constants run L.** 2026-10-22T18 + 0h (`pins.json`, `constant_forcings_run`), file
   `20261022180000-0h-oper-fc.grib2`. Both slots of `global.{z,lsm,slor,sdor}` hold the same values.
   - The KENDA constants assets are republished every few hours with only `dataDate/dataTime` changed: 03Z in
     the frozen copy, 06Z in the online run's copy. The values are identical and the plugin overrides
     date/time/step, so the run is unaffected.
4. **NaN in inputs.**
   - None: 0 NaN in every source state, the regridded fields, the merged states and `x0`. The top-pressure clamp
     did not produce NaN on this date.
   - No step-0 lam message has a bitmap.
   - NaN appears only in `grib.global.*`, exactly at the 1 410 masked points, for every field.
5. **dtype of `y` under fp16 autocast.** float32: `step<s>.y` dtype `torch.float32` in both runs, and the state
   fields are float32.
6. **Step-0 order and file names.**
   - The order is as in §9: 66 prognostics in retrieval (`variables_metadata`) order, then `z lsm slor sdor`.
   - The names are unpadded: `202610221500_0.grib`, `_6`, `_12`, `_18`, `_24`, and `global-…` likewise.
7. **bitsPerValue 24.** Every output message has 24 bits per value (observed with the same library; not
   traced).
8. **Constants input with no hourly file.** For the KENDA constants, `variables_to_fetch` is empty and
   `_cached_fields(..., {})` builds `ekd.from_source("file", [])`.
   - It yields an empty fieldlist without error.
   - The STAC query for that valid time still ran (online run) or was replayed (offline), and succeeded.
   - Both runs completed with the constants taken from the collection assets.
9. **Runner and chunking.**
   - Runner: the default `Runner` (forecaster), single GPU, not `runner.parallel`.
   - `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16` (the notebook's value), `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR`
     unset.
   - Attention backend: triton, as pickled.
   - Precision: fp16 autocast (checkpoint `16-mixed`). The fp32 run sets `precision: "32"`.
   - Compare the C++ fp16 output against `ref-fp16` with the fp16 rollout tables. Mapper chunking changes the
     reference's arithmetic, so it is a perturbation of the same kind as those tables cover
     (`docs/rollout.md`).

## 11. What the C++ must reproduce that is not obvious

- **The 16-bit CCSDS quantisation after the regrid.** The model sees decoded 16-bit values, not `W x`. Apply
  the matrix in double, in stored order, then encode with `grid_ccsds`, 16 bits, through ecCodes and decode.
- **The matrix belongs to the source layout.**
  - Use the exported −180..179.75 matrix.
  - Refuse a message whose geometry differs from the one in its metadata.
  - Refuse a message with a bitmap. In the open-data files only the ocean fields (sve, svn, zos, sithick) have
    one; none is used.
- **lam coordinates come from the model file.** KENDA GRIB has none.
- **Constants.**
  - Global: from an explicit run (L), not from t.
  - KENDA: the static collection assets, with date/time/step overridden.
- **The ECMWF step-3 fallback.** At 03/09/15/21Z the global fields are 3-hour forecasts of the previous run, not
  analyses.
- **The global output.** The 1 410 masked points are NaN, so a bitmap is written. Every global message has
  `numberOfMissing` 1410; `missingValue` reads 9999 on decode. The NaN → −9999 encoding sequence is in
  `docs/spec-io.md` §4.3.
- **The output keys.**
  - Global surface fields carry level 10 and heightAboveGround for z/slor/sdor.
  - Files are named unpadded.
  - Messages are 24 bits per value.
  - The step-0 files hold 70 messages, without tp.
- **tp.** Accumulated from the start of the forecast. It is rescaled ×1000 only through its GRIB param
  `TOT_PREC` (the rescale's `param:` key), in both outputs.
