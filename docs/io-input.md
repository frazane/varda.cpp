# Real-data input: GRIB → x0

The input side of the real-data GRIB I/O. It builds the model input window `x0` from KENDA-CH1 and
ECMWF open-data GRIB files exactly as anemoi-inference does with the Hugging Face
`inference_opendata.yaml`. It also builds the 70 step-0 fields that the output side writes
(`write_initial_state`). The behaviour is specified in `docs/spec-io.md` §1-3 and §5, where each rule
cites its source. This file covers the API, the CLI and the verification.

**Result:** every stage is bitwise equal to the reference run on 2026-10-22T15: the per-source states,
the regridded fields, the MIR matrix, the merged state, `x0` and the step-0 state (§5).

## 1. Files

| file | content |
|---|---|
| `src/varda/io/numpy_log.{h,cpp}` | `logf_np`: numpy 2.5.4's float32 `np.log` (SIMD routine, AVX2/FMA3 dispatch), bit for bit (spec §2.5) |
| `src/varda/io/vertical.{h,cpp}` | `model_to_pressure_level`: top clamp, monotonicity check, log-p weights, interpolation, extrapolation (T, FI, lowest level), FI from HHL (spec §2.2) |
| `src/varda/io/regrid.{h,cpp}` | MIR-linear regrid: matrix file (`.safetensors` export or MIR's own `.mat`), geometry check, `y = W x` in stored order, 16-bit `grid_ccsds` re-encode and decode (spec §2.4) |
| `src/varda/io/input.{h,cpp}` | `build_input`: KENDA prognostics and constants, IFS prognostics and constants, naming, gh → z, HSURF → z, cutout merge, computed forcings, `x0`, step-0 fields |
| `apps/io/varda-prepare.cpp` | CLI: builds `x0` and writes it with per-stage dumps; `--compare` checks each stage against the reference dumps |
| `tests/io/test_io_numpy_log.cpp` | table and special values; `--stdin` runs the exhaustive check against numpy |
| `tests/io/test_io_vertical.cpp` | interpolation, extrapolation and FI against arrays produced by the plugin's own functions in the reference env (bitwise) |
| `tests/io/test_io_input.cpp` | dates, `FieldState`, both matrix formats, the stored-order sum; optional real-data check that MIR's input messages through our regridder decode bitwise like MIR's outputs |

Inputs at run time:
- the converted model with cutout masks (`docs/format.md` §1.1);
- `varda-io.json` (`python/export_io_config.py`, read through `io_config.h`), for the variables'
  patched mars entries, the KENDA namer rules and `initial_state_fields`;
- the regrid matrix `$VARDA_SCRATCH/realdata/assets/regrid-ifs0p25-n320.safetensors`
  (`python/export_regrid_matrix.py`);
- an N320 `grid_ccsds` template: the first message of the HF `templates/ifs-levtype=sfc.grib`;
- the COSMO definitions (`$VARDA_COSMO_DEFINITIONS`).

## 2. API (`varda::io`, `src/varda/io/input.h`)

```cpp
struct InputFiles {
    std::string date;                                   // t, "YYYY-MM-DDTHH" (or ISO-8601)
    std::string kenda_dir;                              // raw/<YYYYmmddTHHMMSS>/<VAR>.grib2, constants/*.grib2
    std::vector<std::string> ifs_files_prev, ifs_files_t;   // open-data files covering t-6h and t
    std::vector<std::string> ifs_constant_files;        // the run the global constants come from (explicit, spec §1.5)
    std::string regrid_matrix, n320_template, io_config, cosmo_definitions;
    bool allow_nans = false;                            // default: a NaN in x0 stops with variable:count
    bool keep_stages = false;                           // keep the per-stage states (InputState::stages)
};
struct InputState {
    std::vector<float> x0;  int64_t T, G, V_in;         // (1, T, G, V_in) physical fp32, model-input order, slot 0 = t-6h
    DateTime date0;  std::vector<DateTime> dates;       // t; all slot dates, oldest first
    std::vector<std::string> initial_names;             // 70: io_config.initial_state_fields (66 prognostics, z lsm slor sdor)
    std::vector<std::vector<float>> initial_values;     // [70][G], date t, merged grid
    std::vector<std::string> log;  std::map<std::string, double> seconds;   // provenance, timings per stage
    InputStages stages;                                 // with keep_stages: source.*, cutout.*, regrid outputs
};
struct InputModel { Config config; std::vector<double> latitudes, longitudes; std::vector<Model::CutoutSource> cutout;
                    static InputModel from(const Model&); static InputModel load(const std::string& model_path); };
InputState build_input(const InputModel&, const InputFiles&);
InputState build_input(const Model&, const InputFiles&);
```

`InputModel::load` reads only the config, `grid.*` and the cutout masks from the converted file. No
weights are touched, so `varda-prepare` needs neither a backend nor a GPU.

Wiring to the rollout (`src/varda/rollout.h`):
```cpp
varda::io::InputState in = varda::io::build_input(*model, files);
varda::RolloutRequest q;  q.x0 = in.x0.data();  q.date0 = in.date0;  q.n_steps = 4;   // lead_time 1d / 6 h
varda::Rollout(*model).run(q);            // or DeviceRollout; in.initial_* go to the step-0 GRIB output
```

Errors are `std::runtime_error`s that name the file, field or key. The library throws on:
- a missing file, message, level or auxiliary;
- a duplicate message;
- a validity date-time other than the requested date (hourly KENDA files, IFS prognostics);
- a bitmap, or a wrong number of values;
- a pressure column that is not strictly increasing;
- a regrid source geometry that differs from the matrix's;
- the N320 template's coordinates differing from `grid.*` under the global mask;
- a variable produced by only one source;
- a model input set more than once or never;
- a NaN in `x0`, unless `allow_nans` is set.

## 3. What runs, in order

For each input date `d` in `[t-6h, t]` (spec §1.3, §2.0-2.2):
1. **KENDA prognostics.** Read `P T U V QV` (80 levels each), `T_2M PS PMSL TD_2M U_10M V_10M`, and
   `HSURF` and `HHL` from the constants. Fields are float-decoded (`codes_get_float_array`), and every
   hourly message must be valid at `d`.
2. Clamp the top level of P, check that P is monotonic, and compute the log-p weights for the 12 model
   levels, once per valid time and shared by all variables. 600 hPa is configured but never used, so it
   is not computed; each target level is independent of the others.
3. Interpolate and extrapolate `T U V QV`. FI comes from HHL and goes through the same steps.
4. Name the fields through `IoConfig::kenda_name` (e.g. `t_500`). Surface fields pass through as decoded.

KENDA constants (§1.4, §2.3):
- `z = f32(double-decoded HSURF × 9.80665)`;
- `FR_LAND`, `SSO_SIGMA` and `SSO_STDH` are float-decoded;
- the same values go into both slots.

IFS (§1.5, §2.4):
- **Message selection.** For each prognostic, the one message per date with the variable's ECMWF param
  (the name without `_<level>`, and `z` on pl becomes `gh`), the right levtype and level, and validity
  `d`. Constants are the step-0 messages of the named constants run.
- **Per field, in parallel:**
  1. check the source geometry against the matrix's recorded one (no bitmap allowed);
  2. decode `x` in double;
  3. compute `y = W x` in double, each row summed sequentially in MIR's stored order;
  4. clone the N320 template, set `decimalScaleFactor=0`, `bitsPerValue=16` and the values;
  5. decode with `codes_get_float_array`. For `gh`, the double decode × 9.80665 is cast to f32 once.

Cutout merge (§2.6):
- The lam_0 points are kept where `grid.lam_0.cutout_mask` is 1 (all of them), then the N320 points
  where `grid.global.cutout_mask` is 1, in ascending order.
- The N320 template's coordinates under the mask are checked bitwise against the global part of
  `grid.latitudes` and `grid.longitudes`.

`x0` (§3):
- Each model input is either a prognostic, a constant or one of the 9 computed forcings. The forcings
  are `compute_forcings(slot date, ..., grid.latitudes, grid.longitudes)`, i.e. on the file's
  coordinates.
- The NaN rule is then applied.
- The step-0 fields are the last slot of the merged states, in `initial_state_fields` order.

## 4. CLI: `varda-prepare`

```
varda-prepare --model M.safetensors --io-config varda-io.json --regrid-matrix R.safetensors \
              --n320-template $VARDA_SCRATCH/hf/templates/ifs-levtype=sfc.grib \
              ( --realdata $VARDA_SCRATCH/realdata/20261022T15        # pins.json + cache/{kenda,ecmwf/files}
              | --date T --kenda DIR --ifs-prev F... --ifs F... --ifs-constants F... ) \
              [--definitions D] [--allow-nans] [--threads N] [--out DIR] [--dump-stages] \
              [--compare <ref run>/dumps] [--mir-mat <ref run>/cache/ecmwf/mir/weights/.../N320-....mat]
```

Two input modes:
- **`--realdata`** takes the date, the KENDA cache, the IFS file of each input date and the constants
  run from `pins.json`.
- **Explicit files** (`--date`, `--kenda`, `--ifs-prev`, `--ifs`, `--ifs-constants`) work for any
  date.

What it writes (`--out DIR`):
- `x0.safetensors`: `x0` F32 `[1, 2, G, 79]`, plus metadata `meta` with `date`, `dates`, `in_vars`,
  `checkpoint_uuid` and `provenance` (every file and message used, timings).
- `initial_state.safetensors`: `field/<name>` F32 `[G]` for the 70 step-0 fields, metadata `names`
  and `date`.
- With `--dump-stages`, `DIR/dumps/` with the reference's dump names:
  - `source.{prognostics,constant_forcings}.{lam_0,global}` (`field/<name>` `[2, n]`; the global ones
    also have `latitudes` and `longitudes`);
  - `cutout.{prognostics,constant_forcings}` (`[2, G]`);
  - `input_tensor` (`x` `[2, 79, G]`, the reference's numpy layout);
  - `regrid.safetensors` (`out/<shortName>@<level>@<YYYYmmddHHMM>+<step>` F64: the decoded 16-bit N320
    values before gh → z).

What `--compare` prints:
- For each stage and variable: `max_abs`, the number of values that are not bitwise equal, and
  bitwise yes/no. The stages are C1 (per source, plus the global coordinates), the regrid outputs
  matched to the reference's `regrid.<k>` by field and date, C2 (`x0` against `input_tensor`, per
  column, plus coordinates against `grid.*`) and the step-0 state (`state0.pre`).
- **Regrid in isolation:** the reference's own MIR input messages (`regrid.<k>.in.grib2`) through our
  regridder, compared with MIR's output messages in f64 and f32, plus the encoding keys
  (`bitsPerValue`, `binaryScaleFactor`, `decimalScaleFactor`, `referenceValue`).
- **C1a**, with `--mir-mat`: the loaded matrix against MIR's cache file (row pointers, stored indices
  and weights, bitwise).

The exit status is 1 if anything differs. The program also prints an FNV-1a hash of `x0` and of the
step-0 fields, for determinism checks.

## 5. Verification (2026-10-23, AMD EPYC 7713, cpu-short)

The reference runs are by the reference agent, with dumps under
`$VARDA_SCRATCH/realdata/20261022T15/<run>/dumps`. All three pre-processed on EPYC 7713 with numpy
dispatch X86_V3 (AVX2, FMA3, no AVX-512):

| run | job | node | type |
|---|---|---|---|
| `ref-fp16` | 285290 | <node> | GPU node |
| `ref-fp32` | 285290 | <node> | GPU node |
| `stub-offline` | 285288 | <node> | cpu-short |

### `logf_np`, exhaustive (job 285295)

numpy 2.5.4 in the reference env streamed `np.log` over every positive finite float32
(`0x00000001 .. 0x7f7fffff`) into `test_io_numpy_log --stdin`.
- **0 of 2 139 095 039 values differ** (18 s).
- Special values follow numpy: NaN → NaN, x < 0 → −NaN, ±0 → −inf, inf → inf.

### Unit tests (ctest, login node)

`test_io_numpy_log`, `test_io_vertical` and `test_io_input` all pass (1.8 s).
- **`test_io_vertical`:** 6 synthetic columns, against the plugin's `interpolate_extrapolate` and
  `destagger_z` in the reference env. The columns hit every extrapolation branch (hsurf < 2000,
  2000-2500 and > 2500 m) and the top clamp. Bitwise for T (78 values), U and FI, and for FI from
  HHL (48 values).
- **`test_io_input` real-data part:** MIR's `regrid.1` input messages through our regridder decode
  bitwise like MIR's outputs.

### Real data, 2026-10-22T15 (jobs 285351, 285354)

`varda-prepare --realdata ... --compare <run>/dumps`. The numbers are identical for each of the three
reference runs:

| stage (spec §5.2) | arrays | values | max_abs | differ | |
|---|---|---|---|---|---|
| C1 `source.prognostics.lam_0` (vertical interp/extrap, FI, surface) | 66 | 151 533 360 | 0 | 0 | bitwise |
| C1 `source.constant_forcings.lam_0` (HSURF·g, FR_LAND, SSO_*) | 4 | 9 183 840 | 0 | 0 | bitwise |
| C1 `source.prognostics.global` (N320 before the mask, z_* after gh·g; + lat/lon) | 66 + 2 | 72 638 720 | 0 | 0 | bitwise |
| C1 `source.constant_forcings.global` (+ lat/lon) | 4 + 2 | 5 420 800 | 0 | 0 | bitwise |
| C1 regrid outputs (f64, matched to `regrid.<k>` by field and date) | 140 | 75 891 200 | 0 | 0 | bitwise |
| regrid in isolation (MIR's 140 input messages; f64 and f32) | 280 | 151 782 400 | 0 | 0 | bitwise; encoding keys 0 differ |
| C1a matrix vs MIR cache `.mat` (542 080 × 1 038 240, nnz 1 505 824) | | | | | identical |
| `cutout.prognostics` | 66 | 222 901 800 | 0 | 0 | bitwise |
| `cutout.constant_forcings` | 4 | 13 509 200 | 0 | 0 | bitwise |
| **C2 `x0` vs `input_tensor`, per column** (incl. the 9 computed forcings) | 79 | 266 806 700 | 0 | 0 | **bitwise** |
| coordinates of the merged state vs `grid.*` | 2 | 3 377 300 | 0 | 0 | bitwise |
| step-0 state (`state0.pre`, 70 fields, date t, order as `initial_state_fields`) | 70 | 118 205 500 | 0 | 0 | bitwise |

- **C1a** was run against `ref-fp16`'s `.mat`. Its sha256 is `1e5fac3b…aed533`, the same as both
  cache files of the run.
- **Thread count.** `x0` and the step-0 fields are bitwise equal at 1, 16 and 64 OpenMP threads
  (identical FNV-1a hashes, job 285354).

### Timing

Job 285354, cpu-short, 64 cores, warm file cache, `--out` without stage dumps:

| threads | build_input | of which KENDA / IFS / merge / x0 | wall incl. writing x0 (1.07 GB) | peak RSS |
|---|---|---|---|---|
| 64 | 4.99 s | 3.07 / 1.16 / 0.24 / 0.40 s | 8.3 s | 4.1 GB |
| 16 | 5.63 s | 3.15 / 1.35 / 0.24 / 0.77 s | 8.9 s | 3.9 GB |
| 1 | 12.7 s | 6.47 / 4.29 / 0.23 / 1.63 s | 16.1 s | 3.6 GB |

The first run on a cold file cache took 26.5 s at 64 threads, 24.3 s of it reading KENDA. The reference
run's whole `execute` takes 244 s (`ref-fp16` manifest).

## 6. Conditions and limits

- **CPU dependence.** `logf_np` is numpy's AVX2/FMA3 routine. A reference pre-processed on an
  AVX-512 host (`X86_V4`) or one without FMA3 would use another routine (spec §2.5, §5.1), and that
  case has not been verified. The reference's manifest records numpy's dispatch.
- **The regrid matrix depends on the source layout.** The real files are −180..179.75. The library
  refuses a message whose geometry differs from the one recorded in the matrix file, or that has a
  bitmap.
- **N320 encode template.** Any N320 `grid_ccsds` handle works: the decoded values depend only on
  the values, the bit count and the scale factors. The HF `ifs-levtype=sfc.grib` template reproduces
  MIR's quantisation for all 140 fields, with the same `bitsPerValue`, `binaryScaleFactor`,
  `decimalScaleFactor` and `referenceValue`.
- **Global constants** come from an explicit run (`--ifs-constants`). The reference takes the latest
  run at start time (18z+0h here, `pins.json`).
- **NaN.** The 2026-10-22 inputs give a NaN-free `x0`. A NaN stops the run, with the variable and
  point count, unless `--allow-nans` (the reference's `allow_nans: true`) is given.
