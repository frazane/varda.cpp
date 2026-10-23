# Spec: real-data I/O (GRIB in, GRIB out) for the Varda forecaster

This document specifies, op by op, what anemoi-inference does with the Hugging Face config
`varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml`. It covers the path from the
KENDA-CH1 and ECMWF open-data GRIB files to the model input window, and from the model outputs to the
two GRIB output files. The C++ (`libvarda` + ecCodes C API) implements this so that it can feed
`Rollout` (`docs/rollout-engine.md`) from real data and write the same GRIB files. The model maths is
not covered here (`docs/cpu.md`, `docs/format.md`), and neither is the window
update (`docs/rollout-engine.md` §1) or the dynamic forcings (`docs/forcings.md`).

Every rule below was read from the source; the file:line next to it is where. Behaviours marked
"(checked)" were confirmed with small experiments in the reference environment on the login node:
synthetic fields, the checkpoint's metadata read from the zip without unpickling, and the reference
agent's cached real input files for 2026-10-22T15, used read-only. No model was run.

## Sources

| component | version | where |
|---|---|---|
| anemoi-inference | `9505dd0926ec3c5dea9308813cc7859bfdad755a` (HF pin) | `$VARDA_SCRATCH/src/anemoi-inference` = installed copy (identical except `_version.py`) |
| anemoi-plugins-meteoswiss | `a1488e1b` (0.1.0) | installed |
| anemoi-plugins-ecmwf-inference / -transform | 0.7.0 / 0.2.0 | installed |
| anemoi-transform | 0.4.2 | installed |
| earthkit-data / earthkit-meteo | 0.19.2 / 1.0.0 | installed |
| ecCodes | eccodeslib 2.47.3.21 (python `eccodes` 2.47.0) | installed; the C++ links the same `libeccodes` |
| eccodes-cosmo-resources | 2.44.0.1 | `.venv/share/eccodes-cosmo-resources/definitions` |
| MIR | mir / mirlib 1.29.0.21 | installed (reference only; not a C++ dependency) |
| numpy | 2.5.4 | installed |
| HF repo | `MeteoSwiss/Varda-single-1.0`; templates at revision 89fc5a6b | `$VARDA_SCRATCH/hf/` (`varda-forecaster-det-sgm/`, `templates/`) |

Environment: `$VARDA_SCRATCH/ref-oper/.venv`. Path abbreviations:

| abbreviation | path (under `$VARDA_SCRATCH/ref-oper/.venv/lib/python3.12/site-packages/`) |
|---|---|
| `SP/` | the site-packages directory itself |
| `I/` | `anemoi/inference/` |
| `MS/` | `anemoi_plugins_meteoswiss/` |
| `P/` | `anemoi/plugins/ecmwf/` |
| `AT/` | `anemoi/transform/` |
| `ED/` | `earthkit/data/` |
| `EM/` | `earthkit/meteo/vertical/xarray/interpolation.py` |
| `VI` | `MS/transform/filters/vertical_interpolation.py` |
| `GFH` | `MS/transform/filters/geopotential_from_height.py` |
| `NP` | numpy 2.5.4 `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` (fetched copy in the reference agent's scratch) |

Real data used for checks, read-only, at `$VARDA_SCRATCH/realdata/20261022T15/`:
- `cache/kenda/{raw,constants,stac}`;
- `cache/ecmwf/files`;
- `pins.json`: `date` 2026-10-22T15; input runs 06z+3h and 12z+3h; constants from run 18z+0h.

---

## 0. Overview

```
date t (config/CLI `date=`), input dates [t-6h, t]                       (I/runner.py:669; lagged [-6h, 0])

prognostics (66 vars)  ── cutout ─┬─ lam_0  : KENDA-CH1 analysis, ICON-CH1 grid (1 147 980 pts)
                                  │           model levels -> model_to_pressure_level (log p) + FI from HHL
                                  └─ global : IFS open data 0.25° (step 0 or 3 h) -> MIR linear -> N320
                                              -> gh*g -> z_*  -> keep the 540 670 points of global/cutout_mask
constants (z lsm slor sdor) ── cutout ─┬─ lam_0  : KENDA constants (HSURF*g -> z, FR_LAND, SSO_*)
                                       └─ global : IFS open data, latest run, step 0, same regrid/mask
computed forcings (9)  ── from the merged lat/lon at [t-6h, t]                      (docs/forcings.md)
        │
        ▼  state: 79 fields, each float32 (2, 1 688 650)   →   x0 (1, 2, G, 79) physical fp32
Rollout: 4 steps (lead_time 1d / 6 h)                                        (docs/rollout-engine.md)
        │  per step: y (G, 67) physical fp32, valid date t + 6h·(s+1)
        ▼
post-processors: tp accumulated from start (float32), tp ×1000
tee ─┬─ grib (ICON templates):   lam points 0..1 147 979                    outputs/<YYYYMMDDHHMM>_<step>.grib
     └─ grib (IFS templates):    points 1 147 980.. scattered back onto N320, NaN elsewhere, IFS paramIds
                                                                     outputs/global-<YYYYMMDDHHMM>_<step>.grib
step 0 (write_initial_state): the 66 prognostics + 4 constants at date t, as read (no tp)
```

**Sizes and index spaces** (checkpoint metadata, read with the runner's own `Checkpoint` class and the
`patch_metadata` applied):

| quantity | value | where it comes from |
|---|---|---|
| G | 1 688 650 data points | `lam_0` 1 147 980 (mask all True) + `global` 540 670 (of 542 080) |
| model inputs | 79 | `metadata_inference.data.data_indices.input` |
| model outputs | 67 | `...output` |
| retrieved prognostic variables | 66 | `I/variables.py:62-80`: include prognostic, exclude computed/diagnostic/forcing |
| retrieved constant forcings | 4: `z lsm slor sdor` | `I/variables.py:41-59` |
| retrieved dynamic forcings | 0, so the `empty` input | `I/variables.py:104-121`; `I/runner.py:744-748` |
| computed forcings | 9 | 4 constant in time (`cos/sin_latitude`, `cos/sin_longitude`), 5 dynamic (`docs/forcings.md` §1) |
| multi-step input | 2 | dates `[t-6h, t]` |
| step | 6 h | `timestep 6h`; `rollout_shift` = output offset = 6 h |
| precision | `16-mixed` | checkpoint; the HF config sets no `precision`, so the reference autocasts to fp16 (`I/runner.py:277-289`) |

Model input order (index: name), the column order of `x0`:

```
 0 10u   1 10v   2 2d   3 2t   4 cos_julian_day  5 cos_latitude  6 cos_local_time  7 cos_longitude  8 insolation
 9 lsm  10 msl  11..22 q_{100,1000,150,200,250,300,400,50,500,700,850,925}
23 sin_julian_day 24 sin_latitude 25 sin_local_time 26 sin_longitude 27 sp 28 sdor 29 slor
30..41 t_{same 12}  42..53 u_{...}  54..65 v_{...}  66 z  67..78 z_{...}
```

Model output order: `10u 10v 2d 2t msl q_*(12) sp tp(18) t_*(12) u_*(12) v_*(12) z_*(12)`. Levels follow
the same string order.

**`patch_metadata`** (`varda-forecaster-det-sgm/metadata.yaml`) is deep-merged into the checkpoint
metadata (`I/metadata.py:1391-1432`, applied at `I/checkpoint.py:187-196`).
- It replaces every `variables_metadata[v].mars.param` with its COSMO/KENDA name (`T`, `QV`, `FI`, `PMSL`,
  `PS`, `T_2M`, `FR_LAND`, `FIS`, `SSO_SIGMA`, `SSO_STDH`, `TOT_PREC`, ...). The originals are the ECMWF
  names (`2t`, `tp`, `z`, ...).
- It adds 18 keys that are not model variables (`*_600`, `w_*`). The runner warns about them; they have
  no effect.
- Flags are kept: `constant_in_time` (z, lsm, slor, sdor), `computed_forcing`, and tp's
  `process: accumulation`, `period: [6h, 12h]`.
- Every request and every output GRIB key below follows from these patched entries.

---

## 1. Inputs

### 1.1 Which input serves which variables

`Runner.execute` (`I/runner.py:639-719`) runs the following for the single dataset `data`, with
`dates = [t + h for h in lagged]` = `[t-6h, t]` (`:669`):

| category | variables (`I/runner.py:728-763`) | config section (`I/config/utils.py:18-39`) | input |
|---|---|---|---|
| prognostics | 66: `10u 10v 2d 2t msl sp` + `q t u v z` × 12 levels | `input` | `cutout`: lam_0 `oper_kenda_opendata` with `model_to_pressure_level`; global `oper_ecmwf_opendata` with `allow_forecast_fallback` |
| constant forcings | `z lsm slor sdor` | `constant_forcings` | `cutout`: lam_0 `oper_kenda_opendata` (HSURF, `geopotential_from_height`); global `oper_ecmwf_opendata` with `constant: true` |
| dynamic forcings | none | (would be `input`) | `empty` |

- Names are looked up in the registries with `_` replaced by `-` (`SP/anemoi/utils/registry.py:162,209`).
  The plugins register `oper-kenda-opendata`, `oper-ecmwf-opendata`, `model-to-pressure-level` and
  `geopotential-from-height` through entry points (`SP/anemoi_plugins_meteoswiss-0.1.0.dist-info/entry_points.txt`).
- Each cutout sub-input gets the cutout's variable list and the dataset's metadata
  (`I/inputs/cutout.py:148-150`).
- The three states are combined by field name; overlapping names are an error (`I/runner.py:681-685, 800-866`).

### 1.2 ecCodes definitions (process-wide, before anything is decoded)

Importing `anemoi_plugins_meteoswiss` calls
`codes_set_definitions_path("<cosmo>/definitions:" + previous path)` (`MS/__init__.py:17-42`).
- In the reference env that is `.venv/share/eccodes-cosmo-resources/definitions:/MEMFS/definitions`.
- It applies to every decode and to every output encode.
- With the COSMO definitions, KENDA messages get the shortNames `T QV U V P PS T_2M TD_2M U_10M V_10M PMSL
  HHL HSURF FR_LAND SSO_SIGMA SSO_STDH`. With the default definitions they decode as `t q u v pres ...`,
  and both HHL and HSURF become `h`.
- IFS open-data messages still decode with ECMWF shortNames under this path (checked on the real files:
  `t gh q u v 2t 10u msl sp lsm z sdor slor`).
- `OperEcmwfOpenDataInput` pops the environment variable `ECCODES_DEFINITION_PATH` around its MIR call
  (`MS/inference/inputs/oper_ecmwf_opendata.py:60-68, 245-249`). The context path that was set above stays.

**C++ rule:** call `codes_context_set_definitions_path(NULL, "<cosmo>/definitions:<default>")` before
the first handle is created, with the same eccodes-cosmo-resources 2.44.0.1 files. The output keys
(§4.4) depend on it as well.

### 1.3 KENDA-CH1, prognostics (lam_0 of `input`)

The input is `OperKendaOpenDataInput(MarsInput)` (`MS/inference/inputs/oper_kenda_opendata.py:125-234`).
For **each** date `d` in `[t-6h, t]` (`:185`):

1. **Requests.** `metadata.mars_requests(variables=66 prognostics, dates=[d], use_grib_paramid=False,
   patch_request=...)` (`:186-191`; `I/metadata.py:795-905`; the filters' `patch_data_request` is the
   identity, `AT/transform.py:133-145`). It returns (checked):
   ```
   levtype=pl  levelist=[50,100,150,200,250,300,400,500,700,850,925,1000] param=[FI,QV,T,U,V]
   levtype=sfc param=[PMSL,PS,TD_2M,T_2M,U_10M,V_10M]
   ```
2. **Variables to provide** (`:194-203`):
   - `requested = {FI QV T U V PMSL PS TD_2M T_2M U_10M V_10M}`;
   - `provide = (requested ∪ extra{P, HSURF, HHL}) − skip{FI}`
     = `{HHL HSURF P PMSL PS QV T TD_2M T_2M U U_10M V V_10M}`;
   - constants (names found in the static assets, `:170-177`) = `HHL, HSURF`;
   - hourly = `P PMSL PS QV T TD_2M T_2M U U_10M V V_10M` (sorted).
3. **Files.**
   - Hourly: one STAC item per variable of collection `ch.meteoschweiz.ogd-analysis-kenda-ch1` at
     `datetime=d`, one asset each (`:40-63, 205-207`). Stored as `<cache>/kenda/raw/<YYYYmmddTHHMMSS>/<VAR>.grib2`
     (`:216-220`).
   - Constants: the collection assets of `ch.meteoschweiz.ogd-forecasting-icon-ch1` of type GRIB
     (`:71-83`), `horizontal_constants_icon-ch1-eps.grib2` and `vertical_constants_icon-ch1-eps.grib2`,
     downloaded once per input instance (`:155-168`).
   - The constants' `date`, `time` and `step` are overridden to `d` / `HHMM` / 0 (`:222-232`); values are
     untouched. The same constants therefore appear once per date.
4. **Content** (checked, 2026-10-22T15). All messages are GRIB2, `unstructured_grid`, 1 147 980 values,
   `grid_simple`, no bitmap, analysis `stepRange 0`:

| file | messages | typeOfLevel | bitsPerValue |
|---|---|---|---|
| `T`, `P` | 80 each | generalVerticalLayer, level 1 (top) .. 80 | 24 |
| `QV`, `U`, `V` | 80 each | generalVerticalLayer | 16 |
| `PS` | 1 | surface | 16 |
| `PMSL` | 1 | meanSea | 16 |
| `T_2M`, `TD_2M` | 1 each | heightAboveGround 2 | 16 |
| `U_10M`, `V_10M` | 1 each | heightAboveGround 10 | 16 |
| HHL (vertical constants) | 81 | generalVertical, level 1 = 22 000 m .. 81 = HSURF | 0 for top levels |
| HSURF (horizontal constants) | 1 of 17 | surface | 16 |

Message selection without COSMO definitions, should it ever be needed (definition-independent keys):

| name | discipline / category / number | surface |
|---|---|---|
| T | 0/0/0 | `typeOfFirstFixedSurface` 150 |
| U, V | 0/2/2, 0/2/3 | 150 |
| QV | 0/1/0 | 150 |
| P | 0/3/0 | 150 |
| PS | 0/3/0 | surface (1) |
| T_2M | 0/0/0 | 103, level 2 |
| HHL | 0/3/6 | 150 with no second surface, 81 levels |
| HSURF | 0/3/6 | surface (1) |

The plain way is to set the definitions path (§1.2) and select by `shortName`.

### 1.4 KENDA-CH1, constants (lam_0 of `constant_forcings`)

The requests for `z lsm slor sdor` are, with param names from the patched mars entries:
`FIS` (step 0), `FR_LAND` (step 0), and `[SSO_SIGMA, SSO_STDH]` (step 12).
- `provide = ({FIS FR_LAND SSO_SIGMA SSO_STDH} ∪ {HSURF}) − {FIS}`. All four are static assets, so no
  hourly file is read.
- The STAC item listing of `d` is still queried (`:207`).
- Fields: `FR_LAND`, `SSO_SIGMA`, `SSO_STDH`, `HSURF` from `horizontal_constants_icon-ch1-eps.grib2`, with the
  date overrides above.
- Then `geopotential_from_height` turns HSURF into FIS (§2.3).

### 1.5 IFS open data (global of `input` and of `constant_forcings`)

`OperEcmwfOpenDataInput(OpenDataInputPlugin(MarsInput))` (`MS/inference/inputs/oper_ecmwf_opendata.py:155-263`,
`P/inference/opendata/opendata.py:294-367`).

**Grid and area.**
- `metadata.grid = "N320"`, `metadata.area = [89.785, 0.0, -89.785, 359.719]` (`I/metadata.py:651-658`).
- They come from the legacy data request: the first of the dataset's `data_request` entries with a truthy
  grid (`I/legacy.py:187-219`). Checked.

**Run and step for a target date `d`** (`MS/...oper_ecmwf_opendata.py:180-208`):
- `L` = latest published run (`Client(source="google").latest(type="fc")`, HEAD probes, `:90-98, 212`).
- `n_back = ceil(max(L-d, 0) / 6h)`, `init = L - 6h·n_back`, `step = d - init`.
- For any `d` within the 12 stored runs and not after `L`, this gives `init = floor_6h(d)`, with
  `step = 0` at 00/06/12/18 UTC and `step = 3 h` at 03/09/15/21 UTC. The 3 h case needs
  `allow_forecast_fallback: true`, which the config sets.
- It raises an error if the date is off the hour, older than 12 runs, newer than `L`, the step is not a
  multiple of 3 h, or the step exceeds 144 h.
- Example (`pins.json`): t = 15 UTC uses 12z+3h; t−6h = 09 UTC uses 06z+3h.

**Requests** (`:233-241`):
- `mars_requests(dates=[init], type="fc")`, then `step = lead` and drop the `None` levelist (`:115-126`).
- COSMO param → variable name minus `_<level>` (`:101-112, 129-138`).
- `z` → `gh` on `pl` (`P/inference/opendata/geopotential_height.py:26-41`, through `patch_data_request`).
- `mappings.yaml` only remaps soil parameters (`opendata.py:98-123`).

Resulting requests:
```
prognostics, per date: pl  levelist=[50..1000 (12)] param=[q,t,u,v,gh]   step=<0|3>
                       sfc param=[10u,10v,2d,2t,msl,sp]                   step=<0|3>
constants,   per date: sfc param=z ; param=lsm ; param=[sdor,slor]        init=L, step=0   (constant: true, :222-223)
```

**Files.**
- `https://storage.googleapis.com/ecmwf-open-data/{yyyymmdd}/{HH}z/ifs/0p25/oper/{yyyymmddHHMMSS}-{step}h-oper-fc.grib2`
  (`SP/ecmwf/opendata/client.py:35-38`; stream `scda` for 06/18 runs only before 2026-05-12, `:560-600`).
- Messages are byte ranges selected through the `.index`; the raw bytes go to MIR (`ED/encoders/grib.py:411-412`).
- In the files (checked, 2026-10-22):
  - the pressure levels come as 14 levels each of `t q u v gh z ...`;
  - surface `z`, `sdor` and `slor` exist only in the step-0 file;
  - packing is `grid_ccsds`, e.g. `t` 500 hPa has 9 bits;
  - no bitmap;
  - **source grid: `Ni=1440 Nj=721`, first point lat 90 / lon 180, last lon 179.75, `iScansNegatively=0`,
    `jScansPositively=0`**, i.e. the −180..179.75 layout (it matters for the regrid matrix, §2.4).

**Constants are not reproducible by design.** With `constant: true` the run is whichever is latest when
the job starts (`init = L`), whatever `t` is. The date is then overridden to `d` (`:251-259`), so both
slots hold the same values. The C++ takes the constant fields from an explicit file and records which run
it was (the reference: 18z+0h).

**After retrieve.**
- `_rename_params` (`opendata.py:208-231`) and `_cosmo_to_ecmwf_field_param` (`MS/...:141-152`) change
  nothing here: the fields already have ECMWF names (checked).

### 1.6 Naming: GRIB message → anemoi variable

The name is computed **after** the pre-processors (§2.0) by `f.clone(name=namer)`, then
`sel(name=variables)` and `order_by(name=variables, valid_datetime="ascending")`
(`I/inputs/ekd.py:200-224`). Fields whose name is not one of the input's variables are dropped silently:
`P`, `HHL`, `*_600`, `HSURF` in the prognostic input.

- **KENDA**: `RulesNamer`, first matching rule wins (`I/inputs/ekd.py:69-130`). A rule matches when
  `original_metadata.get(k) == v` for all its keys; `{level}` is the field's `level` key.

  | GRIB `shortName` | anemoi name |
  |---|---|
  | `SKT`, `T_G` | `skt` (unused) |
  | `T` | `t_{level}` |
  | `U` | `u_{level}` |
  | `V` | `v_{level}` |
  | `OMEGA` | `w_{level}` (unused) |
  | `QV` | `q_{level}` |
  | `FI` | `z_{level}` |
  | `PMSL` | `msl` |
  | `FIS` | `z` |
  | `PS` | `sp` |
  | `T_2M` | `2t` |
  | `TD_2M` | `2d` |
  | `U_10M` | `10u` |
  | `V_10M` | `10v` |
  | `FR_LAND` | `lsm` |
  | `TOT_PREC` | `tp` (unused on input) |
  | `SSO_SIGMA` | `slor` |
  | `SSO_STDH` | `sdor` |

  The pressure-level fields come out of §2.2 with `level` in hPa (`500`), so they are named `t_500`.
  Anything else falls back to the default namer.
- **IFS**: `rules = [({levtype: sol}, "{param}")]` (`opendata.py:320-325`, it overrides any configured
  namer), then the default namer (`I/metadata.py:587-636`).
  - The default namer looks up `(param, levelist)` among the patched variables (COSMO params), which
    never matches an ECMWF field.
  - So the fallback applies: `param` for `levtype` sfc (levelist forced to None, `:623-625`), else
    `f"{param}_{levelist}"`. That gives `t_500`, `z_500` (after gh → z), `2t`, `msl`, `z`, `lsm`, `sdor`,
    `slor`.

### 1.7 Coordinates

- **lam_0**:
  1. `fields[0].grid_points()` fails on the KENDA `unstructured_grid` (checked: `KeyValueNotFoundError`).
  2. `_create_state` then falls back to `metadata.latitudes`, the full 1 688 650-point supporting array
     (`I/inputs/ekd.py:291-316`).
  3. The cutout sees the size mismatch and loads `source0/latitudes` / `source0/longitudes`, which are
     index-based on the source order (`I/inputs/cutout.py:189-219`).
- **global**: `grid_points()` of the N320 field. Bitwise equal to `source1/latitudes` /
  `source1/longitudes` (checked, ecCodes N320 geometry).
- **Merged**: `concat(source0, source1[global_mask])` is bitwise the checkpoint's `latitudes` /
  `longitudes` (checked), i.e. the converted file's `grid.latitudes` / `grid.longitudes`. So the computed
  forcings of the real-data run are the engine's `compute_forcings` on the file's coordinates, bitwise
  (`docs/forcings.md` §5).
- lam longitudes are in [−0.817, 17.711]; global longitudes are in [0, 359.719].

---

## 2. Pre-processing

### 2.0 Where it runs, and decoding

Order inside each sub-input, `EkdInput._create_state` (`I/inputs/ekd.py:246-394`):
1. Time filter: `sel(valid_datetime in dates)`, both dates in one fieldlist (`:320-323`).
2. Pre-processors (`:326`): inner ones from the input config, then top-level ones (none)
   (`I/input.py:75-106`). `ForwardTransformFilter.process` calls `filter.forward(fields)`
   (`I/pre_processors/forward_transform_filter.py:68-82`). The global input also appends
   `OrographyProcessor` (`opendata.py:326`). Filters see raw GRIB shortNames.
3. Naming, selection and ordering (§1.6), then `check_data`: exactly `len(variables) × 2` fields
   (`I/checks.py:25-65`).
4. Values: `state[name] = np.full((2, n), NaN, float32)`, and row `date_idx` = `field.to_numpy(dtype=float32,
   flatten=True)` (`:336-371`). A missing or duplicate date is an error.
   - For a GRIB field this is `codes_get_float_array` (`ED/readers/grib/codes.py:53-63`). Before that,
     `missingValue = FLT_MAX` is set, and values equal to it become NaN if `bitmapPresent`
     (`codes.py:223-229`; `ED/utils/message.py:187`). No input here has a bitmap.
   - For a field produced by a filter it is `astype(float32)` of its array.
   - `codes_get_float_array` equals `float32(codes_get_double_array)` on the fields tried (checked), but
     the C++ uses exactly the call the reference uses.

**C++ rule (decode):** `codes_get_float_array` for every field read directly; `codes_get_double_array`
only for the two products computed in double (HSURF → z, §2.3; gh → z, §2.4).

### 2.1 Required auxiliaries of `model_to_pressure_level`

- `T_2M`, `PS`, `P`, `HSURF`, and `HHL` (for `add_geopotential`) must be in **every** valid-time group,
  otherwise the filter falls back to FDB (`VI:22-28, 72-106`).
- They are there by construction of §1.3. The C++ fails on a missing auxiliary instead.

### 2.2 `model_to_pressure_level` (`VI:108-163`), op by op

Config:
- `IL = [50,100,150,200,250,300,400,500,600,700,850,925,1000]` hPa;
- `EL = [500,600,700,850,925,1000]`;
- `add_geopotential: true`.

Conventions:
- Arrays are `[k = 0..79 (level 1 = top .. 80), n = 0..1 147 979]`; `to_xarray` sorts levels ascending
  (checked).
- `f32`/`f64` are IEEE rounding to float/double.
- Every elementwise operation is rounded on its own: **no FMA contraction** (`-ffp-contract=off`).
- `logf_np` is numpy's float32 log (§2.5); `log` and `powf` are glibc's.

**Bitwise verified.** This recipe was checked on 3 697 real columns of 2026-10-22T15 for T, U, QV and FI
at all 13 levels: 0 values differ.

For each valid-time group (`group_by("valid_datetime")`, `VI:111`):

0. **Read** (`VI:112-115`): `t2m`, `ps`, `p[80]`, `hsurf`, all float-decoded.
1. **Top clamp** (`VI:117`): `p[0][n] = p[0][n] < 5000 ? p[0][n] : f32(5000 - 1e-5)`. The constant is
   exactly 5000.0f. On 2026-10-22, `p[0]` was 4740–4862 Pa, so the clamp was a no-op.
2. **Per shortName** (`VI:119-146`):
   - `P` and `HHL` are dropped;
   - any field whose `typeOfLevel != generalVerticalLayer` passes through unchanged (T_2M, PS, HSURF, ...);
   - `W` would be destaggered (absent);
   - `T`, `U`, `V`, `QV` → `interpolate_extrapolate` (step 4).
3. **Geopotential on full levels** (`VI:148-161, 166-170`; `destagger_z` `VI:430-445`), from float-decoded
   HHL sorted by level:
   ```
   FI[k][n] = f32( f32( 0.5f * f32(HHL[k][n] + HHL[k+1][n]) ) * 9.80665f ),  k = 0..79
   ```
   - This is the mean of the half levels times g: no hydrostatic integration, no virtual temperature.
   - Metadata comes from HHL levels 1..80 with shortName FI and typeOfLevel generalVerticalLayer.
   - FI then goes through `interpolate_extrapolate(..., "FI")`.
4. **`interpolate_extrapolate`** (`VI:193-229`).

   (a) **Interpolation, linear in log p**: `EM:124-179` → `interpolate_monotonic` `EM:20-121`.
   - Target `t0_j = f64(IL_j * 100.0)` Pa (`EM:173`).
   - `p` must increase strictly with k at every point, otherwise a ValueError is raised (`EM:58-62`).
   - `k*` = smallest `k >= 1` with `p[k] >= t0 && p[k-1] <= t0` (`EM:85-93`). On an exact hit the lower
     bracket (upper k) is chosen.
   - If no `k` qualifies (target above the top full level or below the lowest full level), the value is
     **NaN**.

   Weights and value (`EM:106-119`):
   ```
   num = log(t0) - f64(logf_np(p[k*-1]))                         // glibc double log on the constant
   den = f64( f32( logf_np(p[k*]) - logf_np(p[k*-1]) ) )          // difference in float
   r   = num / den                                                // double
   out = f32( (1.0 - r) * f64(x[k*-1])  +  r * f64(x[k*]) )       // each * and + rounded in double
   ```

   (b) **Extrapolation** (`VI:217-228`), only for levels in `EL` and only where `out` is NaN
   (`interp.where(interp.notnull(), extrap)`), i.e. targets below the lowest full level.
   - Constants are Python floats, so they become f32 against f32 arrays (`VI:240-245`): `LAPSE 0.0065`,
     `H1 2000`, `H2 2500`, `T1 298`, `g 9.80665`, `r_d 287.05`.
   - `lg = logf_np( f32(p_t) / ps )` with `p_t = level*100` (an int, so the division is f32).
   - **T** (`VI:248-296`, helpers `VI:394-412`):
     ```
     tt  = t2m + 0.0065f*hsurf
     tmn = min(tt, 298.0f)
     tzp = hsurf > 2500.0f ? tmn : tmn*0.5f + tt*0.5f
     lr  = hsurf < 2000.0f ? 0.0065f : (1.0f/hsurf) * max(tzp - t2m, 0.0f)
     y   = ((lr*287.05f) / 9.80665f) * lg
     T   = t2m * (((1.0f + y) + (y*y)/2.0f) + powf(y, 3.0f)/6.0f)      // y**2 = y*y; y**3 = glibc powf
     ```
   - **FI** (`VI:299-347`):
     ```
     c  = f32((0.0065*287.05)/9.80665)        // computed in double, rounded once
     y  = c*lg
     FI = hsurf*9.80665f - ((287.05f*t2m)*lg) * ((1.0f + y/2.0f) + (y*y)/6.0f)
     ```
   - **U, V, QV** (`VI:350-387`): the value at the lowest model level (k = 79), copied.

   Observed on the 3 697 sample columns: extrapolation at 600 hPa 3 points, 700: 502, 850: 661, 925: 976,
   1000: 2 662; never at 500 or above. Levels 50–400 are never NaN in this domain. No NaN reached the
   output on 2026-10-22.
5. **Output fields** (`VI:182-190`):
   - Each target level becomes an `ArrayField` holding the f32 values **unpacked**, with no GRIB
     re-encoding.
   - Metadata: `typeOfLevel isobaricInhPa`, `level = levelist = int(Pa)/100` hPa, the input shortName.
   - The 600 hPa fields are produced and later dropped by the namer (§1.6).
6. **Return** `out + passthrough` (`VI:163`). The order is irrelevant downstream.

### 2.3 `geopotential_from_height` (constants of lam_0, `GFH:8-24`)

- `HSURF` → a new field with shortName/param `FIS` and values `field.values * 9.80665`; other fields
  pass through.
- `field.values` is a **double** decode, so:
  ```
  z_lam = f32( f64(HSURF via codes_get_double_array) * 9.80665 )
  ```
  - Checked bitwise; the float alternative differs at 287 947 of 1 147 980 points.
  - The new values reach the state through `NewMetadataField(NewDataField)` and `astype(float32)`
    (`AT/fields.py:146-185, 645-664`).
- The namer maps FIS → `z`. `FR_LAND`, `SSO_SIGMA` and `SSO_STDH` are float-decoded as read.

### 2.4 IFS: regrid to N320 (MIR), then gh → z

**Call.**
- `MIRRegrid(grid="N320", area=[89.785,0,-89.785,359.719])` with the defaults `packing ccsds`,
  `accuracy 16` (`P/transform/regrid/regrid.py:45-47`).
- One `mir.Job(grid, area, edition=2, packing="ccsds", accuracy=16, truncation="auto")` per request
  (`P/transform/regrid/backend.py:58-82, 183-231`).
- No interpolation method is set, so MIR 1.29.0's default is used: **`linear`** (finite-element
  triangles, spherical barycentric weights). The cache path is
  `mir/weights/18/linear/LL-0.25x0.25-.../N320-....mat`. No parameter class used here selects another
  method (`SP/mirlib/etc/mir/parameter-class.yaml`).
- `u` and `v` are interpolated as independent scalars.

**Output grid.** Classic reduced Gaussian N320, 542 080 points, rows north to south, each row from
longitude 0 eastwards. Bitwise the checkpoint's `source1/*` coordinates.

**Values** (checked on synthetic fields):
1. **`y = W x` in double.**
   - `x` is the double decode of the open-data message.
   - `W` is 542 080 × 1 038 240, with 2 or 3 non-negative weights per row.
   - **Summing each row sequentially in MIR's stored column order** reproduces MIR bitwise. A sorted
     order changes about 51 000 points by up to 6e-14.
2. **Re-encode** in GRIB2 `grid_ccsds`, 16 bits, decimal scale factor 0. An ecCodes handle with
   `bitsPerValue=16` and the same values decodes bitwise equal to MIR's output (checked).
3. **Decode** back (§2.0).

**The matrix is the only part that cannot be computed in C++ from source.** Re-deriving MIR's
triangulation bitwise is not realistic, and linking mirlib (plus atlas and eckit) is not allowed. So:
- A `python/` tool exports MIR's matrix once to a data file: raw CSR blobs, `indptr` u32, `indices` u32
  (stored order, **not sorted**), `data` f64, plus the source and target geometry and the MIR version.
  - Source of the matrix: MIR's own cache file (an eckit Stream: rows, cols, nnz, endianness, index and
    scalar size, then the blobs; `SP/mir/weight_matrix.py:69-85`), or
    `mir.weight_matrix.WeightMatrix.from_regrid` (`:153-189`), which gave a bitwise-identical matrix
    offline.
  - Do not round-trip through `_write_mat`: it sorts the indices (`:89-90`).
  - The reference's MIR cache lives under `cache/ecmwf/mir/` (`MIR_CACHE_PATH`, set and never restored,
    `MS/...oper_ecmwf_opendata.py:79`).
- **The matrix depends on the source grid layout.** The real files are −180..179.75 (§1.5), and its matrix
  differs from the 0..359.75 one (up to 0.25 on a random field). Only the 0..359.75 matrix has been
  extracted so far. Generate it from a real message's geometry and check its sha256 against the
  reference's cache file.
- The C++ refuses a message whose geometry (`Ni`, `Nj`, first and last lat/lon, scanning mode) differs
  from the recorded geometry, or that has a bitmap (MIR's missing-value handling is not `W x`).

**gh → z** (`OrographyProcessor`, `P/inference/opendata/geopotential_height.py:44-75` →
`AT/filters/fields/orog_to_z.py:36-38, 57-60`):
```
z_<level> = f32( f64(gh_regridded via codes_get_double_array) * 9.80665 )     // g: SP/earthkit/meteo/constants/constants.py:56
```
- Doing it in float32 differs at 217 472 of 542 080 points.
- Surface `z` is not touched: it is requested as `z` on `sfc`.

**C++ recipe, per field.** Select the message → check its geometry → `codes_get_double_array` →
`y = W x` (sequential, stored order, double) → set `y` into an N320 GRIB2 handle (`grid_ccsds`,
`bitsPerValue 16`, from the IFS sample used in the check) → `codes_get_float_array` (or the double decode
×9.80665 for `gh`).

### 2.5 numpy's float32 `log` (`logf_np`)

On the AMD EPYC 7713 nodes (AVX2+FMA3, no AVX-512; numpy dispatch X86_V3), `np.log` on float32 is numpy's
SIMD routine `simd_log_FLOAT` (`NP:526-700`, dispatch `NP:1280-1310`), not glibc `logf`.
- It agrees with glibc `logf` on only about 97% of pressure values.
- Scalar and strided calls use the same routine.

For a positive normal `x`:
```
e = f32( (int)(bits(x) >> 23) - 126 )
m = asfloat( (bits(x) & 0x7fffff) | (126 << 23) )                      // m in [0.5, 1)
if (m <= 0.707106781186547524400844362104849039f) { m = m + m; e = e - 1.0f; }
m = m - 1.0f
num = fmaf(fmaf(fmaf(fmaf(fmaf(P5,m,P4),m,P3),m,P2),m,P1),m,P0)
den = fmaf(fmaf(fmaf(fmaf(fmaf(Q5,m,Q4),m,Q3),m,Q2),m,Q1),m,Q0)
log = fmaf(e, 0.693147180559945309417232121458176568f, num/den)
```
- Special cases: NaN → NaN, x < 0 → −NaN, 0 → −inf, inf → inf (`NP:681-684`).
- Denormals are scaled by 2^100 first (`NP:122-150`); they do not occur here.
- The fused multiply-adds are real `std::fmaf`.

Coefficients (`numpy/_core/src/umath/npy_simd_data.h:112-123`, float literals):

| | value | | value |
|---|---|---|---|
| P0 | 0 | Q0 | 1 |
| P1 | 9.999999999999998702752e-01 | Q1 | 2.612677543073109236779e+00 |
| P2 | 2.112677543073053063722e+00 | Q2 | 2.453006071784736363091e+00 |
| P3 | 1.480000633576506585156e+00 | Q3 | 9.864942958519418960339e-01 |
| P4 | 3.808837741388407920751e-01 | Q4 | 1.546476374983906719538e-01 |
| P5 | 2.589979117907922693523e-02 | Q5 | 5.875095403124574342950e-03 |

- A re-implementation matched `np.log` on 6·10⁶ samples in [1e-3, 1e6] (0 ULP).
- **Test:** compare the C++ port against a numpy dump over all positive finite float32 values (about
  2·10⁹ inputs).
- Double `np.log` on these CPUs is glibc `log`; it is applied only to the 13 targets.

### 2.6 Cutout merge (`I/inputs/cutout.py:161-258`)

The sources are taken in config order: `lam_0`, then `global`.
- Each sub-state is masked on its last axis and appended (`:28-65, 236-241`):
  - lam_0: the mask `lam_0/cutout_mask` (all True), so all 1 147 980 points;
  - global: `global/cutout_mask`, keeping the 540 670 True entries in ascending N320 order.
  - The 1 410 dropped points are N320 indices 62 219 … 90 879, at 42.0–50.7°N.
- Latitudes and longitudes are merged the same way (§1.7).
- Both sources must produce exactly the same field names. A name missing in `global` raises; an extra name
  in `lam_0` keeps the wrong length and fails later on shape.
- Private attributes:
  - `_mask[src]` = a boolean of length G, True on that source's slice (`:221-253`);
  - `date` from the first source, i.e. `t` (`:243-244`).
- Masks are raw supporting arrays, one byte per bool, no header:
  `inference-last/anemoi-metadata/data/{lam_0,global}/cutout_mask.numpy`.

**C++ consequence (format change, via the integrator).** The converted file needs `grid.lam_0.cutout_mask`
and `grid.global.cutout_mask` (U8 or BOOL, from the checkpoint's supporting arrays). Both the input merge
and the global output (§4.3) use them. `source{0,1}/*` are not needed: they equal `grid.*` as shown in
§1.7.

---

## 3. State → model input window

`TensorHandler.prepare_input_tensor` (`I/tensors.py:119-174`), called from `Runner.run`
(`I/runner.py:232-236`), on the combined state (prognostics + constants):

1. `latitudes` / `longitudes` are those of the state (§1.7); `metadata.*` is used only if absent (`:121-128`).
2. **Initial forcings** (`:244-301`), at the dates `[t-6h, t]`:
   - The constant coupled forcings (`z lsm slor sdor`) are already in the state, so they are skipped
     (`:271-277`).
   - The computed constant forcings (`cos/sin_latitude`, `cos/sin_longitude`) and computed dynamic
     forcings (`cos/sin_julian_day`, `cos/sin_local_time`, `insolation`) are computed with
     `ComputedForcings.load_forcings_array(dates, state)` from the state coordinates.
   - Values are float32, computed in double and cast once (`docs/forcings.md` §2-4).
3. **Validation** (`:176-242`):
   - every field has shape `(2, G)`;
   - an infinity raises;
   - with `allow_nans: true`, NaNs are only reported (`:230-240`). There is no imputer, and
     `predict_step` never checks. **A NaN in any input point
     spreads through the graph:** see the NaN rule below.
4. **Tensor**: `np.full((2, 79, G), NaN, float32)`, then `tensor[:, idx(var), :] = field` for every state
   field (`:143-167`). Every one of the 79 inputs must be set, otherwise an error is raised (`:169-172`).
5. **Window**: `torch.from_numpy(np.swapaxes(tensor, -2, -1)[np.newaxis])`, i.e. `(1, 2, G, 79)` float32,
   slot 0 = t−6h (`I/runner.py:442-445`).

No dtype conversion happens after the per-field float32 cast of §2.0, and no unit change at all: the
window holds physical values, and normalisation happens inside `predict_step`.

**Mapping onto the engine** (`src/varda/rollout.h`):
```
x0[0, s, g, i] = field_{name(i)}[s][g]     s = 0 (t-6h), 1 (t);  g in merged order (lam, then global masked);
                                           i = model input index (config indices / variables)
q.x0 = x0;  q.date0 = t;  q.n_steps = ceil(lead_time / 6 h) = 4 for "1d"      (I/runner.py:396)
```
- The 9 computed-forcing columns of both slots are `compute_forcings(t-6h | t, names, grid.latitudes,
  grid.longitudes)`. This is bitwise anemoi's, because the merged coordinates equal `grid.*` (§1.7).
- Constants come from the constants source, the same values in both slots.
- **The rollout from there on is the existing `Rollout` / `DeviceRollout`.** The window update keeps the
  newest slot's constants (`docs/rollout-engine.md` §1), and the per-step forcings use the valid date
  `t + 6h·(s+1)`.
- The reference's dynamic forcings during the rollout use `new_states[...]["latitudes"]`, the input state's
  (`I/runner.py:449, 580-582`), i.e. the same coordinates.

**NaN rule for the C++.**
- The assembled `x0` must be NaN-free.
- With the 2026-10-22 data it is NaN-free by construction: the lam pressure levels are always bracketed or
  extrapolated (§2.2), and the global fields are complete.
- If a NaN appears, the reference would run and spread it. The C++ reports the variable and point count
  and stops, unless `--allow-nans` mirrors the reference.

**Precision.**
- The reference run autocasts to fp16 (`16-mixed`, §0). The HF notebook also shards with
  `runner.parallel.cluster.manual=<n_gpus>` and sets `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16`. Both can
  change the model's float arithmetic, not the I/O.
- The engine's CUDA fp16 path is the matching configuration; CPU fp32 remains the oracle.

---

## 4. Outputs

### 4.1 What the runner writes, and when (`I/runner.py:639-719`)

**Order of operations.**
1. Build the initial state (`:688-694`).
2. Apply the top-level post-processors (`:697-700`).
3. `open` and `write_initial_state` (`:705-709`).
4. Per step: post-processors, then `write_state` (`:711-716`).
5. `close` (`:718-719`).

**Initial state** (`write_initial_state: true`):
- `initial_state_categories` default `[prognostics, constant_forcings]` (`I/config/run.py:92`; `I/runner.py:868-898`).
- `Output.reduce` keeps the **last** date row, i.e. date `t` (`I/output.py:158-178`).
- Values are float32, as read and pre-processed: the input state, not the model input. There are
  **70 fields** and no tp or computed forcings:
  - the 66 prognostics, in the patched `variables_metadata` order: `z_100 z_1000 z_150 z_200 z_250 z_300
    z_400 z_50 z_500 z_700 z_850 z_925 msl sp q_* 2d t_100 t_1000 t_150 t_200 t_250 2t t_300 ... t_925
    u_100 u_1000 10u u_150 ... u_925 v_100 v_1000 10v v_150 ... v_925`;
  - then the constants `z lsm slor sdor` (`I/metadata.py:713-729`; `I/inputs/ekd.py:220-223`).
- No `step` key: outputs default it to 0 (`I/output.py:138`).

**Per-step state** (`I/runner.py:446-551`):
- The **67 model outputs** in output index order (`:523-527`), float32, physical and bounded: the engine's
  `y` (G, 67).
- `date = t + 6h·(s+1)`, `step = 6h·(s+1)` (`:516-520`).
- Yielded only while `step <= lead_time` (`:551`): steps 6, 12, 18, 24 h.
- The other state keys (`latitudes`, `_mask`, ...) are carried over from the input state (`:449`).

### 4.2 Top-level post-processors (initial state and every step, in config order)

1. **`accumulate_from_start_of_forecast: {accumulations: [tp]}`** (`I/post_processors/accumulate.py:63-89`):
   ```
   acc_0 = 0 (float32, zeros_like);  acc_s = f32( acc_{s-1} + max(0.0f, y_tp[s]) )   // allow_negative False
   ```
   - The field is replaced by the accumulator, and `start_steps["tp"] = 0h`.
   - In the initial state tp is absent, so nothing is accumulated and **no tp message is written at
     step 0**.
   - `grib-with-step-zero` (`MS/outputs/grib_with_step_zero.py:17`) is not used: the config says `grib`.
2. **`forward_transform_filter: rescale {scale: 1000, offset: 0, param: TOT_PREC}`**
   (`I/post_processors/transform_filter.py:92-110, 128-152`; `AT/filters/fields/rescale.py:19-25, 46-66`):
   - The state is wrapped into fields whose metadata is the variable's patched mars dict
     (`I/post_processors/earthkit_state.py:145-174`).
   - Fields with `param == TOT_PREC` (only tp) get `tp_out = f32(acc * 1000.0f)`, in a new array; the
     accumulator is untouched.
   - The state is unwrapped back to the same names and order (`:177-213`).
   - No effect on the initial state.

### 4.3 `tee` → two `grib` outputs

`TeeOutput` (`I/outputs/tee.py:68-91`) passes the same state to the lam output, then the global output.
Each is a `GribFileOutput` (`I/outputs/gribfile.py:117-433`, `I/outputs/grib.py:163-363`):
- apply its own post-processors;
- for each field in state order: find the template, build the keys, apply the modifiers, encode, write.

**Per-output post-processors.**
- **lam**: `extract_from_state: lam_0` (`I/post_processors/extract.py:164-219`) keeps `_mask["lam_0"]`,
  i.e. points `0 .. 1 147 979`.
- **global**:
  1. `extract_mask {mask: lam_0/cutout_mask, as_slice: true, inverse: true}` (`extract.py:100-121`) gives
     the slice `[1 147 980, G)`, 540 670 values.
  2. `assign_mask {mask: global/cutout_mask}` (`I/post_processors/assign.py:46-96`) gives a float32 array of
     542 080 NaN, with the values written at the True positions in ascending order: the inverse of the
     merge. That leaves 1 410 NaN points.

**Templates** (`samples: templates/templates_index_{icon,ifs}.yaml`; `I/grib/templates/manager.py:29-56,
122-140`, `samples.py:23-43`, `__init__.py:126-140`):
- The index is a list of `[match, file]` pairs; the first match on the lookup wins. The lookup is: name,
  grid, area, number of points, the variable's mars keys except step/date/time, and the encoding keys.
- The template is the **first message** of the file, which is resolved relative to the working directory.
- No match means an error.

| output | match | template file | first message |
|---|---|---|---|
| ICON (lam) | `levtype pl`, `param ∈ {T QV U V OMEGA FI}` | `icon-ch1-typeOfLevel=isobaricInhPa.grib` | |
| ICON (lam) | `sfc ∈ {T_2M TD_2M U_10M V_10M}` | `icon-ch1-typeOfLevel=heightAboveGround.grib` | |
| ICON (lam) | `sfc ∈ {FR_LAND PS FIS T_G SKT SSO_SIGMA SSO_STDH}` | `icon-ch1-typeOfLevel=surface.grib` | |
| ICON (lam) | `sfc PMSL` | `icon-ch1-typeOfLevel=meanSea.grib` | |
| ICON (lam) | `sfc TOT_PREC` | `icon-ch1-shortName=TOT_PREC.grib` | |
| IFS (global) | `levtype pl` | `ifs-levtype=pl.grib` (78 messages) | q at 50 hPa |
| IFS (global) | `levtype sfc` | `ifs-levtype=sfc.grib` (12 messages) | 10u at heightAboveGround 10 |

- The ICON templates are GRIB2, centre lssw, unstructured grid of 1 147 980 points, `grid_ccsds`
  bitsPerValue 0, PDT 1 (11 for TOT_PREC), step unit minutes.
- The IFS templates are GRIB2, centre ecmf, N320, `grid_ccsds`, PDT 0, local section class od / stream
  oper / type an.
- The files are in `$VARDA_SCRATCH/hf/templates/` (revision 89fc5a6b, `REVISION.json`).

**Keys per message, in this order** (`I/grib/encoding.py:117-196, 207-346`; then a stable sort that puts
`edition`, `typeOfLevel`, `stepType`, `productDefinitionTemplateNumber` first and drops `None`):

| kind | keys |
|---|---|
| lam pl | `edition=2, typeOfLevel=isobaricInhPa, stepType=instant, typeOfGeneratingProcess=2, shortName=<T|QV|U|V|FI>, dataType=fc, date=<YYYYMMDD of t>, time=<HH00 of t>, step=<H>, level=<hPa>` |
| lam sfc | `edition, stepType=instant, typeOfGeneratingProcess, shortName, dataType, date, time, step` |
| lam tp | `edition, stepType=accum, productDefinitionTemplateNumber=8, typeOfGeneratingProcess, shortName=TOT_PREC, dataType, date, time, startStep=0, endStep=<H>` |

- `date` / `time` are always the base time `t`, at step 0 too (`grib.py:246-283`).
- The param is the patched COSMO name, set as `shortName` because it is not numeric (`encoding.py:66-87`).
- **global**: the same keys, then the modifier `patches` (`I/outputs/grib.py:84-160, 348-349`).
  - `re.match` is tried against the anemoi name, anchored at the start only; the first match in dict
    order wins.
  - It replaces `shortName` in place and appends `param=<paramId>`: `10u 165, 10v 166, 2d 168, 2t 167,
    lsm 172, msl 151, sp 134, slor 163, sdor 160, tp 228, z 129, q_* 133, t_* 130, u_* 131, v_* 132,
    z_* 129`.
  - Only the last `patches` entry would be kept (`:115`, an upstream bug); this config has one.

**Encoding** (`I/grib/encoding.py:415-488, 543-593`; `I/outputs/gribfile.py:219-256`, missing value
`-9999` at `:143`):
1. Clone the template handle.
2. If any value is NaN: `nan_to_num` with NaN → −9999 (±inf → ±FLT_MAX), then append `missingValue=-9999`
   and `bitmapPresent=1`. This hits every global message (1 410 points), and lam messages only if an
   input-state field has NaN.
3. Append `generatingProcessIdentifier=255` if not set.
4. Set all keys at once (`SP/gribapi/gribapi.py:1752-1795`):
   - a `k=v,...` string with the type inferred from the text (integer → long, else double, else string);
   - applied with `grib_set_values`;
   - one by one if that fails (`encoding.py:472-480`).
   - **C++:** `codes_set_values` with a `codes_values[]` in the same order and types.
5. `codes_set_double_array(h, "values", f64(values))`. The float32 → float64 conversion is exact.
6. `codes_get_message`, appended to the file and flushed per message (`gribapi.py:545-557`).
7. `check_encoding` reads every key back; it is on by default (`encoding.py:349-412`).

**Files.**
- The path is rendered per message from the encoded handle (`encoding.py:595-621`;
  `I/utils/templating.py:16-44`): `dateTime` = `YYYYMMDDHHMM` of `t`, `step` = end step in hours.
- **`{step:03}` loses its format spec.** `format_dataset_name` maps unknown keys to the literal `"{key}"`
  (`I/decorators.py:270-279`, `I/outputs/gribfile.py:343-347`). So the names are
  `outputs/202610221500_0.grib`, `_6`, `_12`, `_18`, `_24`, and `outputs/global-…` likewise.
- Each path is opened once with `wb`.
- Counts: step 0 has 70 messages in initial-state order; other steps have 67 messages in output order, tp
  18th, `0-H`.

### 4.4 Resulting keys (measured, ecCodes 2.47.3, COSMO definitions first)

| field | lam (ICON template) | global (IFS template + patches) |
|---|---|---|
| t/q/u/v/z on pl | T/QV/U/V/FI (paramId 500014/500035/500028/500030/500006), isobaricInhPa L, PDT 1 | t/q/u/v/z (130/133/131/132/129), isobaricInhPa L, PDT 0 |
| 2t, 2d | T_2M, TD_2M, heightAboveGround 2 | 167, 168, heightAboveGround 2 |
| 10u, 10v | U_10M, V_10M, heightAboveGround 10 | 165, 166, heightAboveGround 10 |
| msl | PMSL, meanSea 0 | 151, meanSea, **level 10** (from the 10u template) |
| sp, lsm, tp | PS / FR_LAND / TOT_PREC, surface 0 | 134 / 172 / 228, surface, **level 10** |
| z sfc, sdor, slor (step 0 only) | FIS / SSO_STDH / SSO_SIGMA, surface 0 | 129 / 160 / 163, **heightAboveGround 10** |
| tp | PDT 8, accum, `0-H`, statistical processing 1 | the same; discipline/category/number 0/1/193 |

All messages:
- `dataDate`/`dataTime` = t;
- instantaneous `forecastTime = H` in hours (the ICON template's minutes are replaced);
- `typeOfGeneratingProcess 2`, `generatingProcessIdentifier 255`, `dataType fc`;
- **bitsPerValue 24** (the template's 0 becomes 24 when values are set; a constant field stays at 0);
- `grid_ccsds` (flags 14, block 32, rsi 128), decimal scale factor 0.

Global messages also have a bitmap with 1 410 missing values out of 542 080. The level/type artefacts and
the ensemble PDT 1 are what anemoi writes; the C++ reproduces them and does not fix them.

---

## 5. Bitwise vs tolerance, and verification checkpoints

### 5.1 What can be bitwise, and why

| stage | expectation | conditions / reason |
|---|---|---|
| KENDA decode | bitwise | same libeccodes 2.47.3, `codes_get_float_array` (§2.0) |
| `model_to_pressure_level` (interp, extrap, FI) | **bitwise** (verified on real columns) | `logf_np` port, glibc `log`/`powf`, no FMA contraction, the double interpolation step as written. **CPU-dependent:** numpy's log dispatch (AVX2 here; AVX-512 hosts unverified; no FMA3 → glibc `logf`) |
| HSURF → z, gh → z | bitwise | double decode × 9.80665, one f32 cast |
| MIR regrid | **bitwise only with MIR's exact matrix**, rows summed in stored order, then 16-bit CCSDS re-encode through ecCodes | the matrix must be MIR 1.29.0 `linear` for the real −180..179.75 source layout (sha256 against the reference's cache). Any other interpolation or summation order: ≈1 ulp, sometimes one 16-bit quantum (≈2e-3 K) after packing |
| cutout merge, coordinates | bitwise | index arithmetic; coordinates equal `grid.*` |
| computed forcings | bitwise | `docs/forcings.md` |
| assembled `x0` | bitwise | if every stage above is bitwise |
| model outputs per step | **tolerance** | `tolerances/rollout/varda.json` (the reference runs fp16 autocast, so use the fp16 tables) |
| tp accumulation, ×1000, masks, NaN fill, −9999 | bitwise given `y` | float32 elementwise, sequential |
| GRIB keys, key order, message order, file names | bitwise | same templates, definitions path, key sequence |
| GRIB message bytes | bitwise given the same float values | same library, template bytes, keys and `codes_set_values` sequence. Lengths depend on the values (CCSDS) |
| decoded output values vs reference | tolerance | model tolerance plus 24-bit packing: one quantum where inputs differ by less than a quantum (≈6e-8 relative) |
| constants (global) | reproducible only with a pinned run | `constant: true` uses the latest run at start time (§1.5) |

### 5.2 Checkpoints (to match the reference dump stages)

| # | stage | compared quantity | criterion |
|---|---|---|---|
| C1 | **per source after pre-processing** | lam_0: 66 prognostics, `(2, 1 147 980)` f32 each, plus constants `z lsm slor sdor`. global: before the mask, each regridded N320 field (542 080) and `z_*` after gh → z; after the mask, `(2, 540 670)` | bitwise; diagnostic only if not bitwise: `max_abs` at the extrapolated points only |
| C1a | regrid matrix | CSR blobs vs MIR's cache file | sha256 / raw equality |
| C2 | **assembled input** | the reference's `(1, 2, G, 79)` input tensor vs `x0` | bitwise, per column; plus coordinates vs `grid.*` |
| C3 | **per-step pre-output state** | 67 fields after the top-level post-processors (tp accumulated and ×1000) | `y`: rollout tolerances (`docs/rollout-engine.md` §4); tp post-processing bitwise given the reference's own `y` (feed the dumped `y` through the C++ post-processors) |
| C4 | per-output state | lam 1 147 980 points; global 542 080 with NaN exactly at the 1 410 masked points | bitwise given C3 |
| C5 | **GRIB files** | file list and names; per message: all keys (`grib_compare -H`-style key by key, including `md5Headers`), bitmap, then values | keys bitwise; values bitwise when the C++ writer is fed the reference's dumped states (encoder-only test, bytes too); end to end within the C3 tolerance plus one packing quantum |
| C6 | step-0 files | 70 messages from the input state | bitwise if C1 is bitwise (no model involved) |

---

## 6. Open questions

**Answered during this work** (real cached files of 2026-10-22T15):
- Source layout of the IFS 0.25° grid: −180..179.75 (first lon 180°), not 0..359.75. The matrix must be
  generated for it.
- Surface `z`, `sdor` and `slor` exist in the step-0 file of the 18z run; open-data messages keep ECMWF
  shortNames under the COSMO definitions path; no bitmaps in any input.

**Open** (the reference dumps or the reference agent's logs should settle them):
1. **CPU of the reference pre-processing.** numpy's float32 log dispatch depends on CPU features. Record
   `numpy._core._multiarray_umath.__cpu_features__` / `__cpu_dispatch__` in the dump. AVX-512 (GPU-node
   hosts?) may differ from the AVX2 path verified here.
2. **Exact MIR matrix of the run.** Compare the sha256 of `cache/ecmwf/mir/weights/18/linear/LL-…` with the
   C++ data file. Does MIR's context, encoding its output, see the COSMO definitions (the environment
   variable is popped, the context path is not)? This affects keys only, not values.
3. **Constants run `L`.** The reference used 18z+0h (`pins.json`). The C++ needs it as an explicit input.
4. **NaN in inputs.** Is any lam input field NaN at some point, e.g. the top-pressure clamp, which did not
   trigger on 2026-10-22? That would also give a step-0 lam message a bitmap and −9999.
5. **dtype of the reference `y` under fp16 autocast** at the output state: float32 is expected
   (`.cpu().numpy()` of the post-processed tensor). Confirm from the dump.
6. **Step-0 field order** (`variables_metadata` order, constants appended) and the **unpadded file names**
   (`_6.grib`): confirm from the reference `outputs/` listing.
7. **bitsPerValue 0 → 24** when values are set on the templates: observed, not traced in ecCodes. It
   reproduces with the same library.
8. **The constants input with no hourly file** (§1.4): `_cached_fields(..., {})` builds a file source with
   an empty list. Confirm that it yields an empty fieldlist and that the STAC query still succeeds; only
   availability is affected, not values.
9. **Parallel runner and mapper chunking** in the HF notebook (`runner.parallel`,
   `ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16`): these change the model's arithmetic, not the I/O. Record which
   settings the reference used, so the model-output comparison picks the right tolerance table.

**For the integrator** (shared files and dependencies this spec implies):
- New tensors in the converted file: the two cutout masks (§2.6). This changes `docs/format.md` §1.1 and
  `convert.py`.
- A new data artefact: the regrid matrix, written by a `python/` tool.
- Runtime data: the eccodes-cosmo-resources 2.44.0.1 definitions and the HF `templates/` directory. The only
  new library is libeccodes 2.47.3.
