# Dynamic forcings of the rollout

What anemoi-inference writes into the newest input slot after each autoregressive step, and how
`src/varda/forcings.{h,cpp}` reproduces it bit for bit (float32). Everything here was read from the sources
below; nothing is assumed.

Sources (all read-only):
- anemoi-inference `9505dd0926ec3c5dea9308813cc7859bfdad755a` (HF release pin; clone in
  `$VARDA_SCRATCH/src/anemoi-inference`, `I/` = `src/anemoi/inference/`);
- earthkit-data 0.20.0 (`ED/` = `earthkit/data/`), earthkit-meteo 0.6.2, anemoi-transform 0.1.26, anemoi-utils,
  numpy 2.3.5, all from `$IE_SCRATCH/venv-ref-varda/lib/python3.12/site-packages`.

## 1. Which forcings, when

- Both models (varda = Varda, dev) have the same 5 *dynamic* forcings, computed (not retrieved):
  `cos_julian_day, cos_local_time, insolation, sin_julian_day, sin_local_time`. They are the variables with
  `computed_forcing: true, constant_in_time: false` in the checkpoint's `anemoi-metadata/anemoi.json`
  (`dataset.data.variables_metadata`) and in `$VARDA_SCRATCH/hf/varda-forecaster-det-sgm/metadata.yaml`.
  Model input indices: varda `[4, 6, 8, 23, 25]` of 79, dev `[4, 6, 9, 26, 28]` of 84.
  The columns of the output buffer follow the caller's `names` list; the engine takes the order from the config.
- The other computed forcings (`cos/sin_latitude`, `cos/sin_longitude`) are `constant_in_time: true`; the runner never
  rewrites them (the roll carries them from the oldest slot, see `$IE_ROOT/docs/rollout.md` §1). The C++ function
  can compute them too (same code path in anemoi, `ED/sources/forcings.py:75-100`); they are not needed in the loop.
- Step `s` (0-based) of `Runner.forecast`: `valid_dates = start + s*rollout_shift + offset` for each
  `output_offsets` entry, `next_dates = valid_dates` (`I/runner.py:409-415`). For these models
  (one output step, `offset = timestep`, `rollout_shift = timestep`: 6 h / 1 h) that is
  `start + (s+1)*timestep`: the valid date of the prediction just made. After the update of the prognostic fields,
  `add_dynamic_forcings_to_input_tensor(…, next_dates, …)` (`I/runner.py:580`, `I/tensors.py:435-472`) calls
  `ComputedForcings.load_forcings_array(dates, state)`, swaps axes, and writes the result at the newest slot
  (`advance_map["outin"]`, `I/tensors.py:466`) for the forcing columns only.
  Engine rule: **after step `s`, forcings(valid date of the step-`s` output) go in the newest slot.**
  No update after the last step (`I/runner.py` "is_last_step").
- The initial window's two slots get their forcings from the same function at the two input dates
  (`add_initial_forcings_to_input_state`); the initial state is not this module's concern.

### 1.1 How the runner decides what is refreshed (and the converted `forcings` config section)

All of it is driven by the checkpoint metadata, never by names:
- **Categories** (`MultiDatasetMetadata.variable_categories`, `I/metadata.py:1598-1627`): `forcing` / `prognostic` /
  `diagnostic` from `metadata_inference.<dataset>.variable_types`; `constant` from `variables_metadata[v].constant_in_time`
  and `computed` from `variables_metadata[v].computed_forcing` (`anemoi.transform` `Variable.is_constant_in_time` /
  `is_computed_forcing`, default False). Selection by category: `select_variables` (`I/metadata.py:660-731`).
- **(a) Dynamic forcings refreshed each step**: `create_dynamic_forcings_providers` (`I/tensors.py:331-352`) asks for
  `include=["computed"], exclude=["constant"]` (computed and not constant in time) and uses their input indices
  `variable_to_input_tensor_index` = `metadata_inference.<dataset>.data_indices.input` (`I/metadata.py:1583-1584`); the
  values are written by `add_dynamic_forcings_to_input_tensor` (`I/tensors.py:435-472`). (The same function also builds
  *coupled* dynamic forcings for forcing variables that are neither computed nor constant, `:335-345`: none exist
  here and the converter refuses a checkpoint that has any.)
- **(b) Constant inputs inherited through the roll**: the runner marks as "reset"/already set every input with
  `typed_variables[v].is_constant_in_time` (`I/runner.py:463-466`); they are never rewritten, and the roll
  (`copy_prognostic_fields_to_input_tensor`, `I/tensors.py:392-432`, `advance_map["inin"]`) moves them from the oldest
  to the newest slot. They were loaded once at start (`create_constant_forcings_providers`, `I/tensors.py:302-329`).
- **(c) Prognostic inputs**: `metadata.prognostic_input_mask` = `data_indices.model.input.prognostic`
  (`I/metadata.py:375-377`), filled from the outputs (`:412-415`); in the converted config these are the inputs listed by
  `indices.residual_input`. The three sets partition the inputs (the converter checks it).
- **Timestep**: `MultiDatasetMetadata.timestep` = `metadata_inference.<dataset>.timesteps.timestep`
  (`I/metadata.py:1509-1511`; the legacy `config.data.timestep`, `:256-260`, is equal and checked). With one output step
  `output_offsets = [timestep]` and `rollout_shift = timestep` (`I/metadata.py:234-244`), so the new slot's valid date is
  `start + (s+1)*timestep`.

`python/convert.py` records this in the config as the top-level `forcings` section (every other config field is
unchanged):
```
"forcings": {"dynamic": [...], "dynamic_input_index": [...], "constant": [...], "constant_input_index": [...],
             "timestep_hours": N, "source": "<metadata fields used>"}
```
Names and indices are in model-input order; `timestep_hours` is inside this section.

## 2. Where the code is

`ComputedForcings.load_forcings_array` (`I/forcings.py:139-177`):
```
source = UnstructuredGridFieldList.from_values(latitudes=state["latitudes"], longitudes=state["longitudes"])
ds = ekd.from_source("forcings", source, date=dates, param=variables)        # ED/sources/forcings.py
...
forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(variables), len(dates), -1)
```
`to_numpy(dtype=float32)` is `ForcingField._values` = `proc(date).astype(np.float32)` (`ED/sources/forcings.py:260-264`):
**everything is computed in float64 and cast once, at the end, round-to-nearest.** The grid points are
`field.to_latlon(flatten=True)` of the unstructured grid: the arrays handed in, no reordering, no unit change.
Note `insolation` is *not* earthkit-meteo here: earthkit-data carries its own copy
(`ED/utils/meteo.py:10-101`, "copied from earthkit.meteo.solar.array"); the generator verifies that
`earthkit.meteo.solar.cos_solar_zenith_angle` gives bitwise the same float32 values (all dates, both grids).

## 3. Formulas (double precision, operation order as in the Python source)

Date: a naive `datetime` taken as UTC (`datetime.fromisoformat` / the runner's dates; no time zone anywhere).
`s_day = hour*3600 + minute*60 + second` (microseconds are zero). Leap years are handled by Python's calendar; the
engine's `add_hours` / `days_from_civil` are proleptic Gregorian (verified 1900, 2000, 2100, 2024).

| forcing | formula | source |
|---|---|---|
| `julian_day` | `d0 + s_day / 86400.0`, `d0` = days since 1 January of the same year (0-based: 1 Jan 00:00 -> 0.0; `delta.days + delta.seconds/86400.0`) | `ED/sources/forcings.py:142-150`, `ED/utils/meteo.py:23-29` |
| `cos_julian_day`, `sin_julian_day` | `a = julian_day / 365.25 * pi * 2` (left to right: `((jd/365.25)*pi)*2`), `cos(a)`, `sin(a)` | `forcings.py:152-158` |
| `local_time` | `hsm = (0 + s_day/86400.0) * 24`; `lt = (lon/360.0*24.0 + hsm) % 24` with Python/numpy floored modulo (longitudes may be negative: the grid has lon in [-0.82, 359.72]) | `forcings.py:160-169` |
| `cos_local_time`, `sin_local_time` | `r = lt / 24 * pi * 2`, `cos(r)`, `sin(r)` | `forcings.py:171-177` |
| `insolation` | `cos_solar_zenith_angle(date, lat, lon)` below | `forcings.py:179-204` |

`cos_solar_zenith_angle` (`ED/utils/meteo.py:32-101`), with `a` the same angle as `cos_julian_day`'s (`jd/365.25*pi*2`):
```
dec_deg = 0.396372 - 22.91327*cos(a) + 4.025430*sin(a) - 0.387205*cos(2a) + 0.051967*sin(2a) - 0.154527*cos(3a) + 0.084798*sin(3a)
tc      = 0.004297 + 0.107029*cos(a) - 1.837877*sin(a) - 0.837378*cos(2a) - 2.340475*sin(2a)        # time correction, h.deg
dec     = deg2rad(dec_deg);  latr = deg2rad(lat)
z       = sin(dec)*sin(latr) + (cos(dec)*cos(latr)) * cos(deg2rad((hour-12)*15 + lon + tc))
insolation = max(z, 0)                                                                              # np.clip(z, 0.0, None)
```
(sum evaluated left to right; `(hour-12)*15` is an int added to `lon` first, then `tc`; only `date.hour` is used,
minutes and seconds are ignored here. `deg2rad(x) = x * (pi/180.0)`, bitwise what numpy 2.3.5 does, checked on random
data.) Values: sin/cos of angles in [-1, 1]; insolation in [0, 1].

## 4. dtype at each stage

| stage | dtype |
|---|---|
| checkpoint supporting arrays `latitudes`, `longitudes` | float64 [G], degrees |
| ForcingMaker (all intermediates, `np.pi`, trig) | float64 |
| `ForcingField._values(dtype=float32)` | `astype(float32)`, once |
| `load_forcings_array` result `(vars, dates, G)` | float32 |
| input tensor `(1, T, G, V)` | float32 (the model's own normalisation happens later, inside `predict_step`) |

The forcings are inserted in **physical** units, like all inputs; the engine's normaliser then applies `x*mul+add`.

## 5. Coordinate source

anemoi-inference does not use the graph node attributes. `Checkpoint._raw_metadata = load_metadata(path,
supporting_arrays=True)` (`I/checkpoint.py:163`); `load_supporting_arrays` (`anemoi/utils/checkpoints.py:146-171`)
reads each entry of `anemoi.json["supporting_arrays_paths"]["data"]` with `np.frombuffer(zipf.read(path), dtype=...)
.reshape(shape)`. `TensorHandler.prepare_input_tensor` puts `metadata.latitudes` / `.longitudes` (= supporting arrays
`latitudes` / `longitudes`, `I/metadata.py:1273-1280`) into the input state unless the input already provides them
(`I/tensors.py:121-125`); `ComputedForcings` reads `current_state["latitudes"]` (the state of the previous step is
passed, `I/runner.py:571-578`). So the coordinates come from the zip members
`inference-last/anemoi-metadata/data/latitudes.numpy` and `.../longitudes.numpy` (raw little-endian float64 buffers
without header, no unpickling), degrees, as stored: varda G = 1 688 650, lat in [-89.7849, 89.7849], lon in
[-0.8171, 359.7188]; dev G = 40 320, lat in [-89.2842, 89.2842], lon in [0, 359.1] (cutout grid
`lam_0` + `global` for varda; `source{0,1}/…` and `*/cutout_mask` are other supporting arrays, unused here).
The reference runs of this project (`rollout/forcings/forcings.pt`) take the same arrays (bitwise agreement below).
If a user supplies other coordinates in the input state, anemoi would use those; the engine always uses the file's.

Converted file (format.md §1, extension): `grid.latitudes`, `grid.longitudes`, **F64 [G]**, degrees, exactly the
zip members (bitwise, tested). `python/convert.py` asserts they agree with `nodes.data.attr` =
[sin lat, sin lon, cos lat, cos lon] (float32 of the radians; measured max abs difference 2.4e-7 (varda),
2.5e-7 (dev); threshold 1e-6). All other tensors and the config are unchanged. Until the engine
consumed these two tensors the files were kept apart in `converted-new/`; the files under `converted/` now carry
them (`converted-new/` was a byte-identical copy and was deleted on 2026-10-23).

## 6. C++ API (`src/varda/forcings.h`)

```cpp
DateTime parse_iso8601_utc(const std::string&);        // "YYYY-MM-DDTHH:MM:SS" (also space, no seconds, trailing Z)
std::string format_iso8601_utc(const DateTime&);
DateTime add_hours(const DateTime&, int64_t hours);     // UTC, leap years, negative ok; no local time zone
void compute_forcings(const DateTime& valid, const std::vector<std::string>& names,
                      const double* lat, const double* lon, size_t G, float* out);   // out[g*names.size()+k]
void compute_forcings(const std::string& valid_iso, ...same...);
```
Unknown names throw `std::invalid_argument`. Double precision, cast to float at the end; `-ffp-contract=off` (project
flag) prevents FMA contraction. OpenMP over grid points; scalars of the date computed once; independent of the thread
count (tested).  One call per step.

## 7. Verification

Generator: `python/reference/make_forcings_fixture.py` (run in `$IE_SCRATCH/venv-ref-varda` on a cpu-short
node; it calls anemoi's own `ComputedForcings.load_forcings_array` from the 9505dd0 clone, with the lat/lon from
`anemoi.utils.checkpoints.load_metadata(..., supporting_arrays=True)`). Output
`$VARDA_SCRATCH/forcings/<model>.safetensors`: 22 dates per model (2024-06-01T06/12, 2024-12-31T18, 2024-02-29T00 and
T23, 2025-01-01T00, 2024-01-01T00, 2023-12-31T23, 2023-03-01T00, 2024-07-01T09, 2024-06-21T15, 2000-02-29T12,
2100-03-01T06, eight 6-hourly steps after 2024-06-01T00, and 2024-06-01T01/02/03). It also re-checks insolation against
`earthkit.meteo.solar.cos_solar_zenith_angle` (equal in every case).

Test: `tests/test_forcings.cpp` (ctest `test_forcings`; needs `$VARDA_SCRATCH` fixtures and `converted/`, skips the
data part with a message otherwise). Per forcing and date it prints max abs difference, count of elements not bitwise
equal and max ulp. Results (OpenMP 16 threads, glibc libm, gcc 12.3):

| model | dates | elements compared | not bitwise equal | max abs | max ulp |
|---|---|---|---|---|---|
| dev (G = 40 320) | 22 | 4 435 200 | 0 | 0 | 0 |
| varda (G = 1 688 650) | 22 | 185 751 500 | 0 | 0 | 0 |

So the target (bitwise equality in float32) is met. Reason it is attainable: numpy 2.3.5 on these nodes calls the
same glibc `sin`/`cos` for float64 as C++ (checked: 2 M random arguments, `np.sin`/`np.cos` array and scalar paths are
bitwise equal to `math.sin`/`math.cos`), and the rounding to float32 absorbs nothing else. Caveat (not tested): this
holds on the CPU used here (EPYC 7713, AVX2); a numpy build dispatching float64 trig to another SIMD implementation
(e.g. AVX-512 SVML) could differ from libm by an ulp in double, which would show as rare 1-ulp float32 differences.

Recorded rollout forcings: `$VARDA_SCRATCH/rollout/forcings/forcings.pt` (23 dates, 2024-06-01T06 + 6 h steps,
varda grid, recorded inside the real runner loop). `VARDA_FORCINGS_DUMP=f.safetensors ./test_forcings` dumps
the C++ values for those dates and `.venv/bin/python -I python/reference/compare_forcings_recorded.py f.safetensors
forcings.pt` compares: **0 of 194 194 750 elements not bitwise equal** (all five forcings, max abs 0).

Converter verification (`python/tests/test_convert.py`, run with `VARDA_CONVERTED`, `VARDA_WEIGHTS_PT`, `VARDA_CKPTS`):
10 tests ok, including: names = contract names + `grid.latitudes`/`grid.longitudes`; those two are F64 `[n_data_nodes]`;
agreement with `nodes.data.attr` < 1e-6; `grid.*` bitwise equal to the zip members read independently; every other
tensor bitwise equal to the Python engine's `weights.pt` and the config equal to its `config.json`. Additionally all
384 old tensors and the metadata of `converted/<model>` were byte-identical to `converted-new/<model>` (that copy since deleted).
