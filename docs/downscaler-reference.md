# Temporal downscaler (varda-td): reference run of anemoi-inference's `temporal_downscaler` runner

The second model of Varda-single-1.0 is the 1-hourly temporal downscaler
(`varda-temporal-downscaler-det-sgm/model.ckpt`, sha256 `452fda53b073…`, uuid `00e40f03-2f7e-4c1c-8e6d-91ed9062249f`).
This page records what the original runner does, and the reference files the C++ side is judged against. The
data come from the forecaster reference outputs of `docs/realdata-reference.md` (2026-10-22T15, steps 0 .. 24 h
every 6 h). Code (tools only): `python/reference/realdata/{run_reference_downscaler.py, job_reference_downscaler.sh,
compare_downscaler.py}`. Artefacts: `$VARDA_SCRATCH/downscale/ref/20261022T15/<tag>/`.

## 1. Results at a glance

| item | result |
|---|---|
| model | inputs `(1, 2, G, 77)`: the state at t and t+6 h (66 prognostics + 11 forcings); output `(1, 6, 1, G, 67)`: the states at t+1 .. t+6 h; 67 = the forecaster's output names (tp is the per-hour precipitation) |
| runs (2026-10-22T15, lead time 24 h = 4 windows, 25 hourly files) | `fp16` (GPU, fp16 autocast, fed by the forecaster fp16 outputs), `fp32` (GPU, autocast off, TF32 off, forecaster fp32), `cpu-fp32` (CPU, pyg attention, forecaster fp32), `fp16-fp32in` (GPU, fp16 autocast, fed by the forecaster fp32 outputs: the downscaler's own fp16 effect on identical inputs) |
| what the model sees | the **GRIB-decoded** forecaster states (24-bit CCSDS packing of the forecaster's GRIB), not its raw outputs: decoded vs raw `rel_absmax` <= 1.2e-7 (section 5.3); the window k slot 1 equals window k+1 slot 0 bitwise |
| constants | `z`, `lsm` from the forecaster's step-0 GRIB (valid t0), the same in every window and both slots; sin/cos lat/lon computed from the cutout coordinates |
| dynamic forcings | computed by earthkit at the two slot dates of each window (t and t+6 h), instant values |
| tp | per-hour model output, ReLU-accumulated from the start of the whole forecast (carried across windows), x1000 (m -> kg m-2); **not** replaced by the forecaster's value at 6 h multiples |
| boundary hours 6/12/18/24 | every file except tp is the forecaster's GRIB value, copied bitwise (66 of 66 messages, all four precisions) |
| fp32 GPU vs CPU | inputs bitwise equal; raw outputs `rel_absmax` <= 5.2e-6; decoded GRIB <= 1.4e-5 |
| fp16 on identical inputs (`fp16-fp32in` vs `fp32`) | raw outputs <= 7.3e-3 (tp), GRIB <= 8.9e-3 |
| fp16 chain (`fp16` vs `fp32`, forecaster precision included) | inputs <= 5.6e-3, raw outputs <= 1.04e-2, GRIB <= 1.09e-2 |

## 2. Environment and rerun

The downscaler pins other `anemoi-core` (models, graphs) and `anemoi-datasets` revisions than the forecaster
(`anemoi-models` d9cf401c vs 3fe7fcc; with the forecaster venv the downscaler fails at `node_attributes(…, batch_size=…)`).
So `$VARDA_SCRATCH/ref-oper-td/.venv` is built from the downscaler's HF `pyproject.toml` (`uv sync`): same
anemoi-inference 9505dd0, anemoi-plugins-meteoswiss a1488e1, torch 2.6.0+cu124, eccodes 2.47.0; only anemoi-models
0.12.1.post30, anemoi-graphs, anemoi-datasets and a few unrelated libraries differ from the forecaster's real-data
environment (`python/reference/realdata/requirements-ref-oper.lock.txt`; freeze of this one:
`$VARDA_SCRATCH/ref-oper-td/requirements-ref-td.lock.txt`; the golden-data venv of the downscaler is another one,
`python/reference/requirements-ref-varda-td.lock.txt`).

The driver is `run_reference_realdata.py`'s sibling: the notebook's environment (ECCODES_DEFINITION_PATH from
eccodes_cosmo_resources, ECCODES_VERSION_CHECK_OFF=1, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, `python -O`), a run directory that stands in for the HF repo root
(`outputs/` = symlinks to the forecaster's 5 LAM + 5 global GRIB files, `templates/`, `varda-temporal-downscaler-det-sgm/`),
then `create_runner(RunConfiguration.load(<HF inference_opendata.yaml>, overrides))` and `runner.execute()`. No network.
Deliberate differences from the HF config: `runner: temporal_downscaler` instead of
`parallel: {base_runner: temporal_downscaler, cluster: {manual: 1}}` (one device); `date`, `lead_time: 1d` (as the
config), `device`, `precision` as overrides; on CPU `--attention-backend pyg` (the pickled backend is `triton`, which has
no CPU driver; the GPU runs use the pickled `triton`).

```
PY=$VARDA_SCRATCH/ref-oper-td/.venv/bin/python; R=$VARDA_SCRATCH/realdata/20261022T15; D=$VARDA_SCRATCH/downscale/ref/20261022T15
# NAME=<job name>; the job copies the code to $VARDA_SCRATCH/jobs/<NAME>/code-<jobid> first
NAME=td-D-fp16 sbatch $VARDA_SLURM_GPU ... python/reference/realdata/job_reference_downscaler.sh \
    "--device cuda --precision checkpoint --forecaster-run $R/ref-fp16 --out $D/fp16"
... "--device cuda --precision 32 --no-tf32 --forecaster-run $R/ref-fp32 --out $D/fp32"
... "--device cpu  --precision 32 --attention-backend pyg --forecaster-run $R/ref-fp32 --out $D/cpu-fp32"   # 64 cores, ~75 GB RSS, 20 min
... "--device cuda --precision checkpoint --forecaster-run $R/ref-fp32 --out $D/fp16-fp32in"
# spreads (CPU node; parts can run in parallel with --pairs a:b --no-extras --out-name part-N.json, then --merge)
$PY -I python/reference/realdata/compare_downscaler.py $D            # -> $D/spreads.json
```

Per run directory: `model_io.safetensors`, `tp_stages.safetensors`, `outputs/` (inputs as symlinks + 25 hourly
`varda-single_202610221500_<step>.grib`, 168-171 MB each), `dumps/`, `manifest.json`, `config.yaml`. 24 GB each.

| file | content |
|---|---|
| `model_io.safetensors` | `x.<k>` [2, G, 77] F32: the tensor given to `model.predict_step` (batch dim removed; the runner's tensor is `[1, 2, G, 77]`), model-input channel order, physical units, slot 0 = t, slot 1 = t+6 h. `y.<k>` [6, G, 67] F32: the raw model output (hours 1..6), before every post-processor (the model's output is already float32 under fp16 autocast). `forcdyn.<k>` [2, G, 5] F32: the 5 dynamic-forcing channels cut out of `x.<k>` (`cos_julian_day cos_local_time insolation sin_julian_day sin_local_time`, channels 4, 6, 8, 23, 25). k = 0..3. Metadata: `in_vars` (77 names), `out_vars` (67), `windows` (dates of the inputs and outputs), `layout`, `precision_arg`, `autocast`, `tf32`, `versions`, `source_sha256` of the 10 forecaster GRIB files (the `uuid` field reads `None` in these four runs: the checkpoint identity is `checkpoint_sha256` in the manifest) |
| `tp_stages.safetensors` | tp [G] F32 at each stage of every state the runner post-processes: `call<n>.pre` (model output), `.acc` (after Accumulate), `.post` (after the x1000 rescale), `written.<j>` (LAM state just before GRIB encoding); metadata `calls`, `written` (JSON: step, date, previous_step, start_steps, fields per call) |
| `dumps/*.safetensors` | window 0 only: `source.prognostics.{lam_0,global}.w0` and `cutout.prognostics.w0` (the decoded GRIB states of t0 and t0+6 h, [2, n]), `source.constant_forcings.*.w0`, `cutout.constant_forcings.w0`, `forcings.constant.0`, `forcings.dynamic.0` (the provider outputs [vars, 2, G]), `written.0..6` (the seven states of window 0 with all fields as the writer got them) |
| `manifest.json` | command, versions, device, autocast, TF32, attention backends, thread count, per-window predict seconds, total, peak CUDA memory, sha256 and size of every input and output file, every post-processor call (step, date, fields changed), every written state |

The runs also reproduce the forecaster's input-file layout: the 10 inputs are `20*.grib` and `global*.grib` as
`ls` sorts them (`_0, _12, _18, _24, _6`); the input selects by metadata, not by file order.

## 3. What the runner does (anemoi-inference 9505dd0, `anemoi/inference/runners/temporal_downscaler.py`)

References are `file:line` of that revision; `td` = `runners/temporal_downscaler.py`.

### 3.1 Windows and dates

- `time_step = temporal_downscaling_window = data_frequency * (input_explicit_times[1] - input_explicit_times[0])` =
  1 h * (6 - 0) = **6 h** (td:75-84). The checkpoint's `explicit_times`: input `[0, 6]`, target `[1..6]`.
  `TemporalDownscalerMetadata.lagged = [0, 6 h]` (td:33-37): input slots are t and t+6 h (looking forward).
- `num_windows = int(lead_time / 6 h)` (td:110): 24 h gives 4 windows; a lead time that is not a multiple only warns and
  drops the remainder (td:112-118). `reference_date` (= `date`) is required (td:67).
- Window k: `window_start = reference_date + k * 6 h`, `window_end = window_start + 6 h` (td:140-141). Input slot 0 = t0 + 6k h,
  slot 1 = t0 + 6(k+1) h. Output hour h = 1..6 of window k is the state at t0 + 6k + h h, global output step 6k + h.
  The six output dates are `window_start + (s+1) * 6 h / (6 - 0)` for s = 0..5 (`temporal_downscaler_stepper`, td:199-227).
- One model call per window (`predict_step(model, x, fcstep=0)`, td:307), inside `torch.inference_mode()` and
  `torch.autocast(device, dtype=self.autocast)` (td:303): fp16 for the checkpoint's `16-mixed`, float32 when `precision: 32`.

### 3.2 Which GRIB messages feed each input slot

The config's `input` is a `cutout` of two `grib` inputs (`lam_0`: `outputs/20*.grib`, `global`: `outputs/global*.grib`),
both with the same `namer` (ICON `shortName` -> anemoi name: `T t_<level>`, `U u_`, `V v_`, `QV q_`, `FI z_`, `PMSL msl`,
`PS sp`, `T_2M 2t`, `TD_2M 2d`, `U_10M 10u`, `V_10M 10v`, `FIS z`, `FR_LAND lsm`, `TOT_PREC tp`; they match the LAM files; the global
files carry IFS short names (`10u`, `2t`, `t`, `q`, ... on `reduced_gg`, 540 670 values with a bitmap), which no rule matches, so the default
namer gives the anemoi names `10u`, `t_500`, ...). There is **no** `constant_forcings` section:
`create_input("constant_forcings")` falls back to the `input` section (runner.py:739-743), and `dynamic_forcings` is
`EmptyInput` (computed forcings only).

- **Prognostics** (td:149-152): `create_input_state(dates=[window_start, window_end], select_reference_date=True, ref_date_index=0)`.
  `EkdInput._filter_by_time` (inputs/ekd.py:166-198) selects the GRIB messages with `valid_datetime in dates` **and**
  `dataDate/dataTime = reference_date` (that is: the forecast started at t0, steps `6k` and `6k+6`). For the 66 prognostic
  names (`10u 10v 2d 2t msl sp` and `q t u v z` x 12 levels `50 100 150 200 250 300 400 500 700 850 925 1000`) this reads two
  messages per name: from `20261022…_<6k>.grib` and `…_<6k+6>.grib` (LAM) and the matching `global-…` files.
  Each source state is `[2, n_points]` F32 (`to_numpy(dtype=float32)` of the GRIB-decoded values).
- **Cutout** (inputs/cutout.py:161-253): each source is masked by the checkpoint's `<src>/cutout_mask` and concatenated, LAM
  first: `G = 1 147 980 (lam_0, all points) + 540 670 (global points outside the LAM; 1 410 of the 542 080 N320 points
  are masked NaN in the global file and are dropped)` = 1 688 650. These masks, and `latitudes/longitudes`, are bitwise those of the
  forecaster reference. The coordinates the cutout carries come from the checkpoint's supporting arrays `source0/*`, `source1/*`
  (the source state's latitudes do not match the field length, so the cutout falls back to them: cutout.py:195-218; the log says
  "Mismatch between latitudes and field points for source lam_0: 1688650 vs 1147980").
  Consequence: the global rows of the model input are the **forecaster's global GRIB output** (which was itself the model's
  output on the cutout grid, written as N320 with `assign_mask`), restricted by the mask.
- **Constants** (td:121-136): `constant_forcings_inputs.create_input_state(dates=[reference_date], constant=True, ref_date_index=0)`,
  **once**, then each field is duplicated to two dates (`np.concatenate([a, a])`) and reused in every window (td:157, with
  `["date"] = window_start`). With the `grib` input `constant=True` has no effect: the filter is `valid_datetime == t0` and
  `dataDate/dataTime` are not selected, so `z` (= `FIS`) and `lsm` (= `FR_LAND`) come from the **step-0 forecaster file**
  (`…_0.grib`, LAM and global), which contains them (70 messages: 66 prognostics + `FIS FR_LAND SSO_SIGMA SSO_STDH`).
  `slor`/`sdor` are not retrieved (the downscaler does not use them). In `x`: `z` is channel 64, `lsm` 9; identical in both
  slots and in all four windows (checked bitwise); `lsm` in [0, 1]; `z` has |max| 58 225 m2/s2 and is the decoded 24-bit value.
- **Computed forcings** (tensors.py `add_initial_forcings_to_input_state`, :244-300, called from `prepare_input_tensor`):
  `dates = input_state["date"] + lagged = [window_start, window_start + 6 h]`. The state's `date` is the first date
  (`ref_date_index=0`). Constants: `cos_latitude sin_latitude cos_longitude sin_longitude` (channels 5, 24, 7, 26) =
  `cos/sin(deg2rad(lat/lon))` of the cutout coordinates, evaluated in float64 and cast to float32 (equal bitwise to
  `np.cos(np.deg2rad(lat)).astype(float32)`). Dynamic (earthkit-data `sources/forcings.py`, applied per date, same value
  for the instant, not an hourly mean): `cos/sin_julian_day` = `cos/sin(2 pi * jd / 365.25)` with
  `jd = days since Jan 1 00:00 of that year + seconds / 86400` (the lat/lon-independent scalar, forcings.py:142-158);
  `cos/sin_local_time` = `cos/sin(2 pi * local_time / 24)` with `local_time = (lon / 360 * 24 + hours since 00:00) mod 24` (lon of
  the cutout in degrees as stored, range about -0.82 .. 359.72; forcings.py:160-177); `insolation` =
  `earthkit.meteo.solar.cos_solar_zenith_angle(date, lat, lon)` (forcings.py:179-204). Checked against numpy on the
  dumped inputs: cos/sin julian and local time agree to <= 3e-8 (float32 rounding), insolation <= 3e-8.
  The dates are `t0 + 6k h` and `t0 + 6(k+1) h` in UTC; for k = 0: 2026-10-22T15 and T21.
- All 77 channels, model-input order (alphabetical by name, `variable_to_input_tensor_index`): `10u 10v 2d 2t cos_julian_day
  cos_latitude cos_local_time cos_longitude insolation lsm msl q_100 … q_925 sin_julian_day sin_latitude sin_local_time
  sin_longitude sp t_100 … t_925 u_… v_… z z_100 … z_925`. The exact list is `in_vars` in the file metadata; the 67 outputs `out_vars`
  are the forecaster's (alphabetical, `tp` between `sp` and `t_100`).
- Nothing in the input is computed from the previous window's *model output*: both slots always come from the forecaster's GRIB files.
  The two slots are decoded floats, so slot 1 of window k equals slot 0 of window k+1 bitwise (checked for all prognostics).

### 3.3 Window outputs and the "initial" state

- `forecast` (td:229-350) first yields (if `write_initial_state`, which defaults to true) an *initial state*
  (td:275-300): date = window_start, step 0, `fields` = the 66 prognostic names (those whose name is also an output name:
  input names minus the forcings and constants, **and no tp**) taken from **slot 0 of the input tensor**
  (`input_tensors_torch[0, boundary_times[0]]`), i.e. the decoded GRIB values (so step 0 of a window is not the model output).
  Then the model runs and the six outputs are yielded one by one (td:309-349), each as the *same* `new_states` dict whose
  `fields` entries are replaced by `output[s, :, i]` (views of the output array); `date`/`step`/`previous_step` are window-relative
  (`step` = 1..6 h).
- `execute` consumes seven states per window (td:173-194): the initial state (`state_idx` 0) and the six outputs.
  `if window_idx != 0 and state_idx == boundary_idx[0]: continue` skips the initial state in windows 1..3 (td:184): it is the
  last output of the previous window. So each window writes hours `6k` (k = 0 only) and `6k+1 .. 6k+6`: **25 states for 24 h**.
  `state["step"] += window_idx * 6 h` (td:188) makes the step global; `previous_step` stays window-relative (for window k >= 1 and
  global step 6k+1 it is `0:00:00`; unused for tp, see 3.5).
- The post-processors (3.4) run on each state before `outputs.write_state` (td:191-194), then the tee/GRIB output writes it
  (`Output.write_state` skips steps not a multiple of `output_frequency: 1h`, all are; then the output's own post-processors).
  Before each window `initial_state = input_states.copy()` also goes through the top-level post-processors (td:167-171)
  and then `outputs.open(initial_state)`: those calls have no tp and change nothing (4 extra calls in `manifest.json["calls"]`).

### 3.4 Post-processors and the hourly outputs

Top level (config): (1) `accumulate_from_start_of_forecast: {accumulations: [tp]}`, (2) `forward_transform_filter: rescale
{scale: 1000, offset: 0, param: TOT_PREC}`. Then the output `tee -> grib` with its own list: `extract_from_state: lam_0`
(keeps the first 1 147 980 points: the LAM), then the MeteoSwiss filter `copy-prognostic-from-forecaster`.

- **Accumulate** (post_processors/accumulate.py:63-90): per call, `acc[tp] += max(0, tp)` (float32, `allow_negative: false`),
  `state.fields[tp] = acc` (the same array object every time), `state["start_steps"][tp] = 0`. The accumulator lives in the
  processor, so it **runs through all windows**: tp at global hour H is the sum of the ReLU'd hourly outputs of hours 1..H
  (a 24 h run: hour 24 holds the 24-hour total). The skipped boundary states are not processed, so nothing is double counted.
  Verified bitwise on the dumps (`tp_check` in `spreads.json`: `acc == cumsum(max(pre, 0))` for all 24 states, all runs).
  The model's own ReLU on tp makes `pre` >= 0 (0 negative points).
- **Rescale** (`ForwardTransformFilter`, `param: TOT_PREC`, selected by its GRIB param in `metadata.yaml`, i.e. `tp`):
  `tp *= 1000` (m -> kg m-2), float32 (`post == acc * float32(1000)` bitwise, all calls).
- **extract_from_state lam_0**: states drop to the 1 147 980 LAM points. **No global output** for the downscaler.
- **copy-prognostic-from-forecaster** (anemoi-plugins-meteoswiss, `transform/filters/copy_prognostic_from_forecaster.py`):
  for each message of the state at lead time `L = step` (instants: `step`; accumulations: `step`): if `L > 0`, `L % 6h == 0`
  (`common_leadtime: 6h`) and the anemoi name is **not** in `params_to_keep = [tp]`, the values are replaced by the forecaster
  GRIB message with the same anemoi name and lead (`outputs/20*.grib`, indexed once; `lead = valid_time - base_time`)
  (:212-256), keeping the downscaler's metadata (`new_field_from_numpy(match.to_numpy(), template=field)`). A missing
  match only warns; none occurred. Result: at hours 6, 12, 18, 24 the 66 non-tp messages are the forecaster's decoded values
  **re-encoded at 24 bits** (bitwise equal values to the forecaster's file: `copy_check`, 66 of 66 for steps 6/12/18/24 in all
  runs) and **tp stays the downscaler's** accumulated value. Step 0 (`L = 0`) is never replaced.
  Hours 1..5, 7..11, ... are the model's own outputs (`y`, float32) encoded at 24 bits CCSDS.
- **tp units and values in the files**: kg m-2 (mm), accumulated 0 .. H h, stepType `accum`, `startStep 0`, `endStep H`
  (`stepRange 0-H`), TOT_PREC (paramId 500041). Peak values: 26.7 mm (hour 1), 286 mm (hour 24) in the fp32 GPU run
  (`post_max_kgm2` in `tp_check`). The downscaler's tp at hour 6 is **not** the forecaster's tp of step 6: they differ by up to
  11.9 mm at points (absmax 20.5 mm), rel_absmax 0.58 (hour 6), 0.45 (hour 24) in the fp32 run (`copy_check.different`).
  The forecaster's tp is the sum of its 6-hourly outputs; the downscaler's is the sum of its hourly outputs; the files keep both.

### 3.5 GRIB encoding of the hourly files

`GribOutput.write_step` (outputs/grib.py:285-370, keys: grib/encoding.py): ICON templates
(`templates/templates_index_icon.yaml`), `encoding: {typeOfGeneratingProcess: 2}`. Observed in every run:

- **25 files**, `outputs/varda-single_202610221500_<step>.grib`, `<step>` = 0 .. 24 **without zero padding**
  (the config's `{step:03}` yields `_0`, `_1`, ..., `_24`; `{dateTime}` = `202610221500`). Dated by the reference date
  (not the valid time): `dataDate 20261022`, `dataTime 1500` in every file.
- **Message count**: step 0: **66** (no tp; the four constants FIS FR_LAND SSO_SIGMA SSO_STDH of the forecaster's step 0 are
  not written); steps 1 .. 24: **67**. Per file: `U_10M V_10M` (heightAboveGround 10), `TD_2M T_2M` (2), `PMSL` (meanSea),
  `QV` x12 (isobaricInhPa 100 1000 150 200 250 300 400 50 500 700 850 925: lexicographic by name), `PS` (surface) , `T` x12,
  `U` x12, `V` x12, `FI` x12, then `TOT_PREC` (surface, accum) last (steps >= 1). Same keys as the forecaster's messages
  (edition 2, centre `lssw`, `gridDefinitionTemplateNumber 101` = ICON unstructured, `numberOfValues 1147980`, `uuidOfHGrid
  17643da2…`, `packingType grid_ccsds`, `bitsPerValue 24`, `typeOfGeneratingProcess 2`, `generatingProcessIdentifier 255`);
  the forecaster file puts TOT_PREC after PS, the hourly file last.
- **Instantaneous fields**: `stepType instant`, `step = H`, `productDefinitionTemplateNumber 1`, `forecastTime H`.
  **tp**: `stepType accum`, `startStep 0`, `endStep H`, PDT 8, `typeOfStatisticalProcessing 1` (accumulation), `lengthOfTimeRange H`,
  `forecastTime 0`. The writer warns "tp accumulation does not have a period set, using previous_step as start": the metadata's
  `period: [1h]` is a list, so `variable.period` is not read, `start = previous_step`, then overridden by
  `start_steps[tp] = 0` from Accumulate (grib/encoding.py:150-170); `previous_step` (window-relative for windows >= 1) never matters.
- **`negative_step_mode: skip`** never fires here: it skips a variable whose encoded base time is before the reference date,
  which can only happen with a `period` (start before 0); with `start_steps` forcing 0 no variable is skipped (0 skip warnings).
- **Step 0** (`varda-single_…_0.grib`) is the initial state of window 0: slot 0 of the input tensor, i.e. the decoded forecaster
  step-0 values, re-encoded: 62 of 66 messages bitwise equal to the forecaster's step-0 messages; 4 differ at re-quantisation
  level (`FI@1000` rel 5.2e-8, `QV@150` 4.4e-8, `QV@200` 8.8e-8, `V@100` 1.1e-7; `copy_check`).
- Written data type: float32 arrays, encoded to 24-bit CCSDS; the GRIB values are the quantised ones (section 6.3).

### 3.6 Surprises and things the C++ side must mirror

1. The model never sees the forecaster's raw floats, only their 24-bit CCSDS-decoded values (decoded vs raw `rel_absmax`
   <= 1.2e-7, `max_abs` <= 7.6e-6 for the winds, 0.002 for `z`). A C++ implementation fed the raw forecast tensors differs
   from the reference by this much; fed the GRIB files it must decode them (ecCodes) and take the 66 fields at the two steps.
2. `z` and `lsm` are read from the step-0 forecaster file, not recomputed or taken from the checkpoint.
3. The `lat/lon` for the sin/cos forcings and the insolation are the checkpoint's coordinates (float64 -> float32 cast of the
   function value), identical to the forecaster's.
4. tp accumulation is global over windows; the files hold 0..H sums in mm; the per-hour model tp is in m.
5. Hours 6/12/18/24: non-tp fields are the forecaster's; the downscaler's own hour-6 output (`y[5]` of each window) is never
   written for those fields (it is still the input of nothing: the next window's slot 0 is the forecaster state).
6. Step 0 is written from the input (decoded), not by the model; windows 1..3 skip it.
7. `previous_step` is window-relative: harmless here because of `start_steps`.
8. The first predict_step of a window is a single call per window; there is no state carried between windows apart from the
   tp accumulator.

## 4. Runs, timings

One job per run, `$VARDA_SCRATCH/jobs/td-D-*/`. `predict_step` seconds per window (synchronised) and totals; all four
windows use identical shapes:

| run | device | precision | windows 0..3 (s) | execute (s) | peak GPU mem / host RSS |
|---|---|---|---|---|---|
| `fp32` | A100-SXM4-80GB, 32 CPU threads | fp32, TF32 off (matmul, cudnn), triton attention | 13.24, 8.48, 8.49, 8.48 | 268 | 25.7 GB / 25 GB |
| `fp16-fp32in` | A100-SXM4-80GB | fp16 autocast, cudnn TF32 on (default), triton | 5.87, 3.34, 1.75, 1.74 | 233 | 16.4 GB / 29 GB |
| `fp16` | NVIDIA PG506-232 (A100-class) on a node shared with two other model jobs | fp16 autocast, triton | 7.45, 3.03, 1.56, 3.02 | 1454 (slow shared filesystem) | 16.4 GB / 28 GB |
| `cpu-fp32` | AMD EPYC 7713, 64 threads, `pyg` attention | fp32 | 192.1, 191.8, 198.9, 200.7 | 1054 | - / 74 GB |

Warm fp16 is 1.7 s per window (1.74 and 1.75 s in `fp16-fp32in` windows 2 and 3; windows 0-1 include CUDA/triton warm-up and the first-call
cost; the `fp16` run's 3.0 s windows ran next to other jobs on the node: use the former). `execute` includes the GRIB reads and the 25 GRIB writes, not just the model. Setup (loading the model, building the runner) 41-112 s.
These are reference-run timings, not engine benchmarks.

## 5. Checks run on the data

### 5.1 tp bookkeeping (all four runs, `tp_check`)
`acc == cumsum(max(model tp, 0))` (bitwise float32), `post == acc * 1000f` (bitwise), the tp the writer got equals `post[lam]`
(the first 1 147 980 values) at every written step: 24 of 24 states, 0 unmatched, in every run.

### 5.2 Boundary hours and the step-0 file (`copy_check`)
Steps 6/12/18/24: 66 of 66 non-tp messages bitwise equal to the forecaster's, tp different (section 3.4); step 0: 62 of 66 bitwise.

### 5.3 GRIB packing loss (`packing`, decoded GRIB inputs vs the forecaster reference run's raw states, 66 prognostics + z, lsm)
For both slots of all windows: worst `rel_absmax` 1.19e-7 (fp16 chain, `u_100`, max_abs 7.6e-6 m/s) and 1.14e-7 (fp32 chain);
`z` max_abs 0.002 m2/s2 (rel 3.4e-8), `lsm` exactly equal. That is the 24-bit CCSDS quantisation of every input the
downscaler gets, relative to the per-field absmax. (The raw side: slot 0 of window 0 = `dumps/state0.pre`, later slots =
`dumps/step<s>.y`, of the forecaster reference run of the same precision.)

## 6. Spreads (seed for per-step tolerances)

Metric (`docs/format.md` section 2): float64, `rel_absmax = max|a - r| / max|r|` over the whole field, with `r` the second name.
The cells show the worst variable of the step (66 or 67 variables) and its `rel_absmax`; the full per-variable `rel_absmax` /
`max_abs` / `absmax` are in `$VARDA_SCRATCH/downscale/ref/20261022T15/spreads.json`
(`pairs.<a>_vs_<b>.model_output.<window>.<hour>.<var>`, `.model_input.<window>.<slot>.<var>`, `.grib.<step>.<message>`).
Median variable per row of the raw outputs: 6e-4 .. 1.5e-3 (fp16-fp32in vs fp32), 1.4e-3 .. 2.5e-3 (fp16 vs fp32), about 1e-6 (cpu-fp32 vs fp32).

* **cpu-fp32 vs fp32**: same inputs bitwise; CPU `pyg` vs GPU `triton` fp32 (the downscaler's fp32 floor).
* **fp16-fp32in vs fp32**: same inputs bitwise; fp16 autocast vs fp32 on the GPU (the downscaler's own fp16 effect).
  The pair `fp16-fp32in vs cpu-fp32` has the same numbers to 2.4e-6 (in `spreads.json`).
* **fp16 vs fp32**: the full chain: the forecaster ran fp16 / fp32 and the downscaler ran fp16 / fp32.

### 6.1 Raw model output `y` (before post-processing), per window and output hour

| window | hour | cpu-fp32 vs fp32 | fp16-fp32in vs fp32 | fp16 vs fp32 |
|---|---|---|---|---|
| 0 | 1 | 2.91e-06 (q_100) | 3.99e-03 (v_50) | 4.29e-03 (v_50) |
| 0 | 2 | 4.00e-06 (tp) | 7.34e-03 (tp) | 7.72e-03 (tp) |
| 0 | 3 | 4.13e-06 (tp) | 5.93e-03 (tp) | 7.56e-03 (tp) |
| 0 | 4 | 4.31e-06 (tp) | 7.14e-03 (tp) | 1.04e-02 (tp) |
| 0 | 5 | 4.10e-06 (tp) | 7.25e-03 (tp) | 6.33e-03 (tp) |
| 0 | 6 | 4.81e-06 (tp) | 5.12e-03 (tp) | 5.23e-03 (tp) |
| 1 | 1 | 3.65e-06 (tp) | 4.21e-03 (tp) | 5.24e-03 (q_925) |
| 1 | 2 | 3.53e-06 (10v) | 3.83e-03 (q_150) | 5.53e-03 (q_200) |
| 1 | 3 | 3.72e-06 (10v) | 4.08e-03 (q_200) | 6.75e-03 (q_200) |
| 1 | 4 | 4.04e-06 (v_400) | 4.12e-03 (q_200) | 6.49e-03 (q_850) |
| 1 | 5 | 3.36e-06 (tp) | 3.69e-03 (v_50) | 6.03e-03 (tp) |
| 1 | 6 | 5.21e-06 (tp) | 3.25e-03 (tp) | 4.35e-03 (tp) |
| 2 | 1 | 2.89e-06 (q_100) | 3.24e-03 (v_50) | 4.81e-03 (q_925) |
| 2 | 2 | 2.89e-06 (q_700) | 3.77e-03 (q_850) | 6.88e-03 (q_925) |
| 2 | 3 | 3.20e-06 (10u) | 4.05e-03 (q_200) | 7.65e-03 (q_850) |
| 2 | 4 | 2.75e-06 (v_925) | 4.65e-03 (tp) | 6.87e-03 (q_850) |
| 2 | 5 | 3.13e-06 (q_850) | 3.96e-03 (v_50) | 5.46e-03 (q_700) |
| 2 | 6 | 3.33e-06 (q_100) | 3.63e-03 (tp) | 4.21e-03 (tp) |
| 3 | 1 | 4.66e-06 (10v) | 3.74e-03 (v_50) | 4.60e-03 (q_925) |
| 3 | 2 | 3.20e-06 (q_500) | 3.61e-03 (v_50) | 6.21e-03 (q_925) |
| 3 | 3 | 3.92e-06 (v_1000) | 4.20e-03 (q_200) | 8.78e-03 (q_850) |
| 3 | 4 | 3.92e-06 (q_850) | 4.21e-03 (q_200) | 7.13e-03 (q_850) |
| 3 | 5 | 3.03e-06 (q_100) | 3.68e-03 (v_50) | 5.67e-03 (q_150) |
| 3 | 6 | 3.32e-06 (q_100) | 2.26e-03 (tp) | 5.76e-03 (v_300) |

Worst `max_abs`: cpu-fp32 vs fp32 2.6e-4 (`v_400`, w1 h4); fp16-fp32in vs fp32 0.149 m/s (`v_50`, w2 h5); fp16 vs fp32 0.395 m/s
(`v_300`, w3 h6). Worst variables over all windows and hours for fp16-fp32in vs fp32: `tp` 7.3e-3 (max_abs 1.1e-4 m),
`q_200` 4.8e-3, `v_50` 4.4e-3 (0.15 m/s), `q_150` 4.1e-3, `q_100` 4.0e-3, `q_850` 3.9e-3.

### 6.2 Model input `x` (decoded GRIB + forcings), per window and slot (worst variable)

| window | slot | cpu-fp32 vs fp32 | fp16 vs fp32 |
|---|---|---|---|
| 0 | 0 | 0.00e+00 (10u) | 0.00e+00 (10u) |
| 0 | 1 | 0.00e+00 (10u) | 3.73e-03 (q_925) |
| 1 | 0 | 0.00e+00 (10u) | 3.73e-03 (q_925) |
| 1 | 1 | 0.00e+00 (10u) | 3.77e-03 (q_100) |
| 2 | 0 | 0.00e+00 (10u) | 3.77e-03 (q_100) |
| 2 | 1 | 0.00e+00 (10u) | 4.06e-03 (q_925) |
| 3 | 0 | 0.00e+00 (10u) | 4.06e-03 (q_925) |
| 3 | 1 | 0.00e+00 (10u) | 5.59e-03 (v_300) |

(`cpu-fp32 vs fp32` is bitwise zero: the same forecaster files. In the `fp16 vs fp32` column the difference is the forecaster's own
fp16/fp32 spread, which grows along its rollout; the forcings and constants are bitwise equal in all runs.)

### 6.3 Decoded hourly GRIB messages, per step (worst message)

Steps 6/12/18/24: the non-tp messages are the forecaster's (copied), so the spread there is the forecaster's own (fp16 vs fp32)
or zero (the same forecaster run), and the worst message is tp for the same-input pairs.

| step | cpu-fp32 vs fp32 | fp16-fp32in vs fp32 | fp16 vs fp32 |
|---|---|---|---|
| 0 | 0.00e+00 (FI@isobaricInhPa:100) | 0.00e+00 (FI@isobaricInhPa:100) | 0.00e+00 (FI@isobaricInhPa:100) |
| 1 | 1.28e-05 (V@isobaricInhPa:50) | 8.91e-03 (V@isobaricInhPa:50) | 5.58e-03 (V@isobaricInhPa:50) |
| 2 | 1.01e-05 (QV@isobaricInhPa:200) | 7.28e-03 (V@isobaricInhPa:100) | 6.87e-03 (V@isobaricInhPa:50) |
| 3 | 1.21e-05 (V@isobaricInhPa:50) | 7.92e-03 (V@isobaricInhPa:50) | 8.47e-03 (V@isobaricInhPa:50) |
| 4 | 1.14e-05 (V@isobaricInhPa:50) | 7.33e-03 (V@isobaricInhPa:50) | 8.29e-03 (V@isobaricInhPa:50) |
| 5 | 1.41e-05 (V@isobaricInhPa:50) | 6.97e-03 (V@isobaricInhPa:50) | 6.44e-03 (U@isobaricInhPa:50) |
| 6 | 2.71e-06 (TOT_PREC@surface:0) | 3.58e-03 (TOT_PREC@surface:0) | 4.63e-03 (V@isobaricInhPa:50) |
| 7 | 7.09e-06 (V@isobaricInhPa:50) | 6.42e-03 (V@isobaricInhPa:50) | 4.67e-03 (V@isobaricInhPa:50) |
| 8 | 9.71e-06 (V@isobaricInhPa:50) | 7.27e-03 (V@isobaricInhPa:50) | 7.03e-03 (V@isobaricInhPa:50) |
| 9 | 1.43e-05 (V@isobaricInhPa:50) | 7.58e-03 (V@isobaricInhPa:50) | 6.98e-03 (V@isobaricInhPa:50) |
| 10 | 1.12e-05 (V@isobaricInhPa:50) | 7.13e-03 (V@isobaricInhPa:50) | 7.25e-03 (V@isobaricInhPa:50) |
| 11 | 1.02e-05 (V@isobaricInhPa:50) | 8.41e-03 (V@isobaricInhPa:50) | 5.01e-03 (QV@isobaricInhPa:250) |
| 12 | 4.49e-06 (TOT_PREC@surface:0) | 3.58e-03 (TOT_PREC@surface:0) | 5.93e-03 (V@isobaricInhPa:300) |
| 13 | 7.64e-06 (V@isobaricInhPa:50) | 6.59e-03 (QV@isobaricInhPa:400) | 5.08e-03 (U_10M@heightAboveGround:10) |
| 14 | 1.02e-05 (V@isobaricInhPa:50) | 6.33e-03 (V@isobaricInhPa:50) | 6.57e-03 (U@isobaricInhPa:1000) |
| 15 | 1.26e-05 (V@isobaricInhPa:50) | 8.69e-03 (V@isobaricInhPa:50) | 6.56e-03 (V@isobaricInhPa:50) |
| 16 | 1.26e-05 (V@isobaricInhPa:50) | 7.55e-03 (V@isobaricInhPa:50) | 7.30e-03 (V@isobaricInhPa:50) |
| 17 | 9.60e-06 (U@isobaricInhPa:50) | 6.33e-03 (V@isobaricInhPa:50) | 6.72e-03 (V@isobaricInhPa:50) |
| 18 | 4.06e-06 (TOT_PREC@surface:0) | 3.58e-03 (TOT_PREC@surface:0) | 4.70e-03 (V@isobaricInhPa:200) |
| 19 | 9.81e-06 (V_10M@heightAboveGround:10) | 5.35e-03 (U@isobaricInhPa:50) | 5.39e-03 (V@isobaricInhPa:50) |
| 20 | 1.07e-05 (QV@isobaricInhPa:300) | 4.57e-03 (QV@isobaricInhPa:200) | 5.18e-03 (V@isobaricInhPa:500) |
| 21 | 9.15e-06 (QV@isobaricInhPa:300) | 5.18e-03 (QV@isobaricInhPa:200) | 6.79e-03 (V@isobaricInhPa:500) |
| 22 | 1.05e-05 (QV@isobaricInhPa:300) | 6.61e-03 (QV@isobaricInhPa:200) | 9.68e-03 (QV@isobaricInhPa:200) |
| 23 | 1.10e-05 (V@isobaricInhPa:50) | 5.45e-03 (U@isobaricInhPa:50) | 1.02e-02 (V@isobaricInhPa:500) |
| 24 | 5.56e-06 (TOT_PREC@surface:0) | 3.91e-03 (TOT_PREC@surface:0) | 1.09e-02 (V@isobaricInhPa:400) |

## 7. Not done / limits

- One date (2026-10-22T15) and one lead time (24 h, 4 windows); no ensemble. The hourly GRIB files were checked with ecCodes keys
  and by value, not with an independent encoder.
- `uuid` in `model_io.safetensors` metadata reads `None` in the four runs (the checkpoint object does not expose it; the
  driver now writes the known uuid for future runs); `checkpoint_sha256` in each `manifest.json` is `452fda53b073…`.
- `fp16` ran on a busy shared node: its timings are not clean (use `fp16-fp32in`). Neither GPU run used CUDA graphs: the reference
  runner calls the eager model.
- The attention backend for the CPU run (`pyg`) differs from the pickled `triton`; the CPU fp32 vs GPU fp32 spread
  (<= 5.2e-6 raw) includes that difference.
