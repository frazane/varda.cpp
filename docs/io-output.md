# GRIB output: io config, `GribOutput`, tools, verification

Implements `docs/spec-io.md` §4 and the output rows of §5 (C3-C6). Files: `python/export_io_config.py`,
`src/varda/io/io_config.{h,cpp}`, `src/varda/io/output.{h,cpp}`, `apps/io/varda-grib-compare.cpp`,
`apps/io/varda-grib-write.cpp`, `tests/io/test_io_output.cpp`.

## 1. The io config (`varda-io.json`)

The engine does not parse YAML. `python/export_io_config.py` reads once, in the reference environment
(`$VARDA_SCRATCH/ref-oper/.venv/bin/python -I`):

- the checkpoint metadata from the zip, through anemoi-inference's own `Checkpoint` class with the Hugging Face
  `patch_metadata` applied (nothing is unpickled), including `select_variables` for the initial-state order;
- `configs/inference/inference_opendata.yaml`, the two template index files, the template file hashes.

```
$VARDA_SCRATCH/ref-oper/.venv/bin/python -I python/export_io_config.py \
    [--hf $VARDA_SCRATCH/hf] [--out $VARDA_SCRATCH/realdata/assets/varda-io.json]      # atomic write
```

It is strict: unknown top-level config keys, mars keys other than `date time step param levtype levelist`,
template-match keys other than `levtype param levelist name time_processing`, modifiers other than `patches`
(with back-references), post-processors other than the five used, non-integer encoding values, and
`allow_nans: false` / `write_initial_state: false` are errors. The loader (`IoConfig::parse`) is equally strict
(unknown or missing keys fail).

### Schema `varda-io/1` (key order is meaningful, the C++ reads it as `ordered_json`)

| key | content |
|---|---|
| `schema` | `"varda-io/1"` |
| `source` | provenance: checkpoint uuid, sha256 of the inference config and the patch, templates revision, sha256 of every template file, anemoi-inference version |
| `grid`, `area`, `number_of_grid_points` | `"N320"`, `[89.785, 0.0, -89.785, 359.719]`, 1 688 650 (the runner's `metadata.grid/area`) |
| `timestep_hours`, `lead_time_hours` | 6, 24 |
| `write_initial_state`, `allow_nans` | true, true |
| `cutout_sources` | `["lam_0", "global"]` |
| `variables` | all 98 entries of the patched `variables_metadata`, in that order: `name`, `in_model`, `mars` (patched: `date levelist levtype param step time`; `levelist` null for `z`; `param` is the COSMO name), `constant_in_time`, `computed_forcing`, `process` (null or `"accumulation"`), `period_hours` (accumulation: end - start of `period`, 6 for tp), `categories` (model variables) |
| `model_variables` | the 80 dataset variables, in model data order |
| `output_variables` | the 67 model outputs in output index order (`10u 10v 2d 2t msl q_* sp tp t_* u_* v_* z_*`; tp is index 18); the engine's `y` column order. `GribOutput` checks it against `Config::var_output` |
| `initial_state_fields` | the 70 step-0 fields: the 66 prognostics in `variables_metadata` order (`z_100 z_1000 ... msl sp q_* 2d t_100 ...`), then `z lsm slor sdor` |
| `kenda_namer_rules` | 19 rules `{match: {shortName}, name}`; first match wins; `{level}` in `name` (`IoConfig::kenda_name`) |
| `post_processors` | top level, config order: `accumulate_from_start_of_forecast {accumulations: [tp], allow_negative: false}`, `rescale {scale 1000, offset 0, param TOT_PREC}` |
| `outputs` | `lam`, `global` (config order) |

Each output: `name`, `kind: "grib"`, `path` (as configured, `outputs/{dateTime}_{step:03}.grib`), `path_effective`
(after `format_dataset_name`: `outputs/{dateTime}_{step}.grib`; the format spec is lost, spec §4.3),
`encoding` (`typeOfGeneratingProcess: 2`), `template_index {source, entries: [{match: {key: value|[values]}, file}]}`
(entry order = lookup order; `file` below the templates directory), `post_processors`
(lam: `extract_from_state lam_0`; global: `extract_mask {lam_0/cutout_mask, as_slice, inverse}`, then
`assign_mask {global/cutout_mask, fill_value nan}`), `modifiers` (global: one `patches` modifier: ordered
`variable: [{regex, keys}]` rules, `effective_patch` = index of the patch anemoi keeps, the last),
`check_encoding: true`, `missing_value: -9999`.

C++: `IoConfig::load(path)`; typed structs in `src/varda/io/io_config.h` (`IoVariable`, `NamerRule`,
`TemplateIndexEntry`, `PostProcessorDef`, `PatchRule`, `OutputDef`); `variable(name)`, `output(name)`,
`kenda_name(shortName, level)` for the input side. `Scalar` is null/int/string with Python `str()` spelling.

## 2. `GribOutput` (`src/varda/io/output.h`)

```
GribOutput(const Model& model, const IoConfig& cfg, templates_dir, out_dir, const DateTime& t);
void write_initial(names, values /* each G floats on the merged grid */);   // step 0, 70 messages per file
void write_step(int step_hours, const float* y /* (G, V_out) physical */);   // 67 messages per file
void close();                      // files appear under their final names only here (written as .part)
```
`out_dir` is the root of the configured relative paths (`out_dir/outputs/...`). The constructor reads
`VARDA_COSMO_DEFINITIONS` and calls `set_definitions_path` (the ICON templates need the COSMO shortNames).
A second constructor takes explicit `GridSource{name, mask}` (tests, `varda-grib-write`).

What it does, in anemoi's order:

1. Top-level post-processors on the whole state: `accumulate` (`acc = f32(acc + max(0, x))`, NaN kept as
   `np.maximum`, accumulator persistent across steps, replaces the field, registers `start_steps[tp] = 0`), then
   `rescale` for the fields whose mars `param` is `TOT_PREC` (`f32(acc * 1000 + 0)`, a new array).
2. Per output: `extract_from_state` (the source's slice of the merged grid), `extract_mask` (`as_slice`,
   `inverse`: `[n_lam, G)`), `assign_mask` (N320 array of NaN, values at the true positions in order).
3. Per field (computed forcings skipped): template lookup (first index entry whose every key's lookup value
   is in the accepted list; lookup = name, time_processing, mars `levtype/param/levelist`, encoding keys;
   cached per variable), `build_keys` (`encoding.py:grib_keys`: key order `edition typeOfLevel stepType
   productDefinitionTemplateNumber` first, then insertion order, None dropped), `apply_patches` (global),
   NaN -> -9999 with `missingValue`/`bitmapPresent` appended, `generatingProcessIdentifier=255` appended,
   one `set_key_vals` (falls back to key by key, as anemoi does), `values` as float64, write + flush per
   message, then `check_encoding` (every key read back; fails the run on mismatch).
4. File name from the encoded handle (`dateTime`, `step`), `{step}` unpadded.

Keys, e.g. (checked in `tests/io/test_io_output.cpp`):
```
lam t_500 step 6:   edition=2,typeOfLevel=isobaricInhPa,stepType=instant,typeOfGeneratingProcess=2,shortName=T,dataType=fc,date=20261022,time=1500,step=6,level=500
lam tp step 18:     edition=2,stepType=accum,productDefinitionTemplateNumber=8,typeOfGeneratingProcess=2,shortName=TOT_PREC,dataType=fc,date=20261022,time=1500,startStep=0,endStep=18
global z_500:       ...,shortName=z,...,step=6,level=500,param=129        (shortName replaced in place, param appended)
```
Not implemented (the exporter refuses them): hindcast modifier, ensembles, GRIB1, back-references in patches,
negative accumulation start steps (anemoi shifts the base time and then refuses to write; the C++ throws),
`stream` mars keys, `variables`/`output_frequency` filters.

## 3. Tools

- `varda-grib-compare TEST REF [--rel-tol X] [--abs-tol Y] [--require-bytes] [--ignore-key K,..]
  [--ignore-data-keys] [--max-report N] [-q]`: files or directories (same relative names). Per message in file
  order: name (`shortName/paramId/typeOfLevel/level/step`) and order, every header key (the ecCodes key
  iterator, arrays up to 10 000 elements; larger ones are values or geometry, counted as "big arrays skipped";
  `md5Headers` read explicitly), bitmap (positions missing in only one), values (count differing, bitwise,
  `max_abs`, `rel_absmax = max_abs / max|REF|` over points present in both; `docs/format.md` §2), and whether the
  message bytes are equal (first differing byte). Exit 0 only if everything is equal within the tolerances
  (default 0: bitwise), 1 on a mismatch, 2 on a usage or read error. It includes `eccodes.h` through
  `GribHandle::raw()` for the key iterator (see requests, §6).
- `varda-grib-write`: the encoder-only driver of §4 (reads the reference's dumps, writes both GRIB sets).

## 4. Encoder-only bitwise test (spec §5.2 C3-C6)

The reference run (`docs/realdata-reference.md`, `$VARDA_SCRATCH/realdata/20261022T15/ref-{fp16,fp32}`) dumps
`state0.pre` (initial state), `step<s>.y` (model output) and `state<s>.pre`, `state<s>.1.ForwardTransformFilter`
(tp after accumulate and x1000). `varda-grib-write` feeds `state0.pre` to `write_initial` and `step<s-1>.y` to
`write_step` (no model, no input side), and checks along the way: `y == state<s>.pre` fields bitwise, and tp
(`accumulate_into` + `rescale` on the dumped state) bitwise equal to the reference's processed tp (C3). The
output is compared with `varda-grib-compare OUT/outputs REF/outputs --require-bytes`.

| job | what | result |
|---|---|---|
| 285346 (cpu-short, 32 cores, 200 G) | `ref-fp16` dumps -> `realdata/20261022T15/cpp-encoder-ref-fp16/outputs` vs `ref-fp16/outputs` | 10 files, 676 messages (70 + 4 x 67, lam and global), 0 failing; **message bytes equal 676 / 676, values bitwise 676 / 676, key mismatches 0**; tp C3 checks bitwise at steps 1-4, `y == state.pre` bitwise; 72 s total, 3.2 s per step (both files) |
| 285350 (cpu-short) | the same for `ref-fp32` | 10 files, 676 messages, **bytes equal 676 / 676**, values bitwise 676 / 676, 0 key mismatches; C3 bitwise |
| 285350 | negative control: encoder output of the fp16 dumps vs the `ref-fp32` outputs | exit 1 as it must: step 0 files identical (140 messages), the 536 step messages differ (6395 key mismatches, mostly the value statistics `average`/`kurtosis`), worst `max_abs` 14.06, `rel_absmax` 1.09e-2 |

The encoder outputs `realdata/20261022T15/cpp-encoder-ref-{fp16,fp32}/` were deleted on 2026-10-23; `varda-grib-write`
regenerates them in about a minute (command above).

So the expectation of spec §5.2 holds: given the reference's own states, the C++ writer's files are
byte-identical to anemoi-inference's, including key order, the NaN bitmap of the global files, tp
accumulation/rescaling and the unpadded file names (`outputs/202610221500_6.grib`,
`outputs/global-202610221500_24.grib`). ecCodes prints 20 "Message validity checks: First fixed surface"
lines on stderr when it loads the IFS templates; they come from the template files, not from the encoding.

Unit tests, `tests/io/test_io_output.cpp` (ctest, `test_io_output`; the real-data parts skip without
`$VARDA_SCRATCH`, the HF templates and `VARDA_COSMO_DEFINITIONS`): 45 checks, all ok. Python-int/float parsing,
accumulate (negative clamp, float32 sums, NaN kept), rescale, `assign_mask`, NaN -> -9999 and +-inf -> +-FLT_MAX,
path rendering; the io config (sizes 98 / 80 / 67 / 70, orders, namer, `{step:03}` -> `{step}`); `build_keys` for
lam pl / sfc / tp / z at step 0 and global pl / tp; the 15 patch names to shortName + paramId; ICON and IFS
template matching for every kind of variable; and a synthetic end-to-end run on the real templates (1 147 980 +
540 670 points, 70 + 2 x 67 messages per set, tp clamp and accumulation, bitmap with 1410 missing, header keys).

## 5. Reproduce

```
source tools/env.sh
cmake -S . -B build-out && cmake --build build-out -j 32
$VARDA_SCRATCH/ref-oper/.venv/bin/python -I python/export_io_config.py
build-out/test_io_output
sbatch $VARDA_SCRATCH/jobs/io-output-encoder/run.sh ref-fp16      # binaries are copied to bin/ first
```

## 6. Requests to the integrator

- `GribHandle` additions would let `varda-grib-compare` stop using `raw()`: key iteration (names, native type,
  size, array getters) and `is_missing` on arrays.
- `docs/format.md` / `docs/design.md` pointers to this file; `spec-io.md` §4.3's file naming is confirmed by the
  reference outputs (`_6.grib`, unpadded).
