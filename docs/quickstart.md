# Quick start: a Varda forecast from real data

Three commands, on this cluster:

```
source tools/env.sh                 # optional: ./varda sources it itself
./varda setup                       # once: build, check (or create) the assets bundle
./varda forecast latest             # or a date: ./varda forecast 2026-10-22T15
```

The result is a directory of GRIB files, `./varda-out/<YYYYmmddTHH>/outputs/`, plus `forecast.json`
(provenance and timings) and `varda-forecast.log`. A one-screen summary is printed at the end.

`./varda forecast` options:

| option | default | |
|---|---|---|
| `DATE` or `latest` | | initial time t in UTC, `YYYY-MM-DDTHH`, hour 00, 03, ..., 21. `latest` is the newest KENDA-CH1 analysis with t−6h also published (the rule of the Hugging Face notebook) |
| `--lead-time H` | 24 (the HF config's) | hours, a multiple of 6; `48h` and `5d` also work |
| `--precision fp16\|fp32` | fp16 | fp16 is the checkpoint's precision (`16-mixed`, what anemoi-inference runs) |
| `--device cuda\|cpu` | cuda | CPU: about 60 s per step (fp32) |
| `--out DIR` | `./varda-out/<YYYYmmddTHH>` | |
| `-- ...` | | any further `varda-forecast` option (`docs/forecast.md` section 2), e.g. `-- --allow-nans` |

`./varda fetch DATE|latest` downloads the inputs of a date without running anything (for example to keep a date
before it leaves the open-data servers).

## 1. What each command does

**`./varda setup`**
1. Builds `build/` with CMake (CUDA and GRIB on; about a minute on the login node), unless `build/varda-forecast` exists
   (`./varda setup --rebuild` forces it). The log is `build/varda-build.log`.
2. Checks the assets bundle against its manifest (sha256 of every file, about a minute). If there is no bundle, it
   assembles one from the verified files already on this cluster (`tools/make_assets.py link`: hard links, each
   file's sha256 checked against its source, a few minutes); elsewhere it says how to build one from scratch
   (section 4).

**`./varda forecast DATE`**
1. On the current node (the login node has network): finds the date's input directory in the input cache, or
   downloads it (`python/fetch_inputs.py`, section 2). `latest` asks the STAC API which date that is.
2. Copies `build/varda-forecast` to a private directory (`$VARDA_SCRATCH/jobs/varda-forecast-<date>-XXXXXX/`,
   removed afterwards), so a rebuild during the run cannot change the binary that runs.
3. Runs it on a GPU node: through `srun` when not inside a Slurm allocation (default
   `-p debug -A $ACCOUNT --gres=gpu:1 -c 32 --mem=200G --time=00:30:00`; `VARDA_SRUN_ARGS` replaces it), directly inside
   an allocation (`salloc`, `sbatch`). It runs `varda-forecast --assets BUNDLE --realdata INPUTDIR --out DIR`.
4. Prints the summary from `forecast.json`: files, sizes, the IFS runs used, timings.

## 2. Where things go

| what | where | size |
|---|---|---|
| build | `build/` (`VARDA_BUILD`) | |
| assets bundle | `$VARDA_SCRATCH/assets/varda-single-1.0/` (`VARDA_ASSETS`) | 1.35 GB |
| input cache, one directory per date | `$VARDA_SCRATCH/inputs/<YYYYmmddTHH>/` (`VARDA_INPUTS`) | about 2.5 GB per date |
| forecast | `./varda-out/<YYYYmmddTHH>/` (`--out`) | 1.28 GB for 24 h (10 files); about 0.25 GB per further 6 h step |

Nothing is deleted automatically: remove old input dates and outputs yourself.

**Assets bundle** (everything derived once per model release):

```
model.safetensors                   the converted checkpoint with the cutout masks (python/convert.py)
varda-io.json                       variables, GRIB output keys, template index (python/export_io_config.py)
regrid-ifs0p25-n320.safetensors     MIR's IFS 0.25 deg -> N320 linear matrix (python/export_regrid_matrix.py)
templates/                          templates/ of the HF repo (ICON and IFS GRIB samples)
definitions/                        ecCodes definitions of eccodes-cosmo-resources 2.44.0.1 (KENDA short names)
manifest.json                       sha256 and size of every file, the sources, checkpoint sha256, tool versions
```

`varda-forecast --assets DIR` takes the five files from it (an explicit `--model`, `--io-config`,
`--regrid-matrix`, `--templates` or `--definitions` overrides its entry). `tools/make_assets.py check` verifies a
bundle.

**Input directory** (`python/fetch_inputs.py`; the layout `varda-forecast --realdata` reads, the same as the frozen
reference cache of `docs/realdata-reference.md`):

```
cache/kenda/raw/<t-6h>/<VAR>.grib2, cache/kenda/raw/<t>/<VAR>.grib2   the 11 KENDA-CH1 variables the run reads
                                                                       (P PMSL PS QV T T_2M TD_2M U U_10M V V_10M), 2.23 GB
cache/kenda/constants/{horizontal,vertical}_constants_icon-ch1-eps.grib2   ICON-CH1 constants, 0.2 GB
cache/kenda/stac/*.json                                                the STAC documents (provenance)
cache/ecmwf/files/<run>-<step>h-oper-fc.subset.grib2 (+ .index)         only the IFS messages used (66 at t-6h,
                                                                       66 at t, 4 constants), about 60 MB
pins.json                                                              the choices made at download time
```

- **IFS messages, not files.** The open-data `.index` gives each message's byte range; only the messages the run
  reads are downloaded (about 60 MB instead of 200-300 MB), exact bytes of the published file. `x0` is bitwise the same as from
  the full files (section 6). `--full-ecmwf` downloads whole files.
- **The ECMWF runs are chosen at download time and recorded** in `pins.json`, as anemoi-inference's
  `oper-ecmwf-opendata` chooses them at start time: the latest published run (`ecmwf_latest_published_run`), the run
  and step for t−6h and t (`input_dates`: the analysis when its run is already published, else a forecast of the
  latest run; at 03/09/15/21Z always a forecast, the 3 h one of the run before), and
  **the run of the global constants** (`constant_forcings_run`: step 0 of the latest run, since the config marks them
  `constant: true`). `fetch_inputs.py --constants-run RUN` names another run; `--ecmwf-latest RUN` pins "latest".
- **A directory with `pins.json` is complete** (it is written last) and is reused without network, so a forecast can be
  rerun later on exactly the same inputs. To fetch a date again, remove its directory.

## 3. How long it takes

Measured on this cluster on 2026-10-23 (section 6):

| step | time |
|---|---|
| `./varda setup` the first time: build + bundle check | 1 min 57 s (about 1 min each) |
| `make_assets.py link` (bundle creation, hard links, sha256 of every file) | 3 min 6 s |
| fetch of one date (2.50 GB, login node) | 22 s |
| `srun -p debug` allocation | seconds when the partition is free (it was here) |
| forecast, 24 h fp16 on an A100 | now 8.3-8.7 s on a node where the files are not cached yet, 7.8 s warm (job 286500, `docs/results.md` section 5). Historical (commit a9dcc66): 20-21 s cold, 14 s of it loading the weights, about 10.5 s warm |
| `./varda forecast latest` from the login node, all of it | 50 s (fetch 22 s, forecast 21 s, the rest srun and the summary; measured with the previous binary, the forecast part is now about 8.5 s cold) |

## 4. Another machine

What is specific to this cluster, and how to replace it:

| | here | elsewhere |
|---|---|---|
| toolchain | `tools/env.sh` (gcc 12.3, CUDA 12.3, OpenBLAS, `ECCODES_HOME`) sourced by `./varda` | write your own env file and set `VARDA_ENV=/path/env.sh`, or `VARDA_ENV=none` with the variables already exported. The build needs CMake >= 3.24, a C++17 compiler, OpenMP, OpenBLAS (`OPENBLAS_HOME`), CUDA (optional) and ecCodes (`ECCODES_HOME` with `eccodeslib/` and `eckitlib/`, e.g. the `eccodeslib` and `eckitlib` wheels' contents) |
| GPU | A100, `CMAKE_CUDA_ARCHITECTURES=80` | other GPUs: build with `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<arch>` before `./varda setup` |
| data root | `VARDA_SCRATCH=$VARDA_SCRATCH` (from `tools/env.sh`) | without `VARDA_SCRATCH`: `~/.cache/varda/`; or set `VARDA_ASSETS`, `VARDA_INPUTS` |
| compute | `srun` with defaults `-p debug -A $ACCOUNT --gres=gpu:1 -c 32 --mem=200G --time=00:30:00` (`--device cpu`: `-p cpu-short -A $ACCOUNT -c 64 --mem=200G --time=01:00:00`) | other Slurm: set `VARDA_SRUN_ARGS`; no Slurm, on a GPU machine: `VARDA_SRUN=0` |
| python | `.venv/bin/python` | any python >= 3.9 (`VARDA_PYTHON`); `fetch_inputs.py` and `make_assets.py check/link` use only the standard library |
| assets | `make_assets.py link` from the files verified here | `make_assets.py build` (below), or copy a bundle (rsync the directory; `make_assets.py check` verifies it) |
| network | login node | the fetch step runs where `./varda` is started; it needs `data.geo.admin.ch` and `storage.googleapis.com` (`fetch_inputs.py --source aws\|ecmwf` for another ECMWF mirror) |

### Assets from scratch

`tools/make_assets.py build` makes the bundle from the Hugging Face download with the project's tools:

```
# 1. HF = $VARDA_SCRATCH/hf: the forecaster, and templates/ with REVISION.json (the HF revision, which
#    export_io_config.py records); fetch_realdata.py templates writes $VARDA_SCRATCH/hf/templates/ (standard library)
huggingface-cli download MeteoSwiss/Varda-single-1.0 --include 'varda-forecaster-det-sgm/*' --local-dir $VARDA_SCRATCH/hf
.venv/bin/python -I python/reference/realdata/fetch_realdata.py templates
# 2. two Python environments: .venv (torch CPU, numpy: the converter) and the reference env (anemoi-inference,
#    mir-python, eccodes, eccodes-cosmo-resources; python/reference/realdata/requirements-ref-oper.lock.txt,
#    docs/realdata-reference.md section 3)
# 3. an IFS 0.25 deg open-data file for the source geometry of the regrid: any input directory has some
./varda fetch latest
# 4. on a compute node (the converter reads the 1.5 GB checkpoint; 3 GB peak)
srun -p cpu-short -A $ACCOUNT -c 16 --mem=32G --time=00:30:00 .venv/bin/python -I tools/make_assets.py build \
    --hf $VARDA_SCRATCH/hf --dest DIR --grib $VARDA_INPUTS/<date>/cache/ecmwf/files/<file>.subset.grib2 \
    [--py .venv/bin/python] [--ref-py $VARDA_SCRATCH/ref-oper/.venv/bin/python]
```

It runs `python/convert.py`, `python/export_io_config.py`, one MIR call on one open-data message (which writes MIR's
interpolation matrix into a private MIR cache) and `python/export_regrid_matrix.py` on that cache, then copies
`templates/` and the definitions of the reference env's `eccodes-cosmo-resources`, and writes the manifest. It took
2 min 24 s on a cpu-short node; the result equals the bundle here (section 6).

## 5. Troubleshooting

| message | cause, fix |
|---|---|
| `KENDA-CH1 analysis ... is incomplete: nothing is published for it` | KENDA-CH1 open data are kept about 24 h. Use `latest`, or a date already in the input cache |
| `... N of the analysis' variables are published, not ...` | the analysis is being published (retry in a few minutes) or expiring |
| `ECMWF ...: HTTP 404` | the IFS run is not (or no longer) published; ECMWF open data keep about 4 days |
| `older than the 12 ECMWF runs kept` | the date is more than 3 days before the latest IFS run |
| `the initial time must be ... hour % 3 == 0` | KENDA-CH1 is hourly but the IFS open data are 3-hourly |
| `no GPU visible on <node>` | inside an allocation without a GPU: request `--gres=gpu:1`, run `./varda` outside the allocation (it then uses `srun`), or `--device cpu` |
| `no assets bundle` / `bundle is incomplete or altered` | `./varda setup`; for a damaged bundle move the directory away first |
| `has pins.json but misses N file(s)` | an input directory was changed by hand: remove it and fetch again (if the date is still published) |
| `NaN in ...` from varda-forecast | an input field has NaN; the reference allows it (`-- --allow-nans`) |
| `srun: error: ...` | partition, account or limits: set `VARDA_SRUN_ARGS` (e.g. `-p gpu-normal -A <acct> --gres=gpu:1 -c 32 --mem=200G`) |

## 6. Verification

Recorded on 2026-10-23; jobs, logs and outputs under `$VARDA_SCRATCH/jobs/quickstart-verify/`. All GPU runs:
A100-SXM4-80GB, `-c 32`, fp16, device rollout, CUDA graph on, 4 steps.

**Frozen date 2026-10-22T15** (job 285865, <node>; `VARDA_INPUTS=$VARDA_SCRATCH/realdata`, so the input directory
is the frozen reference cache, read only):

| run | what | total |
|---|---|---|
| A | `./varda forecast 2026-10-22T15` (inside the job: runs directly, `--assets`, `--quiet`) | 19.95 s (cold node) |
| B | `build-int/varda-forecast` (the version before `--assets`) with the explicit options | 10.61 s |
| C | `build/varda-forecast` (this version) with the explicit options | 10.40 s |

- A, B and C: the 10 GRIB files are **byte-identical** (10/10 `cmp`, 1.28 GB, 676 messages).
- The 2 step-0 files of A are byte-identical to anemoi-inference's (`ref-fp16/outputs/*_0.grib`).
- **x0 from IFS subsets = x0 from full files.** The ECMWF messages of this date were fetched again as byte ranges
  (`fetch_inputs.fetch_ecmwf`, the pinned 18z as "latest"), next to the frozen KENDA cache. The three subset files
  (66, 66 and 4 messages, 61 MB) equal the same byte ranges of the frozen full files, and the `.index` files equal the
  frozen ones. `varda-prepare` gives x0 fnv1a `b840f06bc289bc1f` and initial-state fnv1a `099fe37f726f2874` from
  both.

**`./varda forecast latest`** for real, from the login node, on 2026-10-23 at 10:28:57Z:
- **Date and runs.** latest = **2026-10-23T06** (a 00/06/12/18Z date, not 15Z). The latest ECMWF run was 00z, so:
  - t−6h = 00Z comes from 00z+0h (an analysis);
  - t = 06Z from 00z+6h (the 06z run was not published yet: a 6 h forecast, as anemoi-inference would take at that
    time);
  - the constants from 00z+0h, the same file as t−6h (its subset holds 66 + 4 = 70 messages).
- **Download.** 2.50 GB in 22.4 s: KENDA 2.23 GB, constants 0.21 GB, IFS 62 MB of messages out of 198 MB of files.
- **Forecast.** srun job 285866 (debug, <node>): 21.0 s in all; ready after 17.2 s (input 17.2 s, weights
  13.7 s cold); rollout 3.1 s (726, 714, 715, 714 ms per step); 0.7 s of writing after the last step. 10 files,
  1.28 GB. Whole command: 50.4 s.
- **x0 from subsets = x0 from full files** (job 285867): with the two full IFS files downloaded next to the same
  KENDA files, `varda-prepare` gives x0 fnv1a `82e28fa24c047b13` (initial state `2850bf749f92fb44`) in both cases.
- **Reruns are bitwise.** The same date rerun inside job 285867 (cached inputs, no download) gives byte-identical files
  (10/10, 16.2 s). A 12 h rerun from the login node (srun job 285868) gives 6 files byte-identical to the first run's
  steps 0-12.
- **No parity claim.** As for any date other than 2026-10-22T15 (`docs/forecast.md` section 5), there is no
  anemoi-inference run to compare with.

**Final script, end to end** (after the last edits to `./varda` and `fetch_inputs.py`): `./varda forecast 2026-10-22T15`
from the login node through srun (job 285942, <node>, 21.0 s, exit 0) gives 10 files byte-identical to run B above.
A fresh fetch of 2026-10-23T06 into another root (ECMWF latest pinned to the same 00z) gives 28 GRIB and index files
byte-identical to the first download (2.50 GB, 22 s).

**`fetch_realdata.py` after the refactor** (it now imports `python/opendata.py`): `pick` gives the same date as
`fetch_inputs.py latest` (2026-10-23T06); `templates` into a scratch root gives files identical to the frozen
`$VARDA_SCRATCH/hf/templates`; `ecmwf --date 2026-10-23T06` (still `ecmwf.opendata.Client`, reference env) chooses the
same latest run, input runs, constants run and URLs as `fetch_inputs.py`, and its full files equal the ones
downloaded for the x0 check.

**Error paths** (login node): an expired date (2026-10-21T15: "nothing is published for it"), a non-3-hourly hour, a
malformed date, a bad option and a missing bundle all fail before anything is downloaded or created. Inside a CPU
allocation (cpu-short, job 285872), `./varda forecast` stops with "no GPU visible on <node>" before fetching.

**ctest executables** of `build/` (job 285867, copied to the job directory): 11/11 exit 0.

**Assets from scratch** (job 285919, cpu-short, <node>: `make_assets.py build --hf $VARDA_SCRATCH/hf --grib
<frozen 12z+3h file>`, 2 min 24 s, 3.0 GB peak), compared with the bundle assembled from the verified files:

| file | from scratch vs bundle |
|---|---|
| `model.safetensors` | byte-identical (same checkpoint path: the converter records path, size, mtime and sha256) |
| `varda-io.json` | byte-identical |
| `templates/`, `definitions/` | identical (`diff -r`) |
| `regrid-ifs0p25-n320.safetensors` | the three tensors bitwise equal; MIR's `.mat` sha256 equal (`1e5fac3b…aed533`). The metadata differs in three provenance keys: `run` (the directory MIR's cache was in), `mat_files_sha256` (one `.mat` here, two copies in the reference run) and `verification` (absent: the bitwise check against MIR needs the reference run's dumps, `docs/realdata-reference.md` section 7) |

So a bundle built from scratch gives the same forecast bytes (the engine reads the regrid tensors and the source
geometry, which is the same). From another checkpoint path, `model.safetensors` differs in its `source` metadata only.
