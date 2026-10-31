# Tools: converting a checkpoint

## Convert

```
.venv/bin/python -I python/convert.py CKPT OUT.safetensors
```
`-I` is required by the repository rules; `convert.py` adds its own directory to `sys.path`.
Loading is mmap-based but run it on a compute node (`srun $VARDA_SLURM_CPU`; 16 cores and 64 GB suffice; it
takes about 20 s plus the sha256 of the 1.0-1.5 GB checkpoint). The output is written under a
temporary name and renamed, so readers never see a partial file.

Files: `python/checkpoint.py` (restricted unpickler and stub-tree walker), `python/convert.py`
(mapping table, checks, config), `python/safetensors_io.py` (writer, reader, container validator;
numpy/torch/stdlib only), `python/tests/test_convert.py`.

## How the pickle is restricted

`torch.load(..., pickle_module=...)` is given an `Unpickler` whose `find_class` returns real objects only
for: torch tensor/parameter rebuilders (`_rebuild_tensor_v2`, `_rebuild_parameter[_with_state]`),
`torch.Size/device`, `torch.<dtype>`, the plain containers and scalar types of `builtins`/`collections`,
numpy array reconstruction, `_codecs.encode`, and a `getattr` that only resolves attributes of stubs.
Every other global (anemoi, torch.nn, torch_geometric, omegaconf, ...) becomes an inert `Stub` subclass that
records its qualified name, constructor args, state and items; no foreign code runs. Any other
`torch*._rebuild*` function raises (a stub would silently drop tensors). The zip's
`anemoi-metadata/anemoi.json` is read as plain JSON.

## What is checked

- Supported classes up front (model, encoder/processor/decoder, block, residual, bounding), one dataset
  `data`; `qk_norm` / `update_src_nodes` rejected; Linear/LayerNorm dtype fp32; LayerNorm
  weight present (a missing bias goes to `absent`).
- Model kinds (`docs/format.md` section 1.2): `AnemoiModelEncProcDec` -> `enc_proc_dec`, which must have one output step;
  `AnemoiModelEncProcDecMultiOutInterpolator` (the temporal downscaler) -> `enc_proc_dec_interpolator`, the only kind
  allowed `n_step_output` > 1. For it the converter writes `dims.n_output_steps` = `n_step_output` and `times` from the
  model's `input_times` / `output_times`, which must equal `config.training.explicit_times`, the
  `metadata_inference` relative date indices, have lengths `n_step_input` / `n_step_output`, be strictly increasing,
  and come with a whole-hour data frequency equal to the metadata timestep.
- Two pickle layouts of anemoi-models are read (both in Varda-single-1.0): `model.node_attributes` a
  NamedNodesAttributes and `model._graph_data` a HeteroData (forecaster, anemoi-core 3c5fc21), or a ModuleDict /
  dict per dataset (downscaler, anemoi-core d9cf401, which also has no block `attn_channels`: the query projection
  width is used). The tensors and the config they produce have the same names and meaning.
- Strict accounting: every tensor reachable in the checkpoint (deduplicated by identity) is converted,
  exported to the config (indices, bounding), a duplicate asserted equal to a converted one (HeteroData
  copies of the graphs), or listed in `config.unused` with a reason; otherwise the conversion fails.
- Cross-checks: dimensions vs weights shapes (`enc.emb_src` / `dec.emb_dst` in = n_input_steps * n_input_vars +
  node attr, `dec.out_proj` = (n_output_steps * n_output_vars, C) and its bias), graph `edge_index` range vs
  `edge_inc`, index lists vs the IndexCollection and variable names, identical block hyperparameters, one LayerNorm
  eps, bounding variables vs indices, normaliser buffers vs raw statistics (decides the `unused` reason text).
- Forcings (`build_forcings`, as anemoi-inference classifies them, for both runners): from the metadata categories
  `computed_forcing` / `constant_in_time`, never from names; the metadata's input / output relative date indices must
  number `n_input_steps` / `n_output_steps`.
- Identity in `source`: path, size, sha256, mtime, uuid, run_id, interface id, anemoi-core commit.

Container: tensors ordered 8-byte dtypes (I64) first, then by name; no gaps; header JSON space-padded so the
data section is 8-byte aligned; `__metadata__` = `{"varda_format": "1", "config": <json.dumps(config)>}`.

## Test

```
VARDA_CONVERTED=a.safetensors:b.safetensors VARDA_WEIGHTS_PT=a/weights.pt:b/weights.pt \
  .venv/bin/python -I python/tests/test_convert.py
```
Checks container validity, the config fields `src/varda/config.cpp` reads, the exact tensor name set
(from the contract table, driven by the config's blocks and `absent`, plus `grid.*` and the cutout masks), dtypes/shapes
(including `dec.out_proj` = n_output_steps * n_output_vars rows), bitwise equality with
`weights.pt` and equality of the config with the sibling `config.json` (except `source.checkpoint`/`mtime`),
the interpolator's `times` / multi-step head and, for the Varda downscaler, its values; plus restricted-unpickler tests
(a pickled `os.system` is stubbed, `getattr` on non-stubs and unknown rebuild functions are rejected) and
`interpolator_times` tests (every metadata disagreement is rejected). File-based tests are skipped when the env vars
are unset. An empty `VARDA_WEIGHTS_PT` entry skips the bitwise test for that file (the downscaler has no independent
weight export), e.g. for the three models:

```
S=$VARDA_SCRATCH/converted; W=$REF_SCRATCH/converted   # weights.pt: the directories of the reference weight exports
VARDA_CONVERTED=$S/varda-td/model.safetensors:$S/varda/model.safetensors:$S/dev/model.safetensors \
VARDA_WEIGHTS_PT=:$W/<varda>/weights.pt:$W/<dev>/weights.pt \
VARDA_CKPTS=$VARDA_SCRATCH/hf/varda-temporal-downscaler-det-sgm/model.ckpt:$VARDA_CKPT:$VARDA_DEV_CKPT \
  srun $VARDA_SLURM_CPU .venv/bin/python -I python/tests/test_convert.py
```

## Verification (2026-10-22; one conversion per model and one test run, CPU node)

Outputs in `$VARDA_SCRATCH/converted/<model>/model.safetensors` (`varda`, `dev`). The table is for the first
conversion of Varda; the file in use, `converted/varda/`, was converted later with the cutout masks the GRIB layer
needs (`docs/format.md` section 1.1) and every other tensor is bitwise equal to the first one's.

| | Varda (HF checkpoint) | dev model |
|---|---|---|
| checkpoint tensors / converted / unused | 429 / 384 / 45 | 437 / 384 / 53 |
| tensor bytes | 1 278 572 744 | 993 932 184 |
| file bytes | 1 278 627 808 | 993 988 256 |
| vs `weights.pt` (names, dtypes, shapes, bytes) | identical | identical |
| config vs `config.json` | equal except `source.checkpoint` (HF path) and `source.mtime` | equal |
| vs current bootstrap repack, per tensor | bitwise equal | bitwise equal |
| data section sha256 | 64d6b456...1a3abc (same as bootstrap) | 2edea094...0a2f (same as bootstrap) |
| whole-file sha256 | 8e2fd45a05a84c4818f1e27b47321be624257be684d5a6a8abfeda97524a8e4b | 6973e5e04cec80c578d54f01d0d27c9dc943a42d838baa32484fed28bebfb189 (identical to bootstrap file) |

The Varda file differs from the bootstrap only in the header (config `source.checkpoint`/`mtime`: the
bootstrap recorded the path and mtime of an earlier copy of the same checkpoint file; the HF download has the
same sha256). Varda's raw `statistics` are not the ones folded into the normaliser buffers (max rel.
difference 28.4), as in the reference; the `unused` reason says so.

### Temporal downscaler (`varda-td`, 2026-10-30; CPU node)

`$VARDA_SCRATCH/converted/varda-td/model.safetensors` from the Hugging Face file
`varda-temporal-downscaler-det-sgm/model.ckpt` (sha256 452fda53...a21d0e, uuid `00e40f03-2f7e-4c1c-8e6d-91ed9062249f`).
There is no independent weight export for it; `test_convert.py` checks container, config, names, shapes, grid and the
downscaler's values (all 15 tests pass on the three converted files).

| | Varda temporal downscaler |
|---|---|
| checkpoint tensors / converted / unused | 429 / 384 / 45 (388 tensors in the file with `grid.*` and the two cutout masks) |
| tensor bytes / file bytes | 1 299 947 436 / 1 300 003 628 |
| `dims` | 2 input steps, 6 output steps, 77 in / 67 out / 78 data variables, C 1024, 16 heads x 64, 16 layers, `latent_skip` false |
| `times` | input [0, 6], target [1..6], 1 h; `forcings.timestep_hours` 1 |
| head | `dec.out_proj.weight` (402, 1024) = 6 x 67 rows; `enc.emb_src.weight` (1024, 158) = 2 x 77 + 4 |
| residual / bounding / mappers | skip step 0 / relu(tp) / encoder and decoder 16 chunks |
| forcings | dynamic cos/sin_julian_day, cos/sin_local_time, insolation; constant cos/sin_latitude, cos/sin_longitude, lsm, z |
| normaliser buffers vs raw statistics | max rel. difference 4.61: the raw statistics are not the normaliser's source (as for the forecaster) |
| data section sha256 | e8fda2a13b38f2e622050fcc8c0bedb6fee083b3780a8d89af30406c2a48a45e |
| whole-file sha256 | 9b6c6e77481daa4405f8308c2b98d942224d70ac3b81f7c6e92e7842d27073e9 |
| time | 20 s wall clock, 2.9 GB peak RSS |

Regression: with the extended converter, the forecaster and the dev model re-convert to files whose whole-file sha256
equals the files in use (`converted/varda/`, `converted/dev/`): no change to their tensors or configs.

## User tools (real-data forecasts)

- `varda` (repo root): the entrypoint, `./varda setup | fetch | forecast` (`docs/quickstart.md`).
- `python/fetch_inputs.py`: downloads one date's inputs (KENDA-CH1 variables and only the needed IFS open-data
  messages, as byte ranges) into the `--realdata` layout; standard library only. Shares `python/opendata.py` with the
  reference fetcher `python/reference/realdata/fetch_realdata.py`.
- `tools/make_assets.py check | link | build`: verifies, assembles (hard links from verified files) or rebuilds from the
  Hugging Face download the assets bundle that `varda-forecast --assets` reads (`docs/quickstart.md` §4).

