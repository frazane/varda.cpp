# Tools: converting a checkpoint

## Convert

```
.venv/bin/python -I python/convert.py CKPT OUT.safetensors
```
`-I` is required by the repository rules; `convert.py` adds its own directory to `sys.path`.
Loading is mmap-based but run it on a compute node (`srun -p cpu-short -A $ACCOUNT -c 16 --mem=64G`; it
takes about 20 s plus the sha256 of the 1.0-1.5 GB checkpoint). The output is written under a
temporary name and renamed, so readers never see a partial file.

Files: `python/checkpoint.py` (restricted unpickler and stub-tree walker), `python/convert.py`
(mapping table, checks, config), `python/safetensors_io.py` (writer, reader, container validator;
numpy/torch/stdlib only), `python/tests/test_convert.py`. `convert.py` and `checkpoint.py` are a port of
`$IE_ROOT/engine/{convert,checkpoint}.py`; only the output container differs.

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
  `data`, one output step; `qk_norm` / `update_src_nodes` rejected; Linear/LayerNorm dtype fp32; LayerNorm
  weight present (a missing bias goes to `absent`).
- Strict accounting: every tensor reachable in the checkpoint (deduplicated by identity) is converted,
  exported to the config (indices, bounding), a duplicate asserted equal to a converted one (HeteroData
  copies of the graphs), or listed in `config.unused` with a reason; otherwise the conversion fails.
- Cross-checks: dimensions vs weights shapes, graph `edge_index` range vs `edge_inc`, index lists vs the
  IndexCollection and variable names, identical block hyperparameters, one LayerNorm eps, bounding
  variables vs indices, normaliser buffers vs raw statistics (decides the `unused` reason text).
- Identity in `source`: path, size, sha256, mtime, uuid, run_id, interface id, anemoi-core commit.

Container: tensors ordered 8-byte dtypes (I64) first, then by name; no gaps; header JSON space-padded so the
data section is 8-byte aligned; `__metadata__` = `{"varda_format": "1", "config": <json.dumps(config)>}`.

## Test

```
VARDA_CONVERTED=a.safetensors:b.safetensors VARDA_WEIGHTS_PT=a/weights.pt:b/weights.pt \
  .venv/bin/python -I python/tests/test_convert.py
```
Checks container validity, the config fields `src/varda/config.cpp` reads, the exact tensor name set
(from the contract table, driven by the config's blocks and `absent`), dtypes/shapes, bitwise equality with
`weights.pt` and equality of the config with the sibling `config.json` (except `source.checkpoint`/`mtime`),
plus restricted-unpickler tests (a pickled `os.system` is stubbed, `getattr` on non-stubs and unknown
rebuild functions are rejected). File-based tests are skipped when the env vars are unset.

## Verification (2026-10-22, jobs 283263 dev, 283264 varda, tests 283269; logs in
`$VARDA_SCRATCH/jobs/convert/`)

Outputs in `$VARDA_SCRATCH/converted/<model>/model.safetensors` (Varda with cutout masks: `converted/varda-v2/`; `converted-new/` was a byte-identical copy of `converted/` and was deleted on 2026-10-23).

| | Varda (varda, HF ckpt) | dev |
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
bootstrap recorded the `$VARDA_CKPT` path and its mtime; the HF copy has
the same sha256). Varda's raw `statistics` are not the ones folded into the normaliser buffers (max rel.
difference 28.4), as in the reference; the `unused` reason says so.
