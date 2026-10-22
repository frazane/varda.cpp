# Contracts: converted model file, activation names, golden files

Interfaces between the parts of varda.cpp, so they can be built independently. Change a
contract only deliberately, and update every producer and consumer together.

The model family and its maths are specified in the Python project this engine follows
(`$IE_ROOT/docs/spec-model.md`, `spec-interface.md`, `varda.md`, `rollout.md`;
`$IE_ROOT` = `$HOME/projects/the Python project`). This file only fixes what is
specific to varda.cpp: the on-disk format and the names the C++ engine and the tools share.

## 1. Converted model file (`python/convert.py` → engine)

One file per model: `$VARDA_SCRATCH/converted/<model>/model.safetensors`.

The container is plain [safetensors](https://github.com/huggingface/safetensors):

```
[u64 little-endian N][N bytes UTF-8 JSON header, space-padded so that 8 + N ≡ 0 (mod 8)][data]
header = {"__metadata__": {str: str}, "<name>": {"dtype": "F32"|"I64"|..., "shape": [...],
          "data_offsets": [begin, end]}, ...}
```

- Offsets are relative to the start of the data section. Tensors are contiguous,
  row-major, little-endian, packed with no gaps (as the safetensors library requires), so
  the file is readable by `safetensors.safe_open` too.
- `__metadata__` has exactly these string entries:
  - `"varda_format"`: `"1"` (this section's version);
  - `"config"`: the model config as a JSON string (§1.2).
- The engine maps the file read-only (`mmap`) and never copies a tensor it does not need.

### 1.1 Tensors

Names, shapes and dtypes are exactly those of the Python engine's `weights.pt`, format v3
(`$IE_ROOT/docs/contracts.md` §1, table "Tensor names"): fp32 weights as stored,
`nodes.{data,hidden}.attr` [N, 4] f32, `graph.{enc,proc,dec}.edge_index` [2, E] **I64**
(row 0 source, row 1 destination, stored edge order), `graph.{enc,proc,dec}.edge_attr`
[E, 3] f32, `norm.mul` / `norm.add` [n_data_vars] f32, and per block (`enc.block`,
`proc.{i}`, `dec.block`) `q k v self edge proj mlp1 mlp2` (Linear: `.weight` [out, in],
`.bias` [out]) and `ln_dst ln_mlp` (+ `ln_src` in mapper blocks) (LayerNorm `.weight`,
`.bias`), plus `enc.emb_src enc.emb_dst dec.emb_dst dec.out_norm dec.out_proj`.

Bit-identity rule: for every converted model, each tensor in `model.safetensors` is
bitwise equal (same dtype, shape and bytes) to the tensor of the same name in the Python
engine's `weights.pt` for the same checkpoint, and the two have the same name set
(`python/tests/test_convert.py` checks this when `weights.pt` is available).

Loading is strict: the engine consumes every tensor in the file, and fails if one is left
over or missing (names listed in config `absent` are allowed to be missing).

### 1.2 Config

The `config` JSON has the content of the Python engine's `config.json`, format v3
(`$IE_ROOT/docs/contracts.md` §1: `format_version`, `source`, `architecture`, `dims`,
`variables`, `indices`, `residual`, `bounding`, `mappers`, `blocks`, `absent`, `unused`),
with identical values except `source.checkpoint` (the path converted). Every value is read
from the checkpoint by the converter, never hard-coded; the engine reads sizes, epsilons,
activation variants, index lists and bounding from it and asserts nothing model-specific.

The engine rejects unknown `architecture.*.kind`, block `kind`, `residual.type`,
`bounding[].type` or `dims.gelu_approximate` values at load time with a clear error.

## 2. Activation names (golden data ↔ engine debug hook)

The same keys as `$IE_ROOT/docs/contracts.md` §2: `in.x`, `in.x_norm`, `enc.x_src`,
`enc.x_dst`, `enc.emb_src`, `enc.emb_dst`, `enc.out`, `proc.{i}.out` (i = 0..n_proc−1),
`latent`, `dec.emb_dst`, `dec.block_out`, `dec.out`, `out.pre_bound`, `out.norm`, `out.y`,
with the shapes given there (B = 1). The engine's `ActivationHook` (`src/varda/varda.h`)
receives each key exactly once per `predict_step`, as host fp32, contiguous, with that
shape. On the GPU in fp16 the tensor is upcast to fp32 for the hook.

Metric (identical to the Python project): per key, in float64,
`max_abs = max|a − r|`, `absmax = max|r|`, `rel_absmax = max_abs / absmax` (0 if both are
0, inf if only `absmax` is 0). NaN anywhere in `a` is a failure. Bitwise means
`max_abs == 0`. Elementwise relative error is never used.

## 3. Golden files (`python/export_golden.py`)

The Python project's golden directories (`$IE_SCRATCH/golden/<tag>/activations.pt` +
`manifest.json`) are exported once to `$VARDA_SCRATCH/golden/<tag>.safetensors`:
every key of `activations.pt` as F32 with its shape, and `__metadata__` =
`{"varda_golden": "1", "tag": <tag>, "manifest": <manifest.json as a string>,
"source": <abs path of activations.pt>, "source_size": <bytes>, "checkpoint_uuid": <uuid>}`.
`checkpoint_uuid` pairs the golden data with a model (`source.uuid` in its config): it is
`manifest.checkpoint_identity.uuid` if present, else the uuid in the embedded
`anemoi-metadata/anemoi.json` of the checkpoint the manifest names (old manifests; same
fallback as `$IE_ROOT/engine/compare.py`). Consumers refuse to compare unpaired files.

## 4. Tolerances (`tolerances/*.json`)

Copied unchanged from the Python project (`$IE_ROOT/tolerances/`), keyed by checkpoint
uuid (`checkpoint.uuid`). Tables: `cuda-fp32` (engine fp32 vs `cpu-fp32-pyg`), `cuda-fp16`
(engine fp16 vs `cpu-fp32-pyg`), `cuda-fp16-vs-ref-fp16` (engine fp16 vs
`cuda-fp16-triton`). The C++ CPU fp32 path is held to the `cuda-fp32` table (it is the
reference's own cross-device fp32 spread × 3, floored at 1e-5): a from-scratch fp32
implementation with different GEMM kernels is exactly that kind of neutral perturbation.
Never loosen a tolerance to make a test pass without saying so explicitly.
