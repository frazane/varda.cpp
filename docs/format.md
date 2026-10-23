# Contracts: converted model file, activation names, golden files

Interfaces between the parts of varda.cpp, so they can be built independently. Change a
contract only deliberately, and update every producer and consumer together.

The maths is anemoi's GraphTransformer encoder–processor–decoder (`docs/cpu.md` gives the
op sequence). This file only fixes what is specific to varda.cpp: the on-disk format and the
names the C++ engine and the tools share.

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

Names, shapes and dtypes (format v3): fp32 weights as stored,
`nodes.{data,hidden}.attr` [N, 4] f32, `graph.{enc,proc,dec}.edge_index` [2, E] **I64**
(row 0 source, row 1 destination, stored edge order), `graph.{enc,proc,dec}.edge_attr`
[E, 3] f32, `norm.mul` / `norm.add` [n_data_vars] f32, and per block (`enc.block`,
`proc.{i}`, `dec.block`) `q k v self edge proj mlp1 mlp2` (Linear: `.weight` [out, in],
`.bias` [out]) and `ln_dst ln_mlp` (+ `ln_src` in mapper blocks) (LayerNorm `.weight`,
`.bias`), plus `enc.emb_src enc.emb_dst dec.emb_dst dec.out_norm dec.out_proj`.

In addition (not in `weights.pt`; optional, older files lack them):
`grid.latitudes`, `grid.longitudes` F64 [n_data_nodes], degrees: the checkpoint's supporting
arrays (zip members `*/anemoi-metadata/data/{latitudes,longitudes}.numpy`) that
anemoi-inference uses to compute forcings, checked against `nodes.data.attr` by the
converter (`docs/forcings.md`). `Model::load` consumes them; the rollout uses them.
For a multi-source (cutout) dataset, also `grid.<source>.cutout_mask` U8 0/1 [n_source_points]
per source (the checkpoint's `<source>/cutout_mask` supporting arrays), with metadata key
`cutout` = JSON list of the source names in data-grid order (Varda: `["lam_0", "global"]`).
The data grid is the concatenation of every source's kept points in that order; the converter
checks this bitwise against `source<i>/latitudes|longitudes` and the data coordinates.
`Model::cutout()` exposes them; the GRIB layer uses them (`docs/spec-io.md` §2.6, §4.3).

Bit-identity rule: for every converted model, each `weights.pt` tensor in `model.safetensors` is
bitwise equal (same dtype, shape and bytes) to the tensor of the same name in the independent
weight export `$REF_SCRATCH/converted/<model>/weights.pt` of the same checkpoint (read-only),
and the name sets differ only by `grid.*` (`python/tests/test_convert.py` checks this when
`weights.pt` is available).

Loading is strict: the engine consumes every tensor in the file, and fails if one is left
over or missing (names listed in config `absent` are allowed to be missing).

### 1.2 Config

The `config` JSON, format v3, has the keys `format_version`, `source`, `architecture`, `dims`,
`variables`, `indices`, `residual`, `bounding`, `mappers`, `blocks`, `absent`, `unused`, with
the values of the `config.json` next to the weight export above except `source.checkpoint`
(the path converted), plus an optional
top-level `forcings` section for the rollout (`docs/forcings.md` §1.1):
`{"dynamic": [names], "dynamic_input_index": [model-input indices], "constant": [names],
"constant_input_index": [...], "timestep_hours": int, "source": str}`: the inputs refreshed
with computed forcings after each step, the inputs constant in time (inherited through the
roll), and the step length. Prognostic inputs are `indices.residual_input`. Every value is read
from the checkpoint by the converter, never hard-coded; the engine reads sizes, epsilons,
activation variants, index lists and bounding from it and asserts nothing model-specific.

The engine rejects unknown `architecture.*.kind`, block `kind`, `residual.type`,
`bounding[].type` or `dims.gelu_approximate` values at load time with a clear error.

## 2. Activation names (golden data ↔ engine debug hook)

Keys and shapes (B = 1; T input steps, A = 4 node attributes, C channels):

| key | shape | meaning |
|---|---|---|
| `in.x` | (1, T, G, V_in) | `predict_step` input, physical units, model-input order |
| `in.x_norm` | (1, T, 1, G, V_in) | normalised input |
| `enc.x_src` | (G, T·V_in + A) | assembled data-node features `[t0 vars \| t1 vars \| node attr]` |
| `enc.x_dst` | (H, A) | hidden-node input features |
| `enc.emb_src`, `enc.emb_dst` | (G, C), (H, C) | node embeddings (`enc.emb_src` on all rows) |
| `enc.out`, `proc.{i}.out` (i = 0..n_proc−1) | (H, C) | encoder and processor block outputs |
| `latent` | (H, C) | decoder source input = processor out + `enc.out` |
| `dec.emb_dst`, `dec.block_out` | (G, C) | decoder embedding and block output |
| `dec.out` | (G, V_out) | after the output LayerNorm + Linear |
| `out.pre_bound`, `out.norm` | (1, 1, 1, G, V_out) | normalised, after the residual; after bounding |
| `out.y` | (1, 1, 1, G, V_out) | `predict_step` output, physical units |

Required in every golden dump: `in.x`, `in.x_norm`, `enc.x_src`, `enc.out`, `proc.{i}.out`, `latent`, `dec.out`,
`out.*`; the others are best effort.

The engine's `ActivationHook` (`src/varda/varda.h`)
receives each key exactly once per `predict_step`, as host fp32, contiguous, with that
shape. On the GPU in fp16 the tensor is upcast to fp32 for the hook.

Metric: per key, in float64,
`max_abs = max|a − r|`, `absmax = max|r|`, `rel_absmax = max_abs / absmax` (0 if both are
0, inf if only `absmax` is 0). NaN anywhere in `a` is a failure. Bitwise means
`max_abs == 0`. Elementwise relative error is never used.

## 3. Golden files (`python/export_golden.py`)

The reference golden directories (anemoi runs, read-only: `$REF_GOLDEN/<tag>/activations.pt` +
`manifest.json`, `$REF_GOLDEN_DEV/<tag>/` for the dev model) are exported once to
`$VARDA_SCRATCH/golden/<model>/<tag>.safetensors`:
every key of `activations.pt` as F32 with its shape, and `__metadata__` =
`{"varda_golden": "1", "tag": <tag>, "manifest": <manifest.json as a string>,
"source": <abs path of activations.pt>, "source_size": <bytes>, "checkpoint_uuid": <uuid>}`.
`checkpoint_uuid` pairs the golden data with a model (`source.uuid` in its config): it is
`manifest.checkpoint_identity.uuid` if present, else the uuid in the embedded
`anemoi-metadata/anemoi.json` of the checkpoint the manifest names (old manifests). Consumers refuse to compare unpaired files.

## 4. Tolerances (`tolerances/*.json`)

Calibrated on the reference runs' own spreads (rule in each file's `method`), keyed by checkpoint
uuid (`checkpoint.uuid`). Tables: `cuda-fp32` (engine fp32 vs `cpu-fp32-pyg`), `cuda-fp16`
(engine fp16 vs `cpu-fp32-pyg`), `cuda-fp16-vs-ref-fp16` (engine fp16 vs
`cuda-fp16-triton`). The C++ CPU fp32 path is held to the `cuda-fp32` table (it is the
reference's own cross-device fp32 spread × 3, floored at 1e-5): a from-scratch fp32
implementation with different GEMM kernels is exactly that kind of neutral perturbation.
Never loosen a tolerance to make a test pass without saying so explicitly.

## 5. Rollout files (`varda_rollout` = "1")

Reference rollouts of the original model (anemoi-inference), exported
(`python/reference/export_rollout.py` for Varda, `python/export_rollout.py` for
the dev model) to `$VARDA_SCRATCH/rollout/ref/<tag>/rollout.safetensors` (Varda) and
`$VARDA_SCRATCH/rollout-engine/ref/dev/<tag>/rollout.safetensors`; read by
`apps/varda-rollout`. Tensors: `x0` [2, G, V_in] physical; `y` [N, G, V_out] physical;
`window` [N, 2, G, V_in] (the window step s sees); `forcings` [N−1, G, F] (forcing s = valid
date of output s); `forcing_index`, `prog_in`, `prog_out`, `kept_in` (I64); `mul_in`,
`add_in`, `mul_out`, `add_out`. Metadata: `varda_rollout`, `tag`, `steps`, `start_date`,
`dates`, `forcing_vars`, `in_vars`, `out_vars`, `kept_vars`, `run` (JSON), `checkpoint_uuid`
(when known). Consumers refuse a file whose variables, grid size or uuid differ from the
model's. Per-step tolerances: `tolerances/rollout/<model>.json` (keyed by uuid; method in
`docs/rollout.md`, judging in `docs/rollout-engine.md` §4).
