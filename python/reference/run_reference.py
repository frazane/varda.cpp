"""Run the ORIGINAL anemoi model (the Varda forecaster) once on a deterministic synthetic input and dump golden tensors.

Contract activations (docs/format.md §2) are in the reference golden directories ($REF_GOLDEN, read-only;
python/export_golden.py converts them). This script writes the per-file dumps (input.pt, output.pt,
activations/*.pt) and is the module run_rollout.py imports (model loading, attention backend, variable names).

Runs in $VARDA_SCRATCH/venv-ref-varda only (it unpickles the real anemoi classes). Never import this from src/ or tests/.

Environment: $VARDA_SCRATCH/venv-ref-varda (uv, Python 3.12), lock requirements-ref-varda.lock.txt next to this file.
Recipe (the Varda reference lock $REF_LOCK, plus anemoi-inference at the commit the Hugging Face release pins):

    export UV_CACHE_DIR=<a uv cache on the scratch file system>; V=$VARDA_SCRATCH/venv-ref-varda
    uv venv --python ~/.local/bin/python3.12 $V
    # torch/torchvision from the cu130 index ALONE (never combined with --extra-index-url pypi)
    uv pip install --python $V/bin/python "torch==2.10.0+cu130" "torchvision==0.25.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130
    printf 'torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\n' > constraints.txt
    # the reference lock minus the two torch lines (PyPI + the git pins inside the lock):
    grep -vE '^(torch|torchvision)==' $REF_LOCK > lock-rest.txt
    uv pip install --python $V/bin/python -c constraints.txt -r lock-rest.txt
    uv pip freeze > freeze-before.txt          # identical to $REF_LOCK
    # constraints.txt = every installed package pinned (name==version) EXCEPT anemoi-utils, anemoi-transform, pandas
    # (anemoi-inference 9505dd09 needs anemoi-utils>=0.5.13 and, to import, an anemoi-transform whose Variable has
    # `.units` (0.3.0+; the HF release asks >=0.4.2), which needs pandas<3). Nothing else has to move.
    uv pip install --python $V/bin/python -c constraints-final.txt "anemoi-transform==0.4.2" \
        "anemoi-inference @ git+https://github.com/ecmwf/anemoi-inference@9505dd0926ec3c5dea9308813cc7859bfdad755a"
    # moved: anemoi-utils 0.5.1 -> 0.5.14, anemoi-transform 0.1.26 -> 0.4.2, pandas 3.0.6 -> 2.3.3;
    # added: anemoi-inference, anytree, astropy, astropy-iers-data, healpy, peekle, pydantic-settings, pyerfa,
    # termcolor, tzdata, wcwidth. torch unchanged (docs/rollout.md).

The pickle is the Hugging Face checkpoint ($VARDA_CKPT = varda-forecaster-det-sgm/model.ckpt).

Usage (compute node; see CLAUDE.md):

    srun $VARDA_SLURM_CPU --mem=300G \
        $VARDA_SCRATCH/venv-ref-varda/bin/python python/reference/run_reference.py --device cpu --backend pyg \
        --input $REF_GOLDEN/cpu_float32_pyg/input.pt --out $VARDA_SCRATCH/golden-ref/cpu_float32_pyg

Attention backend override: the checkpoint was pickled with graph_attention_backend="triton", i.e. every
GraphTransformer{Mapper,Processor}Block has `.graph_attention_backend == "triton"` and
`.conv == GraphTransformerFunction.apply` (a Triton kernel, CUDA only). The constructor's fallback to "pyg"
only runs at construction time, not on unpickling, so for `--backend pyg` we set, on every such block:
    block.graph_attention_backend = "pyg"
    block.conv = anemoi.models.layers.conv.GraphTransformerConv(out_channels=block.out_channels_conv)
which is exactly what the constructor would have built for "pyg" (GraphTransformerConv has no parameters).
Unpickling itself still needs `triton` importable (the pickle references anemoi.models.triton.gt).
Other settings: map_location="cpu" (tensors were saved on cuda:0), .eval() (pickled with training=True),
ANEMOI_INFERENCE_NUM_CHUNKS{,_MAPPER,_PROCESSOR} must be unset (asserted), TF32 disabled.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import torch

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH") or sys.exit("VARDA_SCRATCH is not set: source tools/env.sh")
CKPT = os.environ.get("VARDA_CKPT") or sys.exit("VARDA_CKPT is not set: source tools/env.sh")  # the HF checkpoint
OUT_ROOT = f"{VARDA_SCRATCH}/golden-ref"  # one-step dumps; the reference golden data is in $REF_GOLDEN
DATASET = "data"
CHUNK_ENV = ("ANEMOI_INFERENCE_NUM_CHUNKS", "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", "ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR")


def set_attention_backend(model: torch.nn.Module, backend: str) -> list[str]:
    from anemoi.models.layers.block import GraphTransformerBaseBlock
    from anemoi.models.layers.conv import GraphTransformerConv

    changed = []
    for name, mod in model.named_modules():
        if not isinstance(mod, GraphTransformerBaseBlock):
            continue
        if backend == "pyg":
            mod.graph_attention_backend = "pyg"
            mod.conv = GraphTransformerConv(out_channels=mod.out_channels_conv)
        elif backend == "triton":
            from anemoi.models.triton.gt import GraphTransformerFunction

            mod.graph_attention_backend = "triton"
            if isinstance(getattr(mod, "conv", None), torch.nn.Module):
                del mod.conv  # drop a registered GraphTransformerConv submodule
            mod.conv = GraphTransformerFunction.apply
        else:
            raise ValueError(backend)
        changed.append(name)
    return changed


def variable_names(model) -> tuple[list[str], list[str]]:
    idx = model.data_indices[DATASET]
    index_to_name = {i: n for n, i in idx.name_to_index.items()}
    norm = model.pre_processors[DATASET].processors["normalizer"]
    inputs = [index_to_name[int(i)] for i in norm._input_idx]
    outputs = [index_to_name[int(i)] for i in idx.data.output.full]
    return inputs, outputs


def make_input(model, seed: int, ckpt: str | None = None) -> tuple[torch.Tensor, dict]:
    """Deterministic synthetic input, shape (batch=1, time=2, grid=G, vars=V_in), physical units.

    G is the model's data-node count (dev model 40320, Varda 1688650), V_in its input variables.
    Variables with mean-std normalisation: z ~ N(0,1) in normalised space at t=0, t=1 = t=0 + 0.1 N(0,1),
    de-normalised with the model's own pre-processor statistics, then clipped to the dataset [min, max]
    from `model.statistics` (when available) so e.g. q_* stays >= 0. Max-normalised (z, sdor, slor):
    U(0,1) * max. Unnormalised: sin/cos of lat/lon from the real grid; other sin/cos features U(-1,1);
    cloud covers, insolation, lsm U(0,1). Everything except the mean-std variables is identical at t=0, t=1.
    Caveat: `model.statistics` need not be the statistics folded into the normaliser (Varda: global
    normaliser kept from the forked run, `statistics` of the regional dataset); the clipping then uses the
    regional ranges. Harmless for parity: the input only has to be finite and plausible.
    """
    names, _ = variable_names(model)
    norm = model.pre_processors[DATASET].processors["normalizer"]
    mul = norm._norm_mul[norm._input_idx.long()].double()
    add = norm._norm_add[norm._input_idx.long()].double()
    g = torch.Generator().manual_seed(seed)
    nv = len(names)
    n_grid = int(model.model.node_attributes.num_nodes[DATASET])

    z0 = torch.randn(n_grid, nv, generator=g, dtype=torch.float64)
    dz = 0.1 * torch.randn(n_grid, nv, generator=g, dtype=torch.float64)
    u01 = torch.rand(n_grid, nv, generator=g, dtype=torch.float64)

    lat = lon = None
    if ckpt is not None:  # the unpickled model.supporting_arrays is None; the arrays live in the zip
        from anemoi.utils.checkpoints import load_metadata

        arrays = load_metadata(ckpt, supporting_arrays=True)[1].get(DATASET, {})
        if "latitudes" in arrays and "longitudes" in arrays:
            lat = torch.as_tensor(arrays["latitudes"], dtype=torch.float64).deg2rad()
            lon = torch.as_tensor(arrays["longitudes"], dtype=torch.float64).deg2rad()

    lo = hi = None
    stats = getattr(model, "statistics", None) or {}
    stats = stats.get(DATASET, stats)
    if "minimum" in stats and "maximum" in stats and len(stats["minimum"]) == len(model.data_indices[DATASET].name_to_index):
        sel = norm._input_idx.long()
        lo = torch.as_tensor(stats["minimum"], dtype=torch.float64)[sel]
        hi = torch.as_tensor(stats["maximum"], dtype=torch.float64)[sel]

    x = torch.empty(2, n_grid, nv, dtype=torch.float64)
    groups = {"mean-std": [], "max": [], "none": []}
    for j, name in enumerate(names):
        if mul[j] == 1.0 and add[j] == 0.0:
            groups["none"].append(name)
            geo = {"sin_latitude": (lat, torch.sin), "cos_latitude": (lat, torch.cos),
                   "sin_longitude": (lon, torch.sin), "cos_longitude": (lon, torch.cos)}
            if name in geo and geo[name][0] is not None:
                col = geo[name][1](geo[name][0])
            elif name.startswith(("sin_", "cos_")):
                col = 2.0 * u01[:, j] - 1.0
            else:
                col = u01[:, j]
            x[0, :, j] = x[1, :, j] = col
        elif add[j] == 0.0:
            groups["max"].append(name)
            x[0, :, j] = x[1, :, j] = u01[:, j] / mul[j]
        else:
            groups["mean-std"].append(name)
            x[0, :, j] = (z0[:, j] - add[j]) / mul[j]
            x[1, :, j] = (z0[:, j] + dz[:, j] - add[j]) / mul[j]
            if lo is not None:
                x[:, :, j] = x[:, :, j].clamp(lo[j], hi[j])
    recipe = {
        "seed": seed,
        "generator": "torch.Generator().manual_seed(seed); draws in float64: randn(z0), randn(dz), rand(u01), "
        f"each ({n_grid}, {nv}); cast to the run dtype at the end",
        "groups": groups,
        "latlon_from_supporting_arrays": lat is not None and lon is not None,
        "mean_std_clipped_to_dataset_min_max": lo is not None,
    }
    return x.unsqueeze(0).float(), recipe


class Recorder:
    """Forward hooks that keep selected activations (detached, moved to CPU).

    The mappers run in `num_chunks` (=4) chunks of destination nodes, so the embedding Linears are called once
    per chunk. Destination chunks are disjoint and in order, so their outputs are concatenated. The encoder's
    source embedding only sees the source nodes connected to each chunk (overlapping subsets), so instead it is
    recomputed once on all data nodes after the forward pass (`encoder.emb_nodes_src`, row-wise identical math).
    """

    def __init__(self, model):
        self.calls: dict[str, list[torch.Tensor]] = {}
        self.handles = []
        inner = model.model
        self.encoder = inner.encoder[DATASET]
        spec = {
            "pre_processed_input": (model.pre_processors[DATASET], lambda o: {"": o}),
            "encoder.emb_nodes_dst": (self.encoder.emb_nodes_dst, lambda o: {"": o}),
            "encoder": (self.encoder, lambda o: {".x_data_latent": o[0], ".x_latent": o[1]}),
            "processor": (inner.processor, lambda o: {"": o}),
            "decoder.emb_nodes_dst": (inner.decoder[DATASET].emb_nodes_dst, lambda o: {"": o}),
            "decoder": (inner.decoder[DATASET], lambda o: {"": o}),
            # interface.forward is the inner model's *bound method*, so hooks on `inner` never fire;
            # the last bounding module's output is the normalised model output (after residual + bounding).
            "model_output_normalised": (inner.boundings[DATASET][-1], lambda o: {"": o}),
        }
        for i, blk in enumerate(inner.processor.proc):
            spec[f"processor.proc.{i}"] = (blk, lambda o: {"": o[0]})
        for name, (mod, sel) in spec.items():
            self.handles.append(mod.register_forward_hook(self._hook(name, sel)))

    def _hook(self, name, sel):
        def fn(_mod, _inp, out):
            for suffix, t in sel(out).items():
                self.calls.setdefault(name + suffix, []).append(t.detach().to("cpu", copy=True))

        return fn

    def reset(self):
        self.calls = {}

    def collect(self, device, dtype) -> dict[str, torch.Tensor]:
        acts = {}
        for k, ts in self.calls.items():
            assert len(ts) == 1 or k.endswith("emb_nodes_dst"), (k, len(ts))
            acts[k] = torch.cat(ts, dim=0) if len(ts) > 1 else ts[0]
        with torch.inference_mode():
            x_src = acts["encoder.x_data_latent"].to(device=device, dtype=dtype)
            acts["encoder.emb_nodes_src"] = self.encoder.emb_nodes_src(x_src).to("cpu")
        return acts

def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def versions() -> dict:
    pkgs = ["torch", "torch_geometric", "anemoi-models", "anemoi-graphs", "anemoi-training", "anemoi-datasets",
            "anemoi-utils", "anemoi-transform", "anemoi-inference", "pytorch-lightning", "numpy", "triton"]
    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--backend", default="pyg", choices=["pyg", "triton"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2, help="forward passes; >1 also measures determinism")
    ap.add_argument("--input", default=None, help="reuse an existing input.pt instead of generating one")
    ap.add_argument("--out", default=None, help=f"default: {OUT_ROOT}/<device>_<dtype>_<backend>")
    ap.add_argument("--no-intermediates", action="store_true")
    ap.add_argument("--print-model", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    out = Path(args.out or f"{OUT_ROOT}/{device.type}_{args.dtype}_{args.backend}")
    out.mkdir(parents=True, exist_ok=True)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    # Chunked attention changes the summation order; keep the default (1 chunk) and record it.
    chunk_env = {k: os.environ.get(k) for k in CHUNK_ENV}
    assert all(v is None for v in chunk_env.values()), f"unset these for golden runs: {chunk_env}"
    # Training used fp16 autocast with matmul precision "high"; golden fp32 runs must not use TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    t0 = time.perf_counter()
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t_load = time.perf_counter() - t0
    (out / "model.txt").write_text(str(model) + "\n")
    if args.print_model:
        print(model)
    changed = set_attention_backend(model, args.backend)
    model = model.to(device=device, dtype=dtype).eval()

    if args.input:
        x = torch.load(args.input, weights_only=True)
        recipe = json.loads((Path(args.input).parent / "manifest.json").read_text())["input_recipe"]
    else:
        x, recipe = make_input(model, args.seed, args.ckpt)
    in_names, out_names = variable_names(model)
    torch.save(x, out / "input.pt")
    x_dev = x.to(device=device, dtype=dtype)

    rec = None if args.no_intermediates else Recorder(model)
    outputs, times = [], []
    rss_before = peak_rss_gb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for r in range(args.runs):
        if rec:
            rec.reset()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            y = model.predict_step({DATASET: x_dev})[DATASET]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        outputs.append(y.detach().to("cpu", copy=True))
        if rec:
            acts = rec.collect(device, dtype)
            if r == 0:
                acts0 = acts
        print(f"run {r}: {times[-1]:.2f} s, output {tuple(y.shape)} {y.dtype}", flush=True)

    determinism = {}
    for r in range(1, args.runs):
        d = (outputs[r].float() - outputs[0].float()).abs()
        determinism[f"run{r}_vs_run0_output_max_abs"] = d.max().item()
    if rec and args.runs > 1:
        determinism[f"run{args.runs - 1}_vs_run0_max_abs_per_activation"] = {
            k: (acts[k].float() - acts0[k].float()).abs().max().item() for k in acts0
        }
    print("determinism:", json.dumps(determinism, indent=1))

    torch.save(outputs[0], out / "output.pt")
    tensors = {"input.pt": x, "output.pt": outputs[0]}
    if rec:
        (out / "activations").mkdir(exist_ok=True)
        for k, t in acts0.items():
            torch.save(t, out / "activations" / f"{k}.pt")
            tensors[f"activations/{k}.pt"] = t

    manifest = {
        "checkpoint": args.ckpt,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "host": socket.gethostname(),
        "dtype": args.dtype,
        "backend": args.backend,
        "backend_override": {"blocks": changed, "set": f"graph_attention_backend={args.backend!r}, conv=" + (
            "GraphTransformerConv(out_channels=block.out_channels_conv)" if args.backend == "pyg"
            else "GraphTransformerFunction.apply")},
        "torch_num_threads": torch.get_num_threads(),
        "chunk_env": chunk_env,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "model_training_flag_after_eval": model.training,
        "seed": args.seed,
        "input_recipe": recipe,
        "input_layout": "batch, time, grid, variable (data.input.full order)",
        "output_layout": "batch, time, ensemble, grid, variable (data.output.full order)",
        "input_variables": in_names,
        "output_variables": out_names,
        "call": "model.predict_step({'data': input})['data'] under torch.inference_mode()",
        "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype)} for k, t in tensors.items()},
        "versions": versions(),
        "timing_s": {"load": t_load, "forward": times},
        "memory": {
            "peak_rss_gb_after_load": rss_before,
            "peak_rss_gb_after_forward": peak_rss_gb(),
            **({"cuda_max_allocated_gb_forward": torch.cuda.max_memory_allocated(device) / 1024**3,
                "cuda_max_reserved_gb_forward": torch.cuda.max_memory_reserved(device) / 1024**3}
               if device.type == "cuda" else {}),
        },
        "determinism": determinism,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("timing_s", "memory")}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
