"""Run the ORIGINAL anemoi model once on a deterministic synthetic input and dump golden tensors.

Two modes:
  * legacy (default): per-file dumps (input.pt, output.pt, activations/*.pt) of the Varda forecaster; this is also the
    module run_rollout.py imports (model loading, attention backend, variable names).
  * contract (--golden TAG): the docs/format.md §2 activations of any supported checkpoint (forecaster or temporal
    downscaler) as ONE dict in <out>/activations.pt plus manifest.json, which python/export_golden.py converts. The
    forecaster's reference golden directories ($REF_GOLDEN, read-only) were made by the same capture in the reference
    repository; the Varda temporal downscaler's ($VARDA_SCRATCH/golden-ref/varda-td/<tag>) by this script:

    R=$VARDA_SCRATCH/golden-ref/varda-td; PY=$VARDA_SCRATCH/venv-ref-varda-td/bin/python
    CK=$VARDA_SCRATCH/hf/varda-temporal-downscaler-det-sgm/model.ckpt
    srun $VARDA_SLURM_CPU $PY python/reference/run_reference.py --ckpt $CK --golden varda-td/input --input-only \
        --out $R/input                                                           # once: input.pt, input.json
    sbatch $VARDA_SLURM_CPU --mem=400G ... $PY python/reference/run_reference.py --ckpt $CK --golden varda-td/cpu-fp32-pyg \
        --device cpu --backend pyg --input $R/input/input.pt --out $R/cpu-fp32-pyg --runs 2
    sbatch $VARDA_SLURM_GPU -c 16 --mem=400G ... (same) --golden varda-td/cuda-fp32-triton --device cuda --backend triton
    sbatch $VARDA_SLURM_GPU -c 16 --mem=400G ... (same) --golden varda-td/cuda-fp16-triton --device cuda --backend triton \
        --autocast fp16
  Each run: regenerate the input and require it bitwise equal to --input; a hook-free predict_step; --runs capture runs
  (all activations bitwise equal across them); exact contract keys and shapes (T_out = n_output_steps), all finite;
  out.y bitwise equal to the hook-free run (CPU fp32); internal consistency checks (normalisation, input assembly,
  latent, residual on every output step, bounding, de-normalisation), all recorded in manifest.json; the script fails if
  a bitwise one fails.

Reference venvs: the anemoi-core commit each checkpoint was trained with (anemoi.json provenance_training), which
for the downscaler is also the one its Hugging Face pyproject.toml pins:
  * forecaster: $VARDA_SCRATCH/venv-ref-varda (anemoi-core 3c5fc21), recipe below;
  * temporal downscaler: $VARDA_SCRATCH/venv-ref-varda-td = the same recipe with the three anemoi-core lines of
    requirements-ref-varda.lock.txt (anemoi-graphs, -models, -training) at d9cf401c85506f9cb850e36f3b96ea2432ea9b1c
    instead of 3c5fc21 (`sed s/3c5fc2117797e9aaa1c4c0ee5efa63e17af9b985/d9cf401c85506f9cb850e36f3b96ea2432ea9b1c/`);
    `uv pip freeze` equals that edited lock. The downscaler's pickle needs it: in d9cf401 `model.node_attributes`
    is a ModuleDict per dataset (3c5fc21 calls it directly and fails), and the decoder's source is the processor
    output (`x_latent_proc (+ x_latent if latent_skip)`, interpolator.py), whereas the later interpolator.py passes
    the encoder output when latent_skip is False.

Runs in a reference venv only (it unpickles the real anemoi classes). Never import this from src/ or tests/.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # compare_golden (also under python -I)

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


def node_attributes(inner):
    """The NamedNodesAttributes module: `inner.node_attributes` (forecaster, anemoi-core 3c5fc21) or
    `inner.node_attributes[DATASET]` (temporal downscaler, anemoi-core d9cf401: a ModuleDict per dataset)."""
    na = inner.node_attributes
    return na[DATASET] if isinstance(na, torch.nn.ModuleDict) else na


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
    n_grid = int(node_attributes(model.model).num_nodes[DATASET])
    assert int(model.n_step_input) == 2, "the recipe draws exactly two input states"

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

# ------------------------------------------------------------------------------------------------------------------
# Contract golden dump (--golden TAG): docs/format.md §2 keys -> <out>/{activations.pt, manifest.json}
# ------------------------------------------------------------------------------------------------------------------
#
# Ported from the reference repository's golden dumper (the one that wrote $REF_GOLDEN; same capture points, checks
# and manifest fields), generalised to n_output_steps > 1 (the temporal downscaler) and to both pickle layouts.
# Every capture point is a forward hook / pre-hook on a submodule called through nn.Module.__call__, or a wrapper
# that only records a return value; out.y is compared bitwise with a hook-free run, which proves nothing changed.

ENV_RECORD = CHUNK_ENV + ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_ID",
                          "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                          "PYTHONHASHSEED", "TORCH_DETERMINISTIC")
_SEEN_DTYPES: list = []  # original dtype of every tensor passed through cpu_copy (drained by Capture.rec)


def cpu_copy(t: torch.Tensor) -> torch.Tensor:
    """Detached contiguous fp32 CPU copy (the cast happens after capture; the original dtype is recorded)."""
    _SEEN_DTYPES.append(str(t.dtype))
    return t.detach().to("cpu", copy=True).to(torch.float32).contiguous()


def sha256_tensor(t: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


class Capture:
    """Records the contract activations during one predict_step.

    Mapper chunking: each mapper splits its destination nodes into num_chunks contiguous, ordered chunks
    (mapper.py, `max(num_chunks, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER)`) and calls emb_nodes_dst and the mapper block
    once per chunk; those hooks append per-chunk tensors, concatenated in collect() after checking that the
    concatenated *inputs* equal the full tensor seen at the mapper boundary. The encoder's emb_nodes_src sees
    overlapping per-chunk subsets of source nodes, so `enc.emb_src` is recomputed on the full enc.x_src and the
    per-chunk in-model values are compared with it (diagnostic). Boundings work in place: out.pre_bound is a pre-hook
    on the first one, out.norm a hook on the last one (without boundings: pre-hook on the post-processor).
    """

    def __init__(self, model, amp=None):
        import contextlib

        import anemoi.models.layers.mapper as mapper_mod

        self.amp = amp or contextlib.nullcontext
        self.model = model
        inner = model.model
        self.inner = inner
        self.enc = inner.encoder[DATASET]
        self.dec = inner.decoder[DATASET]
        self.calls: dict[str, list] = {}
        self.how: dict[str, str] = {}
        self.handles = []
        self.mapper_mod = mapper_mod
        self.orig_drop = mapper_mod.drop_unconnected_src_nodes
        self.dtypes: dict[str, list] = {}

        def rec(name, value):
            self.calls.setdefault(name, []).append(value)
            self.dtypes.setdefault(name, []).append(list(_SEEN_DTYPES))
            _SEEN_DTYPES.clear()

        def fwd(mod, name, sel):
            self.handles.append(mod.register_forward_hook(lambda _m, a, o: rec(name, sel(a, o))))

        def pre(mod, name, sel):
            self.handles.append(mod.register_forward_pre_hook(lambda _m, a: rec(name, sel(a))))

        pp = model.pre_processors[DATASET]
        fwd(pp, "in.x_norm", lambda a, o: cpu_copy(o))
        pre(self.enc, "enc.in", lambda a: (cpu_copy(a[0][0]), cpu_copy(a[0][1])))
        fwd(self.enc, "enc.fwd", lambda a, o: (cpu_copy(o[0]), cpu_copy(o[1])))
        fwd(self.enc.emb_nodes_src, "enc.emb_src.chunk", lambda a, o: cpu_copy(o))
        fwd(self.enc.emb_nodes_dst, "enc.emb_dst.chunk", lambda a, o: (cpu_copy(a[0]), cpu_copy(o)))
        for i, blk in enumerate(inner.processor.proc):
            fwd(blk, f"proc.{i}.out", lambda a, o: cpu_copy(o[0]))
        fwd(inner.processor, "processor.out", lambda a, o: cpu_copy(o))
        pre(self.dec, "dec.in", lambda a: (cpu_copy(a[0][0]), cpu_copy(a[0][1])))
        fwd(self.dec.emb_nodes_dst, "dec.emb_dst.chunk", lambda a, o: (cpu_copy(a[0]), cpu_copy(o)))
        fwd(self.dec.proc, "dec.block.chunk", lambda a, o: (cpu_copy(a[0][1]), cpu_copy(o[0][1])))
        fwd(self.dec.node_data_extractor, "dec.extractor.chunk", lambda a, o: cpu_copy(o))
        fwd(self.dec, "dec.out", lambda a, o: cpu_copy(o))
        bnd = inner.boundings[DATASET]
        self.n_bound = len(bnd)
        if self.n_bound:
            pre(bnd[0], "out.pre_bound", lambda a: cpu_copy(a[0]))  # cloned before the in-place boundings run
            fwd(bnd[-1], "out.norm", lambda a, o: cpu_copy(o))
        else:
            post = model.post_processors[DATASET]
            assert post is not pp, "post-processor is the pre-processor object: its hooks would fire twice"
            pre(post, "out.pre_bound", lambda a: cpu_copy(a[0]))

        def drop_wrapper(x_src, edge_index, num_nodes):  # records (not changes) the src subset each chunk embeds
            out = self.orig_drop(x_src, edge_index, num_nodes)
            rec("drop_unconnected", (x_src.shape[0], out[2].detach().clone().cpu()))
            return out

        mapper_mod.drop_unconnected_src_nodes = drop_wrapper

    def close(self):
        for h in self.handles:
            h.remove()
        self.mapper_mod.drop_unconnected_src_nodes = self.orig_drop

    def one(self, name):
        v = self.calls.get(name, [])
        assert len(v) == 1, f"hook {name} fired {len(v)} times, expected 1"
        return v[0]

    def chunks(self, name, n):
        v = self.calls.get(name, [])
        assert len(v) == n, f"hook {name} fired {len(v)} times, expected {n}"
        return v

    def collect(self, x_in, y, n_enc: int, n_dec: int) -> tuple[dict, dict]:
        acts, diag, how = {}, {}, self.how
        acts["in.x"] = cpu_copy(x_in)
        _SEEN_DTYPES.clear()
        how["in.x"] = "the predict_step input tensor itself (loaded from --input)"
        acts["in.x_norm"] = self.one("in.x_norm")
        how["in.x_norm"] = "forward hook on model.pre_processors['data'] (output, in_place=False clone)"

        x_src, x_dst = self.one("enc.in")
        acts["enc.x_src"], acts["enc.x_dst"] = x_src, x_dst
        how["enc.x_src"] = f"forward pre-hook on model.model.encoder['data']: args[0][0] (x_data_latent, {x_src.shape[1]} wide)"
        how["enc.x_dst"] = "forward pre-hook on model.model.encoder['data']: args[0][1] (x_hidden_latent)"
        enc_ret_src, enc_out = self.one("enc.fwd")
        acts["enc.out"] = enc_out
        how["enc.out"] = "forward hook on model.model.encoder['data']: output[1]"
        assert torch.equal(enc_ret_src, x_src), "encoder must return its raw x_src unchanged"

        ch = self.chunks("enc.emb_dst.chunk", n_enc)
        assert torch.equal(torch.cat([c[0] for c in ch]), x_dst), "enc.emb_nodes_dst chunk inputs != x_dst in order"
        acts["enc.emb_dst"] = torch.cat([c[1] for c in ch])
        how["enc.emb_dst"] = (f"forward hook on encoder.emb_nodes_dst, fired {n_enc}x (one per contiguous "
                              "dst chunk); outputs concatenated; order verified by concatenated inputs == enc.x_dst")

        with torch.inference_mode(), self.amp():
            w = self.enc.emb_nodes_src  # nn.Linear; F.linear is its forward and does not fire our hook
            acts["enc.emb_src"] = cpu_copy(torch.nn.functional.linear(x_src.to(w.weight.device, w.weight.dtype),
                                                                      w.weight, w.bias))
        self.emb_src_dtype = _SEEN_DTYPES[-1]
        how["enc.emb_src"] = ("RECOMPUTED after the forward: encoder.emb_nodes_src(enc.x_src) on all rows at once "
                              "(the model embeds overlapping per-chunk src subsets, so no full tensor exists); "
                              "see diagnostics.enc_emb_src_in_model_vs_recomputed")
        drops = self.chunks("drop_unconnected", (1 + n_enc) + (1 + n_dec))
        g_n, g_idx = drops[0]
        assert g_n == x_src.shape[0]
        src_chunks = self.chunks("enc.emb_src.chunk", n_enc)
        worst, covered, bitwise = 0.0, torch.zeros(x_src.shape[0], dtype=torch.bool), True
        for (n_sub, idx_c), emb_c in zip(drops[1:1 + n_enc], src_chunks):
            assert n_sub == g_idx.numel()
            rows = g_idx[idx_c]
            covered[rows] = True
            ref = acts["enc.emb_src"][rows]
            worst = max(worst, (emb_c - ref).abs().max().item())
            bitwise &= torch.equal(emb_c, ref)
        how["enc.emb_src"] += f"; in-model per-chunk rows bitwise equal to it: {bitwise} (max_abs {worst:.3e})"
        diag["enc_emb_src_in_model_vs_recomputed"] = {
            "max_abs": worst, "bitwise_equal": bitwise, "src_rows_connected": int(covered.sum()),
            "src_rows_total": int(x_src.shape[0]), "rows_embedded_per_chunk": [int(c.shape[0]) for c in src_chunks]}

        for i in range(len(self.inner.processor.proc)):
            acts[f"proc.{i}.out"] = self.one(f"proc.{i}.out")
            how[f"proc.{i}.out"] = f"forward hook on model.model.processor.proc[{i}]: output[0]"
        last = f"proc.{len(self.inner.processor.proc) - 1}.out"
        assert torch.equal(self.one("processor.out"), acts[last]), "processor output != last block output"

        latent, dec_x_dst = self.one("dec.in")
        acts["latent"] = latent
        how["latent"] = "forward pre-hook on model.model.decoder['data']: args[0][0] (the decoder's source input)"
        assert torch.equal(dec_x_dst, x_src), "decoder dst input must be the raw enc.x_src"

        ch = self.chunks("dec.emb_dst.chunk", n_dec)
        assert torch.equal(torch.cat([c[0] for c in ch]), x_src), "dec.emb_nodes_dst chunk inputs != x_src in order"
        acts["dec.emb_dst"] = torch.cat([c[1] for c in ch])
        how["dec.emb_dst"] = (f"forward hook on decoder.emb_nodes_dst, fired {n_dec}x (contiguous dst "
                              "chunks); outputs concatenated; order verified by concatenated inputs == enc.x_src")
        ch = self.chunks("dec.block.chunk", n_dec)
        assert torch.equal(torch.cat([c[0] for c in ch]), acts["dec.emb_dst"]), "decoder block chunk order"
        acts["dec.block_out"] = torch.cat([c[1] for c in ch])
        how["dec.block_out"] = (f"forward hook on decoder.proc (GraphTransformerMapperBlock), fired {n_dec}x; "
                                "output[0][1] concatenated; order verified by concatenated dst inputs == dec.emb_dst")
        ext = torch.cat(self.chunks("dec.extractor.chunk", n_dec))
        acts["dec.out"] = self.one("dec.out")
        how["dec.out"] = f"forward hook on model.model.decoder['data']: output {tuple(acts['dec.out'].shape)}"
        assert torch.equal(ext, acts["dec.out"]), "node_data_extractor chunk outputs != decoder output"

        acts["out.pre_bound"] = self.one("out.pre_bound")
        if self.n_bound:
            how["out.pre_bound"] = ("forward pre-hook on model.model.boundings['data'][0], args[0] cloned "
                                    "(the boundings work in place)")
            acts["out.norm"] = self.one("out.norm")
            how["out.norm"] = "forward hook on model.model.boundings['data'][-1]: output"
        else:
            how["out.pre_bound"] = "forward pre-hook on model.post_processors['data']: args[0] (no boundings)"
            acts["out.norm"] = acts["out.pre_bound"].clone()
            how["out.norm"] = "= out.pre_bound (the model has no boundings)"
        acts["out.y"] = cpu_copy(y)
        _SEEN_DTYPES.clear()
        how["out.y"] = "return value of model.predict_step({'data': in.x})['data']"
        return acts, diag

    def captured_dtypes(self, y, x_in) -> dict:
        """Original (pre-cast) dtype of each contract key as seen by its hook (list when it was concatenated)."""
        d = self.dtypes
        flat = lambda name, j: sorted({c[j] for c in d[name]})  # noqa: E731
        out = {"in.x": [str(x_in.dtype)], "in.x_norm": flat("in.x_norm", 0), "enc.x_src": flat("enc.in", 0),
               "enc.x_dst": flat("enc.in", 1), "enc.emb_src": [self.emb_src_dtype],
               "enc.emb_dst": flat("enc.emb_dst.chunk", 1), "enc.out": flat("enc.fwd", 1), "latent": flat("dec.in", 0),
               "dec.emb_dst": flat("dec.emb_dst.chunk", 1), "dec.block_out": flat("dec.block.chunk", 1),
               "dec.out": flat("dec.out", 0), "out.pre_bound": flat("out.pre_bound", 0),
               "out.norm": flat("out.norm" if self.n_bound else "out.pre_bound", 0), "out.y": [str(y.dtype)]}
        for i in range(len(self.inner.processor.proc)):
            out[f"proc.{i}.out"] = flat(f"proc.{i}.out", 0)
        return out


def consistency_checks(model, acts: dict, strict: bool = True, amp=None) -> dict:
    """Internal consistency, mirroring the original arithmetic so that 'bitwise' is the expected outcome."""
    import contextlib

    from compare_golden import diff_metric

    inner = model.model
    norm = model.pre_processors[DATASET].processors["normalizer"]
    mul, add = norm._norm_mul.cpu(), norm._norm_add.cpu()
    in_idx, out_idx = norm._input_idx.long().cpu(), norm._output_idx.long().cpu()
    res = {}

    def chk(name, ref, other, expect_bitwise=True):
        m = diff_metric(ref, other)
        m["expect_bitwise"] = expect_bitwise and strict  # under autocast: reported numbers only
        res[name] = m
        return m

    # in.x_norm = (in.x[:, 0:T, None] * mul) + add   (normalizer.py: x.mul_(...).add_(...))
    x = acts["in.x"][:, 0:acts["in.x_norm"].shape[1], None].clone()
    chk("in.x_norm == in.x*mul+add", acts["in.x_norm"], x.mul_(mul[in_idx]).add_(add[in_idx]))
    # enc.x_src = cat(rearrange(x_norm, 'b t e g v -> (b e g) (t v)'), latlons_data)
    xn = acts["in.x_norm"]
    b, t, e, g, v = xn.shape
    flat = xn.permute(0, 2, 3, 1, 4).reshape(b * e * g, t * v)
    na = node_attributes(inner)
    chk("enc.x_src == cat(rearrange(in.x_norm), latlons_data)", acts["enc.x_src"],
        torch.cat([flat, na.latlons_data.cpu()], -1))
    chk("enc.x_dst == latlons_hidden", acts["enc.x_dst"], na.latlons_hidden.cpu())
    # latent (the decoder's source) = last proc out (+ enc.out if latent_skip; fp32 add)
    last = f"proc.{len(inner.processor.proc) - 1}.out"
    if inner.latent_skip:
        chk(f"latent == {last} + enc.out", acts["latent"], acts[last] + acts["enc.out"])
    else:
        chk(f"latent == {last} (latent_skip False)", acts["latent"], acts[last])
    # dec.out = node_data_extractor(dec.block_out), recomputed on all rows at once (the model: per dst chunk)
    with torch.inference_mode(), (amp or contextlib.nullcontext)():
        ext = inner.decoder[DATASET].node_data_extractor
        p = next(ext.parameters())
        rec = ext(acts["dec.block_out"].to(p.device, p.dtype)).cpu()
    chk("dec.out vs node_data_extractor(dec.block_out) [full-row GEMM]", acts["dec.out"], rec, expect_bitwise=False)
    # out.pre_bound = rearrange('(b e g) (t v) -> b t e g v', dec.out) + residual on the prognostic channels,
    # x_skip = x_norm[:, step] expanded over the T_out output steps (residual.py SkipConnection._expand_time)
    t_out = acts["out.pre_bound"].shape[1]
    y = acts["dec.out"].reshape(b, e, g, t_out, -1).permute(0, 3, 1, 2, 4).clone()
    ii = torch.as_tensor(inner._internal_input_idx[DATASET]).long().cpu()
    oi = torch.as_tensor(inner._internal_output_idx[DATASET]).long().cpu()
    step = int(inner.residual[DATASET].step)  # SkipConnection (the converter rejects other residual classes)
    y[..., oi] += xn[:, step].unsqueeze(1)[..., ii]
    chk(f"out.pre_bound == dec.out + x_norm[:, {step}] (prognostic, every output step)", acts["out.pre_bound"], y)
    # bounding: only the bounded channels change; recompute with plain torch ops in config order
    bounded = sorted({int(i) for bd in inner.boundings[DATASET] for i in bd.data_index})
    others = [j for j in range(acts["out.norm"].shape[-1]) if j not in bounded]
    chk(f"out.norm == out.pre_bound outside bounded channels {bounded}",
        acts["out.norm"][..., others], acts["out.pre_bound"][..., others])
    res["bounded_channels"] = bounded
    res["bounded_channels_changed_elements"] = {
        str(j): int((acts["out.norm"][..., j] != acts["out.pre_bound"][..., j]).sum()) for j in bounded}
    z = acts["out.pre_bound"].clone()
    kinds = []
    for bd in inner.boundings[DATASET]:
        kind = type(bd).__name__
        kinds.append(kind)
        di = bd.data_index.long()
        if kind == "ReluBounding":
            z[..., di] = torch.relu(z[..., di])
        elif kind == "HardtanhBounding":
            z[..., di] = torch.clamp(z[..., di], bd.min_val, bd.max_val)
        elif kind == "FractionBounding":
            z[..., di] = torch.clamp(z[..., di], bd.min_val, bd.max_val)
            z[..., di] *= z[..., bd.total_variable.long()]
        else:
            raise NotImplementedError(kind)
    chk(f"out.norm == bounding(out.pre_bound) {kinds}", acts["out.norm"], z)
    # out.y = (out.norm - add[out_idx]) / mul[out_idx]  (normalizer.py: subtract_ then div_)
    chk("out.y == (out.norm - add) / mul", acts["out.y"],
        acts["out.norm"].clone().subtract_(add[out_idx]).div_(mul[out_idx]))
    return res


def expected_shapes(n_grid: int, n_hidden: int, c: int, n_in: int, n_out: int, n_proc: int, n_steps: int,
                    attr_data: int, attr_hidden: int, n_out_steps: int = 1, batch: int = 1) -> dict:
    """Contract shapes (docs/format.md §2); node dimensions flattened batch-major; T_out = n_out_steps."""
    bg, bh = batch * n_grid, batch * n_hidden
    s = {"in.x": (batch, n_steps, n_grid, n_in), "in.x_norm": (batch, n_steps, 1, n_grid, n_in),
         "enc.x_src": (bg, n_steps * n_in + attr_data), "enc.x_dst": (bh, attr_hidden), "enc.emb_src": (bg, c),
         "enc.emb_dst": (bh, c), "enc.out": (bh, c), "latent": (bh, c), "dec.emb_dst": (bg, c),
         "dec.block_out": (bg, c), "dec.out": (bg, n_out_steps * n_out),
         "out.pre_bound": (batch, n_out_steps, 1, n_grid, n_out), "out.norm": (batch, n_out_steps, 1, n_grid, n_out),
         "out.y": (batch, n_out_steps, 1, n_grid, n_out)}
    s.update({f"proc.{i}.out": (bh, c) for i in range(n_proc)})
    return s


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def golden_main(args) -> None:
    """--golden TAG: write the contract activations of one configuration (device, backend, autocast) to --out."""
    import contextlib

    import anemoi.models.layers.mapper as mapper_mod
    from compare_golden import checkpoint_identity, contract_keys, required_keys

    device, dtype, backend = torch.device(args.device), torch.float32, args.backend
    assert args.dtype == "float32", "--golden: weights and input stay fp32 (use --autocast fp16 for fp16)"
    amp = ((lambda: torch.autocast(device_type=device.type, dtype=torch.float16)) if args.autocast == "fp16"
           else contextlib.nullcontext)
    strict = args.autocast == "none"
    if device.type == "cuda":
        assert torch.cuda.is_available()
    out = Path(args.out or f"{OUT_ROOT}/{args.golden}")
    out.mkdir(parents=True, exist_ok=True)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    chunk_env = {k: os.environ.get(k) for k in CHUNK_ENV}
    assert all(v is None for v in chunk_env.values()), f"unset these for golden runs: {chunk_env}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    t0 = time.perf_counter()
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t_load = time.perf_counter() - t0
    changed = set_attention_backend(model, backend)
    model = model.to(dtype=dtype).eval()  # moved to the device after make_input (which indexes with CPU tensors)
    assert not any(m.training for m in model.modules())

    # The input is always regenerated (make_input, --seed); an existing --input must be bitwise equal to it, so the
    # golden set stays reproducible from the repository alone.
    x_gen, recipe = make_input(model, args.seed, args.ckpt)
    input_info = {"path": args.input or None, "exists": bool(args.input) and Path(args.input).exists(),
                  "regenerated_with_seed": args.seed}
    if input_info["exists"]:
        x = torch.load(args.input, weights_only=True)
        input_info["regenerated_bitwise_equal"] = bool(torch.equal(x, x_gen))
        assert input_info["regenerated_bitwise_equal"], f"{args.input} differs from make_input(seed={args.seed})"
    else:
        print(f"WARNING: {args.input} not found, using make_input(seed={args.seed})", flush=True)
        x = x_gen
    if args.input_only:  # write the shared input once (input.pt + input.json), then stop
        save_atomic(x, out / "input.pt")
        write_text_atomic(out / "input.json", json.dumps({"checkpoint_identity": checkpoint_identity(args.ckpt),
                                                         "input_recipe": recipe, "shape": list(x.shape),
                                                         "sha256_fp32_bytes": sha256_tensor(x)}, indent=1) + "\n")
        print(f"wrote {out / 'input.pt'} {tuple(x.shape)}")
        return
    in_names, out_names = variable_names(model)
    n_step_out = int(getattr(model.model, "n_step_output", 1))
    ckpt_identity = checkpoint_identity(args.ckpt)
    model = model.to(device)
    x = x.to(device)
    x_ref = x.clone()

    # 1) plain run: no hooks, no patches
    t0 = time.perf_counter()
    with torch.inference_mode(), amp():
        y_plain = model.predict_step({DATASET: x})[DATASET].detach().clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_plain = time.perf_counter() - t0
    assert torch.equal(x, x_ref), "predict_step modified its input"

    # 2) capture runs; mapper chunks = max(mapper.num_chunks, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER) per mapper
    n_enc = max(model.model.encoder[DATASET].num_chunks, mapper_mod.NUM_CHUNKS_INFERENCE_MAPPER)
    n_dec = max(model.model.decoder[DATASET].num_chunks, mapper_mod.NUM_CHUNKS_INFERENCE_MAPPER)

    def capture_run():
        cap = Capture(model, amp)
        try:
            t0 = time.perf_counter()
            with torch.inference_mode(), amp():
                y = model.predict_step({DATASET: x})[DATASET]
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            acts, diag = cap.collect(x, y, n_enc, n_dec)
            dtypes = cap.captured_dtypes(y, x)
        finally:
            cap.close()
        assert torch.equal(x, x_ref), "predict_step modified its input"
        return acts, diag, dtypes, dt, cap

    acts, diag, cap_dtypes, t_cap, cap = capture_run()
    print(f"plain run {t_plain:.1f} s, capture run {t_cap:.1f} s, peak RSS {peak_rss_gb():.1f} GB", flush=True)
    determinism = None
    if args.runs > 1:  # repeat the capture in the same process; every activation must be bitwise equal
        determinism = {}
        for r in range(1, args.runs):
            acts_r, _, _, _, _ = capture_run()
            determinism[f"run{r}_vs_run0"] = {k: {"bitwise_equal": bool(torch.equal(acts[k], acts_r[k])),
                                                  "max_abs": (acts[k] - acts_r[k]).abs().max().item()}
                                              for k in acts}
            del acts_r
        print("determinism:", json.dumps({r: all(e["bitwise_equal"] for e in d.values())
                                          for r, d in determinism.items()}), flush=True)

    # 3) structure: exactly the contract keys, full shapes, fp32/CPU/contiguous, finite
    inner = model.model
    n_proc = len(inner.processor.proc)
    keys = contract_keys(n_proc)
    na = node_attributes(inner)
    lat_d, lat_h = na.latlons_data, na.latlons_hidden
    shapes = expected_shapes(x.shape[2], lat_h.shape[0], inner.num_channels, x.shape[3], len(out_names), n_proc,
                             int(model.n_step_input), lat_d.shape[1], lat_h.shape[1], n_step_out, batch=x.shape[0])
    assert set(acts) == set(keys) == set(shapes), set(acts) ^ set(keys)
    acts = {k: acts[k] for k in keys}
    for k, t in acts.items():
        assert tuple(t.shape) == shapes[k], (k, tuple(t.shape), shapes[k])
        assert t.dtype == torch.float32 and t.device.type == "cpu" and t.is_contiguous() and not t.requires_grad, k
        assert torch.isfinite(t).all(), f"non-finite values in {k}"

    # 4) checks
    checks = {"out.y == plain run (no capture) bitwise": torch.equal(acts["out.y"], cpu_copy(y_plain))}
    _SEEN_DTYPES.clear()
    consistency = consistency_checks(model, acts, strict, amp)
    print(json.dumps({"checks": checks, "consistency": consistency, "diagnostics": diag}, indent=1), flush=True)
    failed = [n for n, ok in checks.items() if not ok] if device.type == "cpu" and strict else []
    failed += [n for n, m in consistency.items()
               if isinstance(m, dict) and m.get("expect_bitwise") and not m["bitwise_equal"]]
    if determinism is not None:
        failed += [f"determinism {r} {k}" for r, d in determinism.items() for k, e in d.items() if not e["bitwise_equal"]]

    save_atomic(acts, out / "activations.pt")
    manifest = {
        "tag": args.golden,
        "contract": "docs/format.md §2",
        "script": "python/reference/run_reference.py --golden",
        "command": " ".join(["python", *sys.argv]),
        "checkpoint": args.ckpt,
        "checkpoint_identity": ckpt_identity,
        "input": {**input_info, "sha256_fp32_bytes": sha256_tensor(acts["in.x"])},
        "input_recipe": recipe,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "host": socket.gethostname(),
        "dtype": "float32",
        "autocast": None if strict else {"device_type": device.type, "dtype": "torch.float16",
                                         "scope": "predict_step (plain and capture runs), under inference_mode"},
        "stored_dtype": "float32 (cast on CPU after capture; original dtypes in tensors.*.captured_dtype)",
        "backend": backend,
        "backend_override": {"blocks": changed, "set": (
            "graph_attention_backend='pyg', conv=GraphTransformerConv(out_channels=block.out_channels_conv)"
            if backend == "pyg" else "graph_attention_backend='triton', conv=GraphTransformerFunction.apply")},
        "model_class": f"{type(inner).__module__}.{type(inner).__name__}",
        "n_output_steps": n_step_out,
        "model_training_flag_after_eval": model.training,
        "torch_num_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "chunk_env": chunk_env,
        "env": {k: os.environ.get(k) for k in ENV_RECORD},
        "mapper_num_chunks": n_enc if n_enc == n_dec else {"encoder": n_enc, "decoder": n_dec},
        "versions": versions(),
        "anemoi_models_source": _direct_url("anemoi-models"),
        "call": "model.predict_step({'data': in.x})['data'] under torch.inference_mode()"
                + ("" if strict else " and torch.autocast(float16)"),
        "input_layout": "batch, time, grid, variable (data.input.full order)",
        "output_layout": "batch, time, ensemble, grid, variable (data.output.full order)",
        "input_variables": in_names,
        "output_variables": out_names,
        "keys": list(acts),
        "keys_missing": [k for k in keys if k not in acts],
        "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype), "captured_dtype": cap_dtypes[k],
                        "required": k in required_keys(n_proc), "absmax": t.abs().max().item(),
                        "sha256": sha256_tensor(t), "capture": cap.how[k]} for k, t in acts.items()},
        "sha256_out.y": sha256_tensor(acts["out.y"]),
        "checks": checks,
        "consistency": consistency,
        "diagnostics": diag,
        **({"determinism_runs": args.runs, "determinism": determinism} if determinism is not None else {}),
        "failed_checks": failed,
        "timing_s": {"load": t_load, "forward_plain": t_plain, "forward_capture": t_cap},
        "memory": {"peak_rss_gb": peak_rss_gb(),
                   **({"cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3}
                      if device.type == "cuda" else {})},
        "file_size_bytes": (out / "activations.pt").stat().st_size,
    }
    write_text_atomic(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out} ({manifest['file_size_bytes'] / 1e9:.2f} GB)")
    if failed:
        raise SystemExit(f"FAILED checks: {failed}")


def _direct_url(dist: str):
    try:
        text = importlib.metadata.distribution(dist).read_text("direct_url.json")
        return json.loads(text) if text else None
    except importlib.metadata.PackageNotFoundError:
        return None


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
    ap.add_argument("--golden", metavar="TAG", default=None,
                    help="contract mode: write activations.pt + manifest.json (docs/format.md §2) to --out "
                         f"(default {OUT_ROOT}/TAG); --runs = capture runs (>1: all activations must be bitwise equal)")
    ap.add_argument("--autocast", default="none", choices=["none", "fp16"],
                    help="--golden: torch.autocast(device, float16) around predict_step, as anemoi-inference runs it")
    ap.add_argument("--input-only", action="store_true",
                    help="--golden: only write the synthetic input (input.pt, input.json) to --out, for the other tags")
    args = ap.parse_args()
    if args.golden:
        return golden_main(args)

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
