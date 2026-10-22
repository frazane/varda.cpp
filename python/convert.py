"""Convert an anemoi checkpoint into one `model.safetensors` (docs/format.md section 1).

    .venv/bin/python -I python/convert.py CKPT OUT.safetensors

Port of the Python project's engine/convert.py (docs/contracts.md section 1 there): same mapping, same
strict accounting, same config; only the container differs (safetensors, config JSON in __metadata__).

The checkpoint is read only through `checkpoint.py` (restricted unpickler); the zip's
`anemoi-metadata/anemoi.json` is read with `zipfile` for provenance fields (uuid, run_id, commit)
only, and the file's sha256 is recorded. Every config value comes from the checkpoint's modules and
tensors. Every tensor reachable in the checkpoint (deduplicated by identity) is converted, exported into
`config.json`, or listed in `config.json["unused"]` with a reason; anything else makes the conversion fail.

Supported limits (one dataset named "data", one output step, the model / component / block / residual /
bounding classes in the tables below) are checked up front by `check_supported`, which fails with a pointer
to the Python project's docs/new-checkpoint.md §2 and §5.
"""

import datetime
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not add the script dir

import checkpoint as ck  # noqa: E402
import safetensors_io as st_io  # noqa: E402

# --------------------------------------------------------------------------------------------
# Mapping table: checkpoint path -> contract name. Specific to the anemoi GraphTransformer
# EncProcDec family; everything below the table is generic.
# --------------------------------------------------------------------------------------------

DATASET = "data"  # dataset key used throughout the checkpoint paths (checked by check_supported)
SUPPORTED_DOC = "$IE_ROOT/docs/new-checkpoint.md §2 (what is supported) and §5 (adding a component kind)"

# Plain tensors: contract name -> checkpoint path.
TENSORS = {
    "norm.mul": "pre_processors.data.processors.normalizer._norm_mul",
    "norm.add": "pre_processors.data.processors.normalizer._norm_add",
    "nodes.data.attr": "model.node_attributes.latlons_data",
    "nodes.hidden.attr": "model.node_attributes.latlons_hidden",
}

# Linear / LayerNorm modules outside the blocks: contract name -> (module path, kind) (.weight, .bias).
MODULES = {
    "enc.emb_src": ("model.encoder.data.emb_nodes_src", "Linear"),
    "enc.emb_dst": ("model.encoder.data.emb_nodes_dst", "Linear"),
    "dec.emb_dst": ("model.decoder.data.emb_nodes_dst", "Linear"),
    "dec.out_norm": ("model.decoder.data.node_data_extractor.0", "LayerNorm"),
    "dec.out_proj": ("model.decoder.data.node_data_extractor.1", "Linear"),
}

# Graphs: contract prefix -> (provider module path, HeteroData edge-store key of the same edges).
GRAPHS = {
    "enc": ("model.encoder_graph_provider.data", "(data,to,hidden)"),
    "proc": ("model.processor_graph_provider", "(hidden,to,hidden)"),
    "dec": ("model.decoder_graph_provider.data", "(hidden,to,data)"),
}
GRAPH_EDGE_INDEX = "edge_index_base"  # provider buffer (int32) -> graph.X.edge_index (int64)
GRAPH_EDGE_ATTR = "edge_attr"
GRAPH_EDGE_INC = "edge_inc"
HETERO = "model._graph_data._edge_store_dict.{key}._mapping"  # .edge_index, .edge_length, .edge_dirs

# Transformer blocks: contract prefix -> checkpoint module path ({i} = layer index).
BLOCKS = {
    "enc.block": "model.encoder.data.proc",
    "proc.{i}": "model.processor.proc.{i}",
    "dec.block": "model.decoder.data.proc",
}
PROCESSOR = "model.processor"  # holds num_layers; its `proc` ModuleList holds the blocks

# Architecture (config.json "architecture", format v3): checkpoint class -> component kind. The engine dispatches
# on the kinds; a class missing here is rejected by check_supported() before anything is converted.
MODEL_KINDS = {"anemoi.models.models.encoder_processor_decoder.AnemoiModelEncProcDec": "enc_proc_dec"}
COMPONENTS = {  # part -> (module path, {class: kind})
    "encoder": ("model.encoder.data",
                {"anemoi.models.layers.mapper.GraphTransformerForwardMapper": "graph_transformer_forward_mapper"}),
    "processor": ("model.processor",
                  {"anemoi.models.layers.processor.GraphTransformerProcessor": "graph_transformer_processor"}),
    "decoder": ("model.decoder.data",
                {"anemoi.models.layers.mapper.GraphTransformerBackwardMapper": "graph_transformer_backward_mapper"}),
}
BLOCK_KINDS = {  # block class -> (kind, is_mapper)
    "anemoi.models.layers.block.GraphTransformerMapperBlock": ("graph_transformer_mapper_block", True),
    "anemoi.models.layers.block.GraphTransformerProcessorBlock": ("graph_transformer_processor_block", False),
}

# Per-block modules: contract suffix -> (module name, kind, mapper-only).
BLOCK_MODULES = {
    "q": ("lin_query", "Linear", False),
    "k": ("lin_key", "Linear", False),
    "v": ("lin_value", "Linear", False),
    "self": ("lin_self", "Linear", False),
    "edge": ("lin_edge", "Linear", False),
    "proj": ("projection", "Linear", False),
    "ln_dst": ("layer_norm_attention", "LayerNorm", False),
    "ln_src": ("layer_norm_attention_src", "LayerNorm", True),
    "ln_mlp": ("layer_norm_mlp_dst", "LayerNorm", False),
    "mlp1": ("node_dst_mlp.0", "Linear", False),
    "mlp2": ("node_dst_mlp.2", "Linear", False),
}
BLOCK_ALIASES = {"layer_norm_attention_dest": "layer_norm_attention"}  # same module, asserted
BLOCK_GELU = "node_dst_mlp.1"

# Index tensors exported to config.json["indices"] (read, not converted).
INDEX_SOURCES = {
    "norm_input": "pre_processors.data.processors.normalizer._input_idx",
    "norm_output": "pre_processors.data.processors.normalizer._output_idx",
    "residual_input": "model._internal_input_idx.data",
    "residual_output": "model._internal_output_idx.data",
}
# Cross-checks against the IndexCollection (dataset-order index lists).
INDEX_CHECKS = {
    "norm_input": "pre_processors.data.processors.normalizer.data_indices.data.input.full",
    "norm_output": "pre_processors.data.processors.normalizer.data_indices.data.output.full",
}
# Model-order name -> index maps of the IndexCollection (`root.data_indices[DATASET].model`). anemoi keeps them
# in private attributes; they are read only through model_name_to_index().
INDEX_COLLECTION_NAME_MAPS = {"input": "_name_to_index_model_input", "output": "_name_to_index_model_output"}

# Bounding module class -> config "type"; and its attribute -> config key.
BOUNDING_TYPES = {
    "anemoi.models.layers.bounding.ReluBounding": ("relu", {}),
    "anemoi.models.layers.bounding.HardtanhBounding": ("hardtanh", {"min_val": "min", "max_val": "max"}),
    "anemoi.models.layers.bounding.FractionBounding": (
        "fraction",
        {"min_val": "min", "max_val": "max", "total_variable": "total_index"},
    ),
}
BOUNDINGS = "model.boundings.data"
RESIDUAL = "model.residual.data"
RESIDUAL_TYPES = {"anemoi.models.layers.residual.SkipConnection": "skip"}
# Mapper modules whose `num_chunks` (destination-node chunks, mapper.py:383-385) goes to config.json "mappers".
MAPPERS = {"enc": "model.encoder.data", "dec": "model.decoder.data"}
NORMALIZER = "pre_processors.data.processors.normalizer"
FORMAT_VERSION = 3  # v2 = v1 + "mappers"; v3 = v2 + "architecture", "blocks", "absent", source.sha256,
#                      dims.n_output_steps (docs/contracts.md §1 "Versions")

# The raw `statistics` are not necessarily the ones folded into the normaliser buffers (varda: the
# buffers were kept from the forked run, `update_ds_stats_on_ckpt_load.states: false`). check_statistics()
# picks the reason; the forward pass only ever reads the buffers.
STATS_FOLDED_REASON = "raw dataset statistics (fp64); norm.mul/norm.add are already derived from them"
STATS_STALE_REASON = ("raw dataset statistics (fp64); NOT the statistics folded into norm.mul/norm.add "
                      "(max rel. difference {err:.3g}); unused by the forward pass, which reads only the buffers")

# Checkpoint tensors that are deliberately not converted: (regex on the primary path, reason).
# First match wins. Anything reachable that matches no rule and wasn't converted is an error.
UNUSED_RULES = [
    (r"\.edge_inc$", "per-batch index increments [[n_src],[n_dst]]; the engine handles batching itself"),
    (r"^pre_processors_tendencies\.|^statistics_tendencies\.",
     "tendency normaliser/statistics; model is not a tendency model (SkipConnection residual), unused in forward"),
    (r"^statistics\.", STATS_FOLDED_REASON),
    (r"(^|\.)data_indices\.",
     "index list of the IndexCollection; the engine's lists are exported in config.json indices"),
    (r"\._decoding_forcing_input_idx", "only read by the autoencoder model, unused by EncProcDec"),
    (r"\._graph_data\._node_store_dict\.",
     "training-graph node store (raw coordinates / area weights); forward uses model.node_attributes"),
]


# --------------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------------


class ConvertError(RuntimeError):
    pass


def check(cond, msg):
    if not cond:
        raise ConvertError(msg)


def _match_key(d, part):
    for k in d:
        s = "(" + ",".join(map(str, k)) + ")" if isinstance(k, tuple) else str(k)
        if s == part:
            return k
    raise KeyError(part)


def get(obj, path):
    """Resolve a dotted checkpoint path through modules, stub attributes, dicts and lists."""
    for part in path.split(".") if path else []:
        if isinstance(obj, ck.Stub):
            d = obj.__dict__
            if ck.is_module(obj):
                for slot in ("_modules", "_parameters", "_buffers"):
                    if part in d[slot]:
                        obj = d[slot][part]
                        break
                else:
                    check(part in d, f"no attribute {part!r} while resolving {path!r}")
                    obj = d[part]
            elif part in d:
                obj = d[part]
            else:
                obj = d["_stub_items"][_match_key(d["_stub_items"], part)]
        elif isinstance(obj, dict):
            obj = obj[_match_key(obj, part)]
        elif isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        else:
            raise ConvertError(f"cannot resolve {part!r} of {path!r}")
    return obj


def plain(obj):
    return ck.to_plain(obj)


class Accounting:
    """Tracks which checkpoint tensors (by identity) were converted, exported or declared unused."""

    def __init__(self, root):
        self.entries = ck.tensor_entries(root)  # id -> [path, tensor, kind, aliases]
        self.converted = {}  # id -> contract name(s)
        self.exported = {}  # id -> reason
        self.duplicates = {}  # id -> reason
        self.module_aliases = ck.aliases(root)

    def _entry(self, t, what):
        check(id(t) in self.entries, f"{what}: tensor not reachable through checkpoint.tensor_entries")
        return self.entries[id(t)]

    def convert(self, name, t, path):
        e = self._entry(t, path)
        self.converted.setdefault(id(t), []).append(name)
        return e

    def export(self, t, path, reason):
        self._entry(t, path)
        self.exported[id(t)] = reason

    def duplicate(self, t, path, reason):
        self._entry(t, path)
        self.duplicates[id(t)] = reason

    def unused(self, rules=UNUSED_RULES):
        out = []
        for tid, (path, t, kind, aliases) in self.entries.items():
            if tid in self.converted:
                continue
            if tid in self.duplicates:
                reason = self.duplicates[tid]
            elif tid in self.exported:
                reason = self.exported[tid]
            else:
                reason = next((r for pat, r in rules if re.search(pat, path)), None)
                if reason is None:
                    reason = next(
                        (r for a in aliases for pat, r in rules if re.search(pat, a)), None
                    )
            out.append((path, reason))
        missing = [p for p, r in out if r is None]
        check(not missing, f"{len(missing)} checkpoint tensors unaccounted for:\n  " + "\n  ".join(sorted(missing)))
        return [{"path": p, "reason": r} for p, r in sorted(out)]


def contiguous_copy(t):
    return t.detach().clone(memory_format=torch.contiguous_format)


# --------------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------------


def convert_linear(acc, root, weights, absent, name, mod_path, kind):
    """A Linear or LayerNorm (``kind``) -> ``{name}.weight`` / ``{name}.bias``. A missing bias (``bias=False``)
    is recorded in ``absent`` (config.json "absent"); a missing weight (LayerNorm without affine) is rejected."""
    mod = get(root, mod_path)
    check(mod._qualname.endswith("." + kind), f"{mod_path} is {mod._qualname}, expected {kind}")
    params = mod.__dict__["_parameters"]
    check(params.get("weight") is not None,
          f"{mod_path}: {kind} without a weight (e.g. LayerNorm(elementwise_affine=False)) is not supported by the "
          f"contract; see {SUPPORTED_DOC}")
    for p in ("weight", "bias"):
        t = params.get(p)
        if t is None:  # Linear(bias=False) / LayerNorm(bias=False): no tensor; the engine reads it as absent
            absent.append(f"{name}.{p}")
            continue
        check(t.dtype == torch.float32, f"{mod_path}.{p}: dtype {t.dtype}, expected float32")
        acc.convert(f"{name}.{p}", t, f"{mod_path}.{p}")
        weights[f"{name}.{p}"] = contiguous_copy(t)
    return mod


def block_paths(root):
    """[(contract prefix, checkpoint path, block kind, is_mapper)], processor layer count from the checkpoint."""
    n_layers = int(get(root, PROCESSOR).num_layers)
    check(len(get(root, PROCESSOR + ".proc")._modules) == n_layers, "processor.num_layers != len(proc)")
    out = []
    for prefix, path in BLOCKS.items():
        in_proc = "{i}" in prefix
        for pre, pth in ([(prefix.format(i=i), path.format(i=i)) for i in range(n_layers)] if in_proc
                         else [(prefix, path)]):
            blk = get(root, pth)
            check(blk._qualname in BLOCK_KINDS,
                  f"{pth}: block class {blk._qualname} is not supported (known: {sorted(BLOCK_KINDS)}); "
                  f"see {SUPPORTED_DOC}")
            kind, is_mapper = BLOCK_KINDS[blk._qualname]
            check(is_mapper != in_proc,
                  f"{pth}: {kind} where a {'processor' if in_proc else 'mapper'} block is expected")
            check(is_mapper == ("layer_norm_attention_src" in blk._modules), f"{pth}: layer_norm_attention_src")
            out.append((pre, pth, kind, is_mapper))
    return out, n_layers


def convert_tensors(root, acc):
    weights, absent = {}, []
    for name, path in TENSORS.items():
        t = get(root, path)
        acc.convert(name, t, path)
        weights[name] = contiguous_copy(t)
    for name, (path, kind) in MODULES.items():
        convert_linear(acc, root, weights, absent, name, path, kind)

    # Graphs
    graph_nodes = {}
    for g, (provider, hetero_key) in GRAPHS.items():
        ei = get(root, f"{provider}.{GRAPH_EDGE_INDEX}")
        ea = get(root, f"{provider}.{GRAPH_EDGE_ATTR}")
        inc = get(root, f"{provider}.{GRAPH_EDGE_INC}").reshape(-1).tolist()
        check(ei.dtype in (torch.int32, torch.int64) and ei.dim() == 2 and ei.shape[0] == 2, f"{g}: edge_index")
        check(ea.dtype == torch.float32 and ea.shape[0] == ei.shape[1], f"{g}: edge_attr shape/dtype")
        check(int(ei[0].max()) < inc[0] and int(ei[1].max()) < inc[1] and int(ei.min()) >= 0,
              f"{g}: edge_index out of range of edge_inc {inc}")
        acc.convert(f"graph.{g}.edge_index", ei, provider)
        acc.convert(f"graph.{g}.edge_attr", ea, provider)
        weights[f"graph.{g}.edge_index"] = ei.detach().to(torch.int64).contiguous()
        weights[f"graph.{g}.edge_attr"] = contiguous_copy(ea)
        graph_nodes[g] = tuple(inc)
        # The HeteroData copy must equal the provider buffers.
        hpath = HETERO.format(key=hetero_key)
        h_ei = get(root, hpath + ".edge_index")
        h_len = get(root, hpath + ".edge_length")
        h_dirs = get(root, hpath + ".edge_dirs")
        check(torch.equal(h_ei.to(torch.int64), weights[f"graph.{g}.edge_index"]), f"{g}: HeteroData edge_index differs")
        check(torch.equal(torch.cat([h_len, h_dirs], -1), ea), f"{g}: HeteroData edge_length|edge_dirs != edge_attr")
        acc.duplicate(h_ei, hpath, f"duplicate of graph.{g}.edge_index (HeteroData copy), asserted equal")
        acc.duplicate(h_len, hpath, f"duplicate of graph.{g}.edge_attr[:, :1] (HeteroData copy), asserted equal")
        acc.duplicate(h_dirs, hpath, f"duplicate of graph.{g}.edge_attr[:, 1:] (HeteroData copy), asserted equal")

    # Blocks
    blocks, n_layers = block_paths(root)
    for prefix, path, _, is_mapper in blocks:
        blk = get(root, path)
        for suffix, (mod_name, kind, mapper_only) in BLOCK_MODULES.items():
            if mapper_only and not is_mapper:
                check(mod_name not in blk._modules, f"{path}: unexpected {mod_name}")
                continue
            convert_linear(acc, root, weights, absent, f"{prefix}.{suffix}", f"{path}.{mod_name}", kind)
        for alias, target in BLOCK_ALIASES.items():
            if alias in blk._modules:
                check(blk._modules[alias] is blk._modules[target], f"{path}.{alias} is not {target}")
    return weights, absent, blocks, n_layers, graph_nodes


def block_dims(root, blocks):
    """Hyperparameters read from the blocks; asserted identical across all of them."""
    seen = {}
    for prefix, path, *_ in blocks:
        blk = get(root, path)
        vals = {
            "num_heads": int(blk.num_heads),
            "head_dim": int(blk.out_channels_conv),
            "attn_channels": int(blk.attn_channels),
            "qk_norm": bool(blk.qk_norm),
            "update_src_nodes": bool(blk.update_src_nodes),
            "mlp_hidden": int(get(blk, BLOCK_MODULES["mlp1"][0]).out_features),
            "edge_attr_dim": int(get(blk, BLOCK_MODULES["edge"][0]).in_features),
            "gelu_approximate": str(get(blk, BLOCK_GELU).approximate),
        }
        eps = {float(get(blk, m[0]).eps) for m in BLOCK_MODULES.values() if m[1] == "LayerNorm" and m[0] in blk._modules}
        check(len(eps) == 1, f"{path}: LayerNorm eps differ {eps}")
        vals["layer_norm_eps"] = eps.pop()
        for k, v in vals.items():
            check(seen.setdefault(k, v) == v, f"{path}: {k}={v} differs from other blocks ({seen[k]})")
    check(not seen["qk_norm"], f"qk_norm=True is not supported by the contract; see {SUPPORTED_DOC}")
    check(not seen["update_src_nodes"], f"update_src_nodes=True is not supported by the contract; see {SUPPORTED_DOC}")
    # LayerNorms outside the blocks (the decoder's output head) must share the one eps the contract records.
    for path, kind in MODULES.values():
        if kind == "LayerNorm":
            eps = float(get(root, path).eps)
            check(eps == seen["layer_norm_eps"], f"{path}: LayerNorm eps {eps} != the blocks' {seen['layer_norm_eps']}")
    check(seen["num_heads"] * seen["head_dim"] == seen["attn_channels"], "num_heads * head_dim != attn_channels")
    return seen


def check_statistics(root, data_names):
    """Max relative difference between the normaliser buffers and the same buffers recomputed from the raw
    `statistics` (InputNormalizer formulas, float64 then rounded to float32). 0 means they are the source."""
    norm = get(root, NORMALIZER)
    stats = getattr(root, "statistics", None)
    stats = stats.get(DATASET) if isinstance(stats, dict) else None
    if not stats or any(len(stats.get(k, ())) != len(data_names) for k in ("mean", "stdev", "maximum", "minimum")):
        return float("inf")
    methods, default = plain(norm.methods), str(norm.default)
    mul, add = get(norm, "_norm_mul").double(), get(norm, "_norm_add").double()
    worst = 0.0
    for i, n in enumerate(data_names):
        m = methods.get(n, default)
        mean, std, mx, mn = (float(stats[k][i]) for k in ("mean", "stdev", "maximum", "minimum"))
        exp = {"mean-std": (1 / std, -mean / std), "std": (1 / std, 0.0), "max": (1 / mx, 0.0),
               "min-max": (1 / (mx - mn), -mn / (mx - mn)), "none": (1.0, 0.0)}.get(m)
        if exp is None:
            return float("inf")
        for got, e in zip((float(mul[i]), float(add[i])), exp):
            e32 = float(torch.tensor(e, dtype=torch.float32))
            worst = max(worst, abs(got - e32) / abs(e32) if e32 else abs(got))
    return worst


def names_by_index(name_to_index, n=None):
    inv = sorted(name_to_index.items(), key=lambda kv: kv[1])
    check([i for _, i in inv] == list(range(len(inv))), "name_to_index is not a permutation of 0..n-1")
    check(n is None or len(inv) == n, f"name_to_index has {len(inv)} entries, expected {n}")
    return [k for k, _ in inv]


def check_supported(root):
    """Reject what the contract can't express before converting anything, each with a clear message.
    Returns config.json "architecture" (the classes and their component kinds)."""
    model = get(root, "model")
    check(model._qualname in MODEL_KINDS,
          f"model class {model._qualname} is not supported (known: {sorted(MODEL_KINDS)}); see {SUPPORTED_DOC}")
    datasets = list(getattr(model, "dataset_names", None) or list(root.data_indices))
    check(datasets == [DATASET],
          f"datasets {datasets}: the contract supports exactly one dataset, named {DATASET!r} (multi-dataset models "
          f"are not supported yet); see {SUPPORTED_DOC}")
    n_out_steps = int(getattr(model, "n_step_output", 1))  # older anemoi: no attribute, one output step
    check(n_out_steps == 1,
          f"n_step_output = {n_out_steps}: the contract supports one output step only; see {SUPPORTED_DOC}")
    arch = {"model": {"class": model._qualname, "kind": MODEL_KINDS[model._qualname]}}
    for part, (path, kinds) in COMPONENTS.items():
        cls = get(root, path)._qualname
        check(cls in kinds, f"{part} ({path}) is {cls}; supported: {sorted(kinds)}; see {SUPPORTED_DOC}")
        arch[part] = {"class": cls, "kind": kinds[cls]}
    res = get(root, RESIDUAL)._qualname
    check(res in RESIDUAL_TYPES, f"residual class {res} is not supported (known: {sorted(RESIDUAL_TYPES)}); "
                                 f"see {SUPPORTED_DOC}")
    for k, b in get(root, BOUNDINGS)._modules.items():
        check(b._qualname in BOUNDING_TYPES, f"bounding[{k}] class {b._qualname} is not supported (known: "
                                             f"{sorted(BOUNDING_TYPES)}); see {SUPPORTED_DOC}")
    return arch


def model_name_to_index(di, side):
    """The model-order {name: index} map ("input" or "output") of an IndexCollection. anemoi keeps these in
    private attributes (INDEX_COLLECTION_NAME_MAPS); this is the only place that reads them."""
    attr = INDEX_COLLECTION_NAME_MAPS[side]
    model = di.__dict__.get("model")
    found = sorted(k for k in model.__dict__ if k.startswith("_name_to_index")) if isinstance(model, ck.Stub) else []
    check(model is not None and attr in model.__dict__,
          f"IndexCollection.model has no {attr!r} (found {found}): anemoi renamed its private index maps; update "
          f"INDEX_COLLECTION_NAME_MAPS in python/convert.py")
    return plain(model.__dict__[attr])


def build_config(root, acc, weights, absent, blocks, n_layers, graph_nodes, ckpt_path, arch):
    model = get(root, "model")
    dims = block_dims(root, blocks)
    C = int(model.num_channels)
    check(dims["attn_channels"] == C, "attn_channels != num_channels")

    n_input_steps = int(model.n_step_input)
    check(int(root.n_step_input) == n_input_steps, "root.n_step_input != model.n_step_input")
    n_in = int(model.num_input_channels[DATASET])
    n_out = int(model.num_output_channels[DATASET])
    n_nodes = {k: int(v) for k, v in get(model, "node_attributes").num_nodes.items()}
    attr_ndims = {k: int(v) for k, v in get(model, "node_attributes").attr_ndims.items()}
    check(attr_ndims["data"] == attr_ndims["hidden"], "node attribute dims differ")
    node_attr_dim = attr_ndims["data"]

    # Variable names
    di = root.data_indices[DATASET]
    data_names = names_by_index(plain(di.name_to_index))
    in_names = names_by_index(model_name_to_index(di, "input"), n_in)
    out_names = names_by_index(model_name_to_index(di, "output"), n_out)

    dim_checks = {
        "emb_src in": (weights["enc.emb_src.weight"].shape[1], n_input_steps * n_in + node_attr_dim),
        "dec.emb_dst in": (weights["dec.emb_dst.weight"].shape[1], n_input_steps * n_in + node_attr_dim),
        "enc.emb_dst in": (weights["enc.emb_dst.weight"].shape[1], node_attr_dim),
        "emb_src out": (weights["enc.emb_src.weight"].shape[0], C),
        "out_proj": (tuple(weights["dec.out_proj.weight"].shape), (n_out, C)),
        "data nodes": (weights["nodes.data.attr"].shape[0], n_nodes["data"]),
        "hidden nodes": (weights["nodes.hidden.attr"].shape[0], n_nodes["hidden"]),
        "nodes attr dim": (weights["nodes.data.attr"].shape[1], node_attr_dim),
        "norm size": (weights["norm.mul"].shape[0], len(data_names)),
        "edge attr dim": (weights["graph.enc.edge_attr"].shape[1], dims["edge_attr_dim"]),
        "enc edge inc": (graph_nodes["enc"], (n_nodes["data"], n_nodes["hidden"])),
        "proc edge inc": (graph_nodes["proc"], (n_nodes["hidden"], n_nodes["hidden"])),
        "dec edge inc": (graph_nodes["dec"], (n_nodes["hidden"], n_nodes["data"])),
    }
    for k, (a, b) in dim_checks.items():
        check(a == b, f"dimension cross-check {k}: {a} != {b}")

    # Indices
    indices = {}
    for key, path in INDEX_SOURCES.items():
        t = get(root, path)
        acc.export(t, path, f"index list exported to config.json indices.{key}")
        indices[key] = [int(v) for v in t.tolist()]
    for key, path in INDEX_CHECKS.items():
        check([int(v) for v in get(root, path).tolist()] == indices[key], f"{key} differs from {path}")
    check(len(indices["norm_input"]) == n_in and len(indices["norm_output"]) == n_out, "index list sizes")
    check([data_names[i] for i in indices["norm_input"]] == in_names, "norm_input does not match input names")
    check([data_names[i] for i in indices["norm_output"]] == out_names, "norm_output does not match output names")
    check([in_names[i] for i in indices["residual_input"]] == [out_names[i] for i in indices["residual_output"]],
          "residual input/output indices name different variables")

    # Residual
    res = get(root, RESIDUAL)  # class checked by check_supported
    residual = {"type": RESIDUAL_TYPES[res._qualname], "step": int(res.step)}

    # Bounding, in application order (ModuleList order)
    bounding = []
    bmods = get(root, BOUNDINGS)._modules
    for k in sorted(bmods, key=int):
        b = bmods[k]
        typ, attrs = BOUNDING_TYPES[b._qualname]
        entry = {"type": typ}
        idx = b.data_index
        acc.export(idx, f"{BOUNDINGS}.{k}.data_index", f"index exported to config.json bounding[{k}].index")
        entry["index"] = [int(v) for v in idx.tolist()]
        for attr, key in attrs.items():
            v = getattr(b, attr) if attr in b.__dict__ else None
            if isinstance(v, torch.Tensor):
                acc.export(v, f"{BOUNDINGS}.{k}.{attr}", f"index exported to config.json bounding[{k}].{key}")
                v = [int(x) for x in v.tolist()]
            else:
                v = float(v)
            entry[key] = v
        n2i = plain(b.name_to_index)
        check(n2i == model_name_to_index(di, "output"), f"bounding[{k}] name_to_index != model output order")
        check(sorted(out_names[i] for i in entry["index"]) == sorted(plain(b.variables)), f"bounding[{k}] variables/index mismatch")
        if "total_index" in entry:
            check(len(entry["total_index"]) == 1, "fraction bounding: expected one total variable")
        bounding.append(entry)

    # Mappers: destination-node chunk count (mapper.py:383, max(num_chunks, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER);
    # the env var must be unset for reference runs, so the module attribute is the count). Mirrored by the
    # engine for bit-parity: chunking changes GEMM row counts, not the math.
    mappers = {}
    for g, path in MAPPERS.items():
        n = int(get(root, path).num_chunks)
        check(n >= 1, f"{path}.num_chunks = {n}")
        mappers[g] = {"num_chunks": n}

    # Provenance and identity (uuid, run_id, size, sha256: docs/contracts.md §1)
    meta = ck.anemoi_metadata(ckpt_path)
    ident = ck.checkpoint_identity(ckpt_path)
    commit = meta["provenance_training"]["git_versions"]["anemoi.models"]["git"]["sha1"]
    st = Path(ckpt_path).stat()

    return {
        "format_version": FORMAT_VERSION,
        "source": {
            "checkpoint": ident["path"],
            "size_bytes": ident["size_bytes"],
            "sha256": ident["sha256"],
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
            "uuid": ident["uuid"],
            "run_id": ident["run_id"],
            "interface_id": str(root.id),
            "anemoi_core_commit": commit,
        },
        "architecture": arch,
        "dims": {
            "num_channels": C,
            "num_heads": dims["num_heads"],
            "head_dim": dims["head_dim"],
            "mlp_hidden": dims["mlp_hidden"],
            "num_processor_layers": n_layers,
            "n_input_steps": n_input_steps,
            "n_output_steps": 1,  # checked by check_supported
            "n_data_vars": len(data_names),
            "n_input_vars": n_in,
            "n_output_vars": n_out,
            "n_data_nodes": n_nodes["data"],
            "n_hidden_nodes": n_nodes["hidden"],
            "node_attr_dim": node_attr_dim,
            "edge_attr_dim": dims["edge_attr_dim"],
            "layer_norm_eps": dims["layer_norm_eps"],
            "gelu_approximate": dims["gelu_approximate"],
            "latent_skip": bool(model.latent_skip),
        },
        "variables": {"data": data_names, "input": in_names, "output": out_names},
        "indices": indices,
        "residual": residual,
        "bounding": bounding,
        "mappers": mappers,
        "blocks": {prefix: {"kind": kind, "class": get(root, path)._qualname, "path": path}
                   for prefix, path, kind, _ in blocks},
        "absent": absent,
    }


# --------------------------------------------------------------------------------------------
# Grid coordinates for the dynamic forcings (docs/forcings.md)
# --------------------------------------------------------------------------------------------

GRID_DATASET = "data"  # key of `supporting_arrays_paths` that anemoi-inference loads for the model's dataset


def read_grid_coordinates(ckpt_path, node_attr):
    """The latitudes / longitudes anemoi-inference feeds to its computed forcings: F64 [G], degrees, as stored.

    anemoi-inference (`metadata.latitudes`, `tensors.py:121-125`) reads the supporting arrays `latitudes` and
    `longitudes` of the dataset from the checkpoint zip: the members named in
    `anemoi-metadata/anemoi.json["supporting_arrays_paths"]["data"]` (raw `np.frombuffer` buffers, no header).
    Read here with zipfile + numpy only, no unpickling. Asserted to agree (float32 precision) with the stored
    node attributes `nodes.data.attr` = [sin lat, sin lon, cos lat, cos lon], so a coordinate / node mismatch
    cannot go unnoticed.
    """
    with zipfile.ZipFile(ckpt_path) as z:
        names = [n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json")]
        check(len(names) == 1, f"expected one anemoi-metadata/anemoi.json, found {names}")
        paths = json.loads(z.read(names[0]))["supporting_arrays_paths"]
        check(GRID_DATASET in paths and isinstance(paths[GRID_DATASET], dict)
              and {"latitudes", "longitudes"} <= set(paths[GRID_DATASET]),
              "supporting_arrays_paths has no dataset-level latitudes / longitudes")
        out = {}
        for key in ("latitudes", "longitudes"):
            e = paths[GRID_DATASET][key]
            check(set(e) == {"path", "shape", "dtype"} and e["dtype"] == "float64" and len(e["shape"]) == 1,
                  f"{key}: unexpected supporting array entry {e}")
            raw = z.read(e["path"])
            a = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(e["shape"])
            check(a.nbytes == len(raw), f"{key}: size mismatch")
            out[key] = torch.from_numpy(a.copy())  # F64 [G], bytes exactly as stored
    lat, lon = out["latitudes"], out["longitudes"]
    check(lat.shape == lon.shape and lat.shape[0] == node_attr.shape[0],
          f"coordinate length {tuple(lat.shape)} / {tuple(lon.shape)} != nodes.data.attr {tuple(node_attr.shape)}")
    check(bool(torch.isfinite(lat).all() and torch.isfinite(lon).all()), "non-finite coordinates")
    check(float(lat.abs().max()) <= 90.0, "latitudes outside [-90, 90]")
    # [sin lat, sin lon, cos lat, cos lon] of the radian coordinates, stored as float32
    rad = torch.deg2rad
    ref = torch.stack([torch.sin(rad(lat)), torch.sin(rad(lon)), torch.cos(rad(lat)), torch.cos(rad(lon))], 1)
    err = float((ref - node_attr.to(torch.float64)).abs().max())
    check(err <= 1e-6, f"coordinates disagree with nodes.data.attr: max abs difference {err:.3g} (> 1e-6)")
    print(f"grid coordinates: G={lat.shape[0]}, lat [{float(lat.min()):.4f}, {float(lat.max()):.4f}], "
          f"lon [{float(lon.min()):.4f}, {float(lon.max()):.4f}]; max |trig(coords) - nodes.data.attr| = {err:.3g}")
    return {"grid.latitudes": lat, "grid.longitudes": lon}


def build_forcings(meta, in_names, residual_input_names):
    """Config section "forcings": which model inputs the rollout refreshes, from the checkpoint's metadata.

    Mirrors anemoi-inference 9505dd0 (docs/forcings.md section 1):
      dynamic  = computed_forcing and not constant_in_time  (`create_dynamic_forcings_providers`, tensors.py:331-352,
                 category "computed" excluding "constant")  -> recomputed for the new valid date every step;
      constant = constant_in_time                           (runner.py:463-466 `reset`; the roll carries them);
      prognostic = the rest, taken from the model output (already `indices.residual_*`).
    Input indices come from metadata_inference.<dataset>.data_indices.input (metadata.py:1583-1584). The timestep is
    metadata_inference.<dataset>.timesteps.timestep (metadata.py:1509-1511).
    """
    vm = meta["dataset"][DATASET]["variables_metadata"]
    inf = meta["metadata_inference"][DATASET]
    vtypes = inf["variable_types"]
    idx = {k: int(v) for k, v in inf["data_indices"]["input"].items()}
    check(idx == {n: i for i, n in enumerate(in_names)}, "metadata_inference data_indices.input != converted input names")
    check(len(inf["timesteps"]["output_relative_date_indices"]) == 1, "multi-step output is not supported")
    check(len(inf["timesteps"]["input_relative_date_indices"]) == 2, "unexpected number of input steps")
    ts = str(inf["timesteps"]["timestep"])
    m = re.fullmatch(r"(\d+)h", ts)
    check(m is not None and int(m.group(1)) > 0, f"timestep {ts!r}: only whole hours ('<N>h') are supported")
    cfg_ts = meta["config"]["data"].get("timestep")
    check(cfg_ts is None or cfg_ts == ts, f"config.data.timestep {cfg_ts!r} != metadata_inference timestep {ts!r}")

    dynamic, constant = [], []
    for n in in_names:
        v = vm[n]
        is_const, is_comp = bool(v.get("constant_in_time", False)), bool(v.get("computed_forcing", False))
        in_forcing, in_prog = n in vtypes["forcing"], n in vtypes["prognostic"]
        if is_const:
            check(in_forcing, f"{n}: constant in time but not a forcing")
            constant.append(n)
        elif is_comp:
            dynamic.append(n)
        else:
            # not recomputed, not constant: must be a prognostic input (fed back from the output)
            check(in_prog and not in_forcing,
                  f"{n}: a non-computed dynamic forcing (read from an input source) is not supported")
    check(sorted(set(in_names) - set(dynamic) - set(constant)) == sorted(residual_input_names),
          "inputs not refreshed by forcings != residual (prognostic) inputs")
    check(dynamic and set(dynamic) <= set(vtypes["forcing"]), "dynamic forcings must be forcing variables")
    return {
        "dynamic": dynamic,
        "dynamic_input_index": [idx[n] for n in dynamic],
        "constant": constant,
        "constant_input_index": [idx[n] for n in constant],
        "timestep_hours": int(m.group(1)),
        "source": "anemoi.json: dataset.data.variables_metadata[*].{computed_forcing,constant_in_time}, "
                  "metadata_inference.data.{variable_types,data_indices.input,timesteps.timestep}",
    }


def convert(ckpt_path, out_path):
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    root = ck.load(str(ckpt_path))
    arch = check_supported(root)
    acc = Accounting(root)
    weights, absent, blocks, n_layers, graph_nodes = convert_tensors(root, acc)
    config = build_config(root, acc, weights, absent, blocks, n_layers, graph_nodes, ckpt_path, arch)
    config["forcings"] = build_forcings(
        ck.anemoi_metadata(ckpt_path), config["variables"]["input"],
        [config["variables"]["input"][i] for i in config["indices"]["residual_input"]])
    stats_err = check_statistics(root, config["variables"]["data"])
    rules = UNUSED_RULES if stats_err <= 1e-6 else [
        (pat, STATS_STALE_REASON.format(err=stats_err) if r == STATS_FOLDED_REASON else r) for pat, r in UNUSED_RULES]
    config["unused"] = acc.unused(rules)  # raises if anything is unaccounted for

    n_ck = len(acc.entries)
    n_conv = len(acc.converted)
    check(n_conv + len(config["unused"]) == n_ck, "accounting mismatch")
    check(len(set(weights)) == len(weights), "duplicate contract names")

    weights.update(read_grid_coordinates(ckpt_path, weights["nodes.data.attr"]))
    check(len(set(weights)) == len(weights), "duplicate contract names (grid.*)")

    meta = {"varda_format": "1", "config": json.dumps(config)}
    nbytes = st_io.write_safetensors(out_path, weights, meta)  # temporary name, then rename
    size = out_path.stat().st_size
    assert size == nbytes
    print(f"normaliser buffers vs raw statistics: max rel. difference {stats_err:.3g}"
          + ("" if stats_err <= 1e-6 else " (statistics are NOT the normaliser's source; buffers used)"))
    print(f"{n_ck} checkpoint tensors: {n_conv} converted into {len(weights)} weights, "
          f"{len(config['unused'])} unused/exported")
    print(f"wrote {out_path} ({size / 1e6:.1f} MB)")
    return weights, config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] in ("-h", "--help"):
        print("usage: python -I python/convert.py CKPT OUT.safetensors", file=sys.stderr)
        return 2
    convert(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
