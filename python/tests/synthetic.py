#!/usr/bin/env python
"""Synthetic tiny models with an independent float64 reference: a generality test.

The two real checkpoints share one set of dimensions (C=1024, 16x64 heads, T=2, ...), so
parity runs on them cannot catch model-specific assumptions in the engine. This script
writes, per configuration in CONFIGS, into an output directory:

  <name>.model.safetensors    converted format (docs/format.md section 1), source.uuid
                              "synthetic-<name>-v1"
  <name>.golden.safetensors   float64 reference rounded to F32, golden format (section 3),
                              tag "synthetic/<name>", all contract keys for this n_proc

and, with --write-tolerances DIR, tolerances/synthetic-<name>.json (section 4; one file per
model because apps/common.h load_tolerances matches one checkpoint.uuid per file).
--check-tolerances DIR verifies that those files are what this script would write.

numpy + stdlib only: run as `.venv/bin/python -I python/tests/synthetic.py --out DIR`.
Deterministic: every random draw comes from numpy.random.default_rng(seed).

The reference follows $IE_ROOT/docs/spec-model.md and spec-interface.md, implemented
independently of the engine (no shared code, different formulation: dense per-destination
softmax in float64):
  * normalisation: x*mul rounded to fp32, then +add rounded to fp32 (two fp32 ops, no FMA;
    spec-interface 2.3), so in.x_norm and everything derived are bit-exact inputs for the engine;
  * x_data = [t0 vars | t1 vars | node attr] (time major, attributes last);
  * pre-norm block: x_r = lin_self(LN(xd)) is added BEFORE the projection, the residual is
    the un-normalised xd; the edge term e is added to BOTH k and v; softmax subtracts the
    per-destination max and divides by sum + 1e-16 (pyg); zero in-degree gives 0 attention;
  * mapper blocks use separate src/dst LayerNorms (ln_src, ln_dst), processor blocks one;
  * encoder src = emb_src(x_data), dst = emb_dst(hidden attr); decoder src = latent (NOT
    embedded), dst = emb_dst(raw x_data), its own LayerNorm + Linear after the block;
  * latent = processor(z) + z if latent_skip, else processor(z);
  * residual: y[out_idx] += x_norm[step][in_idx] (fp32 normalised input); then bounding in
    config order (a fraction bounding uses the already bounded total); then
    out.y = (y - add) / mul in the output gather.
"""
import argparse
import hashlib
import json
import math
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- safetensors (stdlib + numpy)
_DT = {np.dtype(np.float32): "F32", np.dtype(np.int64): "I64", np.dtype(np.float64): "F64"}
_NP = {"F32": np.float32, "I64": np.int64, "F64": np.float64}


def write_st(path, tensors, meta):
    """Plain safetensors, docs/format.md section 1: u64 N, JSON header space-padded so 8+N is
    a multiple of 8, packed little-endian data. Written to a temporary name, then renamed."""
    import os
    hdr, off, blobs = {"__metadata__": meta}, 0, []
    # Wide dtypes first: the packed data has no gaps, so this keeps every offset a multiple of its
    # element size (the engine's reader requires aligned tensors).
    order = sorted(tensors.items(), key=lambda kv: -kv[1].dtype.itemsize)
    for name, arr in order:
        arr = np.ascontiguousarray(arr)
        b = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        hdr[name] = {"dtype": _DT[arr.dtype], "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(data[a:b], dtype=_NP[v["dtype"]]).reshape(v["shape"])
    return out, h["__metadata__"]


# ---------------------------------------------------------------- variable layouts
def layout_small(_):
    """12 data vars, 10 inputs, 8 outputs; all three bounding types; residual pairs shuffled."""
    return dict(
        Vd=12, Vi=10, Vo=8,
        norm_input=list(range(10)), norm_output=[0, 1, 2, 3, 4, 5, 10, 11],
        res_in=[9, 0, 3, 2, 5, 7], res_out=[1, 0, 2, 3, 4, 5],
        bounding=[
            {"type": "relu", "index": [6]},
            {"type": "hardtanh", "index": [0, 7], "min": -0.5, "max": 0.75},
            {"type": "fraction", "index": [1, 2], "min": 0.0, "max": 1.0, "total_index": [0, 7]},
        ])


def layout_real(_):
    """The real envelope's index spaces: 85 data vars with the diagnostic 'tp' (data index 34)
    not an input, 13 forcings not outputs; V_in 84, V_out 72; tp = out 22 (relu), tcc = out 23
    (hardtanh 0..1), hcc/lcc/mcc = out 4,5,6 (fraction with total tcc). Same as the checkpoints."""
    forcing_in = [4, 5, 6, 7, 9, 11, 26, 27, 28, 29, 31, 32, 71]  # model-input indices
    forcing_data = [i if i < 34 else i + 1 for i in forcing_in]
    norm_input = [i for i in range(85) if i != 34]
    norm_output = [i for i in range(85) if i not in forcing_data]
    assert len(norm_input) == 84 and len(norm_output) == 72 and norm_output[22] == 34 and norm_output[23] == 35
    res_out = [o for o, d in enumerate(norm_output) if d != 34]
    res_in = [norm_output[o] if norm_output[o] < 34 else norm_output[o] - 1 for o in res_out]
    return dict(
        Vd=85, Vi=84, Vo=72, norm_input=norm_input, norm_output=norm_output, res_in=res_in, res_out=res_out,
        bounding=[
            {"type": "relu", "index": [22]},
            {"type": "hardtanh", "index": [23], "min": 0.0, "max": 1.0},
            {"type": "fraction", "index": [4, 5, 6], "min": 0.0, "max": 1.0, "total_index": [23]},
        ])


def layout_mini(_):
    """3 outputs: one relu, one hardtanh, one fraction (total = the hardtanh output)."""
    return dict(
        Vd=6, Vi=5, Vo=3, norm_input=[0, 1, 2, 3, 4], norm_output=[0, 1, 5],
        res_in=[2, 4], res_out=[0, 2],
        bounding=[
            {"type": "relu", "index": [0]},
            {"type": "hardtanh", "index": [1], "min": -0.4, "max": 0.6},
            {"type": "fraction", "index": [2], "min": 0.05, "max": 0.9, "total_index": [1]},
        ])


LAYOUTS = {"small": layout_small, "real": layout_real, "mini": layout_mini}

# C, heads, mlp, n_proc, T (n_input_steps), A (node_attr_dim), Ae (edge_attr_dim), G data nodes,
# H hidden nodes. a-d are the reviewer's configurations; e mimics the real envelope at tiny node
# counts; f has T=2 and one output variable per bounding type.
CONFIGS = {
    "a": dict(seed=1, C=256, heads=32, mlp=520, n_proc=1, T=3, A=3, Ae=2, gelu="tanh", latent_skip=True,
              res_step=-2, absent_kind=True, layout="small", G=300, H=70),
    "b": dict(seed=2, C=512, heads=2, mlp=1024, n_proc=3, T=1, A=5, Ae=1, gelu="none", latent_skip=False,
              res_step=0, absent_kind=False, layout="small", G=300, H=70),
    "c": dict(seed=3, C=768, heads=12, mlp=1536, n_proc=2, T=2, A=4, Ae=3, gelu="none", latent_skip=True,
              res_step=-1, absent_kind=False, layout="small", G=300, H=70),
    "d": dict(seed=4, C=2048, heads=8, mlp=512, n_proc=2, T=2, A=4, Ae=3, gelu="tanh", latent_skip=True,
              res_step=-1, absent_kind=True, layout="small", G=300, H=70),
    "e": dict(seed=5, C=1024, heads=16, mlp=4096, n_proc=4, T=2, A=4, Ae=3, gelu="none", latent_skip=True,
              res_step=-1, absent_kind=False, layout="real", G=301, H=67),
    "f": dict(seed=6, C=256, heads=4, mlp=640, n_proc=2, T=2, A=2, Ae=1, gelu="tanh", latent_skip=True,
              res_step=-1, absent_kind=False, layout="mini", G=173, H=41),
}


def uuid_of(name):
    return f"synthetic-{name}-v1"


# ---------------------------------------------------------------- tolerances
# Tables (docs/format.md section 4). "synthetic-fp32": engine fp32 (CPU and CUDA) vs the float64
# reference rounded to fp32. "synthetic-fp16": engine CUDA fp16 vs the same.
FP32_TOL, FP32_EXACT_TOL = 1e-5, 1e-6
FP16_TOL, FP16_EXACT_TOL = 3e-3, 1e-6
EXACT_KEYS = ("in.x", "in.x_norm", "enc.x_src", "enc.x_dst")  # copies / input assembly


def contract_keys(n_proc):
    k = ["in.x", "in.x_norm", "enc.x_src", "enc.x_dst", "enc.emb_src", "enc.emb_dst", "enc.out"]
    k += [f"proc.{i}.out" for i in range(n_proc)]
    k += ["latent", "dec.emb_dst", "dec.block_out", "dec.out", "out.pre_bound", "out.norm", "out.y"]
    return k


def tolerance_file(name):
    cfg = CONFIGS[name]
    keys = contract_keys(cfg["n_proc"])
    return {
        "method": "Synthetic generality models (python/tests/synthetic.py, docs/testing.md). The reference is an "
                  "independent float64 numpy implementation of the spec, rounded to F32. synthetic-fp32: engine fp32 "
                  "(CPU and CUDA) vs the reference, 1e-5 per key (measured worst 9.2e-7; see "
                  "docs/testing.md), 1e-6 for copies and input assembly (in.x, in.x_norm, enc.x_src, enc.x_dst; "
                  "bitwise in practice). synthetic-fp16: engine CUDA fp16 vs the same reference, single value for "
                  "all non-exact keys, about 2x the worst measured over all models and keys (docs/testing.md). "
                  "Metric: rel_absmax (docs/format.md section 2).",
        "model": f"synthetic-{name}",
        "checkpoint": {"uuid": uuid_of(name)},
        "synthetic-fp32": {k: FP32_EXACT_TOL if k in EXACT_KEYS else FP32_TOL for k in keys},
        "synthetic-fp16": {k: FP16_EXACT_TOL if k in EXACT_KEYS else FP16_TOL for k in keys},
    }


# ---------------------------------------------------------------- model generation
def make(seed, C, heads, mlp, n_proc, T, A, Ae, gelu, latent_skip, res_step, absent_kind, layout, G, H):
    rng = np.random.default_rng(seed)
    hd = C // heads
    L = LAYOUTS[layout](None)
    Vd, Vi, Vo = L["Vd"], L["Vi"], L["Vo"]
    var_data = [f"v{i}" for i in range(Vd)]
    norm_input, norm_output = L["norm_input"], L["norm_output"]
    t = {}
    t["norm.mul"] = rng.uniform(0.5, 2, Vd).astype(np.float32)
    t["norm.add"] = rng.normal(0, 0.3, Vd).astype(np.float32)
    t["nodes.data.attr"] = rng.normal(0, 1, (G, A)).astype(np.float32)
    t["nodes.hidden.attr"] = rng.normal(0, 1, (H, A)).astype(np.float32)

    def graph(name, ns, nd, deg):
        src, dst = [], []
        for d in range(nd):
            if d % 7 == 3:
                continue  # zero in-degree destinations
            k = rng.integers(1, deg + 1)
            src += list(rng.integers(0, ns, k))
            dst += [d] * k
        perm = rng.permutation(len(src))
        ei = np.stack([np.array(src)[perm], np.array(dst)[perm]]).astype(np.int64)
        t[f"graph.{name}.edge_index"] = ei
        t[f"graph.{name}.edge_attr"] = rng.normal(0, 1, (ei.shape[1], Ae)).astype(np.float32)

    graph("enc", G, H, 9)
    graph("proc", H, H, 6)
    graph("dec", H, G, 4)
    F = T * Vi + A
    absent = []
    absent_set = set()
    blocks = ["enc.block"] + [f"proc.{i}" for i in range(n_proc)] + ["dec.block"]
    if absent_kind:
        for b in blocks:
            for l in ["q", "k", "v", "self", "edge"]:
                absent_set.add(f"{b}.{l}.bias")
        absent_set.add("dec.out_norm.bias")
        absent_set.add("enc.block.ln_mlp.bias")

    def lin(name, o, i):
        t[name + ".weight"] = (rng.normal(0, 1, (o, i)) / np.sqrt(i)).astype(np.float32)
        if name + ".bias" in absent_set:
            absent.append(name + ".bias")
        else:
            t[name + ".bias"] = rng.normal(0, 0.2, o).astype(np.float32)

    def ln(name):
        t[name + ".weight"] = rng.uniform(0.5, 1.5, C).astype(np.float32)
        if name + ".bias" in absent_set:
            absent.append(name + ".bias")
        else:
            t[name + ".bias"] = rng.normal(0, 0.2, C).astype(np.float32)

    lin("enc.emb_src", C, F)
    lin("enc.emb_dst", C, A)
    lin("dec.emb_dst", C, F)
    ln("dec.out_norm")
    lin("dec.out_proj", Vo, C)
    for b in blocks:
        for l in ["q", "k", "v", "self", "proj"]:
            lin(f"{b}.{l}", C, C)
        lin(f"{b}.edge", C, Ae)
        lin(f"{b}.mlp1", mlp, C)
        lin(f"{b}.mlp2", C, mlp)
        ln(f"{b}.ln_dst")
        ln(f"{b}.ln_mlp")
        if b in ("enc.block", "dec.block"):
            ln(f"{b}.ln_src")
    cfg = {
        "format_version": 3,
        "source": {"checkpoint": "synthetic", "uuid": None},  # filled by the caller
        "architecture": {
            "model": {"kind": "enc_proc_dec"},
            "encoder": {"kind": "graph_transformer_forward_mapper"},
            "processor": {"kind": "graph_transformer_processor"},
            "decoder": {"kind": "graph_transformer_backward_mapper"}},
        "dims": {
            "num_channels": C, "num_heads": heads, "head_dim": hd, "mlp_hidden": mlp,
            "num_processor_layers": n_proc, "n_input_steps": T, "n_output_steps": 1,
            "n_data_vars": Vd, "n_input_vars": Vi, "n_output_vars": Vo,
            "n_data_nodes": G, "n_hidden_nodes": H, "node_attr_dim": A, "edge_attr_dim": Ae,
            "layer_norm_eps": 1e-5, "gelu_approximate": gelu, "latent_skip": latent_skip},
        "variables": {"data": var_data, "input": [var_data[i] for i in norm_input],
                      "output": [var_data[i] for i in norm_output]},
        "indices": {"norm_input": norm_input, "norm_output": norm_output,
                    "residual_input": L["res_in"], "residual_output": L["res_out"]},
        "residual": {"type": "skip", "step": res_step},
        "bounding": L["bounding"],
        "mappers": {"enc": {"num_chunks": 4}, "dec": {"num_chunks": 4}},
        "blocks": {b: {"kind": "graph_transformer_mapper_block" if b in ("enc.block", "dec.block")
                       else "graph_transformer_processor_block"} for b in blocks},
        "absent": absent,
        "unused": [],
    }
    x = rng.normal(0, 1, (1, T, G, Vi)).astype(np.float32)
    return cfg, t, x


# ---------------------------------------------------------------- float64 reference
def reference(cfg, t, x):
    d = cfg["dims"]
    C, heads = d["num_channels"], d["num_heads"]
    hd = C // heads
    eps = float(np.float32(d["layer_norm_eps"]))  # the engine reads a float
    tanh = d["gelu_approximate"] == "tanh"
    W = lambda n: t[n].astype(np.float64)

    def lin(n, a):
        y = a @ W(n + ".weight").T
        return y + W(n + ".bias") if n + ".bias" in t else y

    def ln(n, a):
        m = a.mean(-1, keepdims=True)
        v = ((a - m) ** 2).mean(-1, keepdims=True)  # biased variance
        y = (a - m) / np.sqrt(v + eps) * W(n + ".weight")
        return y + W(n + ".bias") if n + ".bias" in t else y

    erf = np.vectorize(math.erf, otypes=[np.float64])

    def gelu(a):
        if tanh:
            return 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a ** 3)))
        return 0.5 * a * (1 + erf(a / math.sqrt(2)))

    def block(b, gname, xs, xd, mapper):
        ei = t[f"graph.{gname}.edge_index"]
        ea = t[f"graph.{gname}.edge_attr"].astype(np.float64)
        s_ln = ln(b + (".ln_src" if mapper else ".ln_dst"), xs)
        d_ln = ln(b + ".ln_dst", xd)
        q, k, v, xr = lin(b + ".q", d_ln), lin(b + ".k", s_ln), lin(b + ".v", s_ln), lin(b + ".self", d_ln)
        e = lin(b + ".edge", ea)
        src, dst = ei[0], ei[1]
        kk = (k[src] + e).reshape(-1, heads, hd)  # edge term in k ...
        vv = (v[src] + e).reshape(-1, heads, hd)  # ... and in v
        s = (q[dst].reshape(-1, heads, hd) * kk).sum(-1) / math.sqrt(hd)
        out = np.zeros((xd.shape[0], heads, hd))
        for i in range(xd.shape[0]):
            m = dst == i
            if not m.any():
                continue  # zero in-degree: attention output 0
            si = s[m]
            a = np.exp(si - si.max(0))
            a /= a.sum(0) + 1e-16
            out[i] = (a[:, :, None] * vv[m]).sum(0)
        h = lin(b + ".proj", out.reshape(-1, C) + xr) + xd  # x_r before the projection; un-normalised residual
        return lin(b + ".mlp2", gelu(lin(b + ".mlp1", ln(b + ".ln_mlp", h)))) + h

    ni, no = cfg["indices"]["norm_input"], cfg["indices"]["norm_output"]
    mul32, add32 = t["norm.mul"], t["norm.add"]
    X = x
    xn32 = (X * mul32[ni]).astype(np.float32)  # two separate fp32 roundings (spec-interface 2.3)
    xn32 = (xn32 + add32[ni]).astype(np.float32)
    xn = xn32.astype(np.float64)
    T = xn.shape[1]
    acts = {"in.x": X.astype(np.float64), "in.x_norm": xn.reshape(1, T, 1, *xn.shape[2:])}
    xsrc = np.concatenate([xn[0, i] for i in range(T)] + [W("nodes.data.attr")], -1)
    acts["enc.x_src"] = xsrc
    acts["enc.x_dst"] = W("nodes.hidden.attr")
    es = lin("enc.emb_src", xsrc)
    ed = lin("enc.emb_dst", W("nodes.hidden.attr"))
    acts["enc.emb_src"], acts["enc.emb_dst"] = es, ed
    z = block("enc.block", "enc", es, ed, True)
    acts["enc.out"] = z
    h = z
    for i in range(d["num_processor_layers"]):
        h = block(f"proc.{i}", "proc", h, h, False)
        acts[f"proc.{i}.out"] = h
    lat = h + z if d["latent_skip"] else h
    acts["latent"] = lat
    dd = lin("dec.emb_dst", xsrc)  # decoder dst input: the raw assembled input, own embedding
    acts["dec.emb_dst"] = dd
    bo = block("dec.block", "dec", lat, dd, True)  # decoder src = latent, not embedded
    acts["dec.block_out"] = bo
    y = lin("dec.out_proj", ln("dec.out_norm", bo))
    acts["dec.out"] = y.copy()
    step = cfg["residual"]["step"]
    for ri, ro in zip(cfg["indices"]["residual_input"], cfg["indices"]["residual_output"]):
        y[:, ro] += xn[0, step][:, ri]
    y_oshape = (1, 1, 1) + y.shape
    acts["out.pre_bound"] = y.copy().reshape(y_oshape)
    for bd in cfg["bounding"]:  # config order
        idx = bd["index"]
        if bd["type"] == "relu":
            y[:, idx] = np.maximum(y[:, idx], 0)
        else:
            c = np.clip(y[:, idx], bd["min"], bd["max"])
            if bd["type"] == "fraction":
                c = c * y[:, bd["total_index"]]  # total broadcasts if it is a single index; already bounded
            y[:, idx] = c
    acts["out.norm"] = y.copy().reshape(y_oshape)
    acts["out.y"] = ((y - W("norm.add")[no]) / W("norm.mul")[no]).reshape(y_oshape)
    return acts


# ---------------------------------------------------------------- driver
def generate(name, out):
    c = CONFIGS[name]
    cfg, t, x = make(**c)
    cfg["source"]["uuid"] = uuid_of(name)
    ref = reference(cfg, t, x)
    n_proc = c["n_proc"]
    keys = contract_keys(n_proc)
    assert set(ref) == set(keys), set(ref) ^ set(keys)
    gold = {k: ref[k].astype(np.float32) for k in keys}  # float64 reference rounded to F32
    for k, a in gold.items():
        assert np.isfinite(a).all(), k
    # the golden in.x is the model input exactly (float32)
    assert np.array_equal(gold["in.x"], x)
    write_st(f"{out}/{name}.model.safetensors", t, {"varda_format": "1", "config": json.dumps(cfg)})
    manifest = {
        "tag": f"synthetic/{name}", "contract": "docs/format.md section 2",
        "script": "python/tests/synthetic.py", "generator": "python/tests/synthetic.py",
        "reference": "independent float64 numpy implementation of spec-model.md / spec-interface.md, "
                     "normalisation in two fp32 roundings, rounded to F32 at export",
        "seed": c["seed"], "config": c, "numpy": np.__version__,
        "checkpoint_identity": {"uuid": uuid_of(name)}, "keys": keys,
        "shapes": {k: list(gold[k].shape) for k in keys},
    }
    meta = {"varda_golden": "1", "tag": f"synthetic/{name}", "manifest": json.dumps(manifest, indent=1),
            "source": "python/tests/synthetic.py", "source_size": str(sum(a.nbytes for a in gold.values())),
            "checkpoint_uuid": uuid_of(name)}
    write_st(f"{out}/{name}.golden.safetensors", gold, meta)
    absmax = max(float(np.abs(gold[k]).max()) for k in keys)
    print(f"wrote {name}: C={c['C']} heads={c['heads']} n_proc={n_proc} T={c['T']} G={c['G']} H={c['H']} "
          f"absent={len(cfg['absent'])} keys={len(keys)} max|act|={absmax:.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="output directory for <name>.model/.golden.safetensors")
    ap.add_argument("--only", default=",".join(CONFIGS), help="comma-separated config names")
    ap.add_argument("--write-tolerances", metavar="DIR", help="write DIR/synthetic-<name>.json")
    ap.add_argument("--check-tolerances", metavar="DIR", help="fail unless DIR/synthetic-<name>.json is up to date")
    ap.add_argument("--hash", action="store_true", help="print the sha256 of this script and exit")
    a = ap.parse_args()
    if a.hash:
        print(hashlib.sha256(open(__file__, "rb").read()).hexdigest())
        return 0
    names = a.only.split(",")
    for n in names:
        if n not in CONFIGS:
            ap.error(f"unknown config {n}")
    status = 0
    if a.write_tolerances:
        for n in names:
            with open(f"{a.write_tolerances}/synthetic-{n}.json", "w") as f:
                json.dump(tolerance_file(n), f, indent=1)
                f.write("\n")
    if a.check_tolerances:
        for n in names:
            p = f"{a.check_tolerances}/synthetic-{n}.json"
            try:
                ok = json.load(open(p)) == tolerance_file(n)
            except (OSError, ValueError):
                ok = False
            if not ok:
                print(f"tolerance file {p} is missing or differs from the generator (rerun --write-tolerances)")
                status = 1
    if a.out:
        for n in names:
            generate(n, a.out)
    return status


if __name__ == "__main__":
    sys.exit(main())
