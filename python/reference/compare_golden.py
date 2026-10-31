"""Compare golden dumps: two contract `activations.pt` files key by key, or two legacy run_reference.py dirs.
The metric and checkpoint_identity are what run_rollout.py imports.

Usage:
    python python/reference/compare_golden.py REF OTHER [--json out.json] [--fail-above X]
    python python/reference/compare_golden.py --contract MODEL.safetensors GOLDEN [GOLDEN ...] [--json out.json]

REF / OTHER are either an `activations.pt` file (docs/format.md §2), a golden directory containing one
(e.g. $REF_SCRATCH/golden/cpu-fp32-pyg), or a legacy run_reference.py directory
(with input.pt, output.pt, activations/*.pt; old per-file mode, kept for the cross-device comparisons).

Metric (contract mode), per key, computed in float64 on CPU; trivial to re-implement in the engine tests:

    d          = |other - ref|                       (elementwise)
    max_abs    = max(d)
    ref_absmax = max(|ref|)
    rel_absmax = max_abs / ref_absmax                (0 if both are 0, inf if only ref_absmax is 0)

rel_absmax is the error relative to the tensor's own scale; elementwise relative error is meaningless near
zero, so it is not the pass/fail metric. Any NaN in either tensor makes max_abs NaN (reported, and fails
--fail-above). Keys are visited in contract order (contract_keys(n_proc), n_proc taken from the proc.{i}.out keys
present); keys missing in one dict, shape or dtype mismatches are reported. --fail-above X exits with status 1 if
any common key has rel_absmax > X (or NaN), or a shape mismatch; use --fail-above 0 for a bitwise check. Only
needs torch (runs in either venv).

--contract checks golden sets against the contract of a converted model (docs/format.md §1.2 config, read from the
safetensors header only): exactly the contract keys for its processor depth, the §2 shapes from its dims (T_out =
n_output_steps: dec.out (G, T_out*V_out), out.* (1, T_out, 1, G, V_out)), F32, every value finite for an
activations.pt (out.y only for an exported .safetensors, which also must carry the model's checkpoint_uuid). Exit 1 on
any mismatch.

contract_keys / required_keys / legacy_map / contract_order and checkpoint_identity are self-contained (this
directory runs in the reference venvs and imports nothing from python/); checkpoint_identity returns the same
fields as python/checkpoint.py's. Change both together.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import torch

# ------------------------------------------------- contract keys (docs/format.md §2) and checkpoint identity

_HEAD = ["in.x", "in.x_norm", "enc.x_src", "enc.x_dst", "enc.emb_src", "enc.emb_dst", "enc.out"]
_TAIL = ["latent", "dec.emb_dst", "dec.block_out", "dec.out", "out.pre_bound", "out.norm", "out.y"]
_REQUIRED_HEAD = ["in.x", "in.x_norm", "enc.x_src", "enc.out"]
_REQUIRED_TAIL = ["latent", "dec.out", "out.pre_bound", "out.norm", "out.y"]
_PROC = re.compile(r"^proc\.(\d+)\.out$")


def contract_keys(n_proc: int) -> list[str]:
    """Every docs/format.md §2 key, in table order, for a model with ``n_proc`` processor blocks."""
    return _HEAD + [f"proc.{i}.out" for i in range(n_proc)] + _TAIL


def required_keys(n_proc: int) -> list[str]:
    """The required keys of docs/format.md §2 (the others are best effort)."""
    return _REQUIRED_HEAD + [f"proc.{i}.out" for i in range(n_proc)] + _REQUIRED_TAIL


def legacy_map(n_proc: int) -> dict[str, str]:
    """reference/run_reference.py per-file dump -> contract key ("activations/processor.pt", the processor
    output, is the last block's output and is left out)."""
    return {
        "input.pt": "in.x",
        "activations/pre_processed_input.pt": "in.x_norm",
        "activations/encoder.x_data_latent.pt": "enc.x_src",
        "activations/encoder.emb_nodes_src.pt": "enc.emb_src",
        "activations/encoder.emb_nodes_dst.pt": "enc.emb_dst",
        "activations/encoder.x_latent.pt": "enc.out",
        **{f"activations/processor.proc.{i}.pt": f"proc.{i}.out" for i in range(n_proc)},
        "activations/decoder.emb_nodes_dst.pt": "dec.emb_dst",
        "activations/decoder.pt": "dec.out",
        "activations/model_output_normalised.pt": "out.norm",
        "output.pt": "out.y",
    }


def contract_shapes(dims: dict, batch: int = 1) -> dict[str, tuple]:
    """docs/format.md §2 shapes from a converted model's config `dims` (T_out = n_output_steps)."""
    G, H, C = dims["n_data_nodes"], dims["n_hidden_nodes"], dims["num_channels"]
    T, T_out, A = dims["n_input_steps"], dims["n_output_steps"], dims["node_attr_dim"]
    vi, vo = dims["n_input_vars"], dims["n_output_vars"]
    bg, bh = batch * G, batch * H
    s = {"in.x": (batch, T, G, vi), "in.x_norm": (batch, T, 1, G, vi), "enc.x_src": (bg, T * vi + A),
         "enc.x_dst": (bh, A), "enc.emb_src": (bg, C), "enc.emb_dst": (bh, C), "enc.out": (bh, C), "latent": (bh, C),
         "dec.emb_dst": (bg, C), "dec.block_out": (bg, C), "dec.out": (bg, T_out * vo),
         "out.pre_bound": (batch, T_out, 1, G, vo), "out.norm": (batch, T_out, 1, G, vo),
         "out.y": (batch, T_out, 1, G, vo)}
    s.update({f"proc.{i}.out": (bh, C) for i in range(dims["num_processor_layers"])})
    return s


def safetensors_header(path) -> tuple[dict, int]:
    """(header dict, data start offset) of a safetensors file."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def check_contract(model_path, golden_path) -> dict:
    """Keys, shapes, dtype and finiteness of one golden set against a converted model's contract (see docstring)."""
    import numpy as np

    mh, _ = safetensors_header(model_path)
    cfg = json.loads(mh["__metadata__"]["config"])
    want = contract_shapes(cfg["dims"])
    keys_want = contract_keys(cfg["dims"]["num_processor_layers"])
    golden_path = Path(golden_path)
    res = {"model": str(model_path), "golden": str(golden_path), "errors": [], "shapes": {}}
    if golden_path.suffix == ".safetensors":
        h, start = safetensors_header(golden_path)
        meta = h.pop("__metadata__", {})
        if meta.get("checkpoint_uuid") != cfg["source"]["uuid"]:
            res["errors"].append(f"checkpoint_uuid {meta.get('checkpoint_uuid')} != model {cfg['source']['uuid']}")
        shapes = {k: (tuple(v["shape"]), v["dtype"]) for k, v in h.items()}
        finite_keys = ["out.y"] if "out.y" in h else []
        def values(k):
            b, e = h[k]["data_offsets"]
            return np.memmap(golden_path, dtype="<f4", mode="r", offset=start + b, shape=((e - b) // 4,))
    else:
        acts = load_activations(golden_path)
        shapes = {k: (tuple(t.shape), {torch.float32: "F32"}.get(t.dtype, str(t.dtype))) for k, t in acts.items()}
        finite_keys = list(acts)
        values = lambda k: acts[k]  # noqa: E731
    if list(shapes) != keys_want and sorted(shapes) != sorted(keys_want):
        res["errors"].append(f"keys: missing {sorted(set(keys_want) - set(shapes))}, "
                             f"extra {sorted(set(shapes) - set(keys_want))}")
    for k in keys_want:
        if k not in shapes:
            continue
        shape, dt = shapes[k]
        res["shapes"][k] = list(shape)
        if shape != want[k]:
            res["errors"].append(f"{k}: shape {shape} != contract {want[k]}")
        if dt != "F32":
            res["errors"].append(f"{k}: dtype {dt} != F32")
    res["finite"] = {}
    for k in finite_keys:
        v = values(k)
        ok = bool(torch.isfinite(v).all()) if isinstance(v, torch.Tensor) else bool(np.isfinite(v).all())
        res["finite"][k] = ok
        if not ok:
            res["errors"].append(f"{k}: non-finite values")
    res["ok"] = not res["errors"]
    return res


def n_proc_of(keys) -> int:
    """Processor depth implied by the ``proc.{i}.out`` keys present (0 if none)."""
    idx = [int(m.group(1)) for k in keys if (m := _PROC.match(k))]
    return max(idx) + 1 if idx else 0


def contract_order(keys) -> list[str]:
    """Contract keys first in table order, then any extra keys sorted."""
    keys = set(keys)
    order = contract_keys(n_proc_of(keys))
    return [k for k in order if k in keys] + sorted(keys - set(order))


def checkpoint_identity(path, sha256=True):
    """{path, uuid, run_id, size_bytes, sha256} of a checkpoint (as python/checkpoint.py's checkpoint_identity)."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json")), None)
        if member is None:
            raise ValueError(f"{path}: no anemoi-metadata/anemoi.json member (not an anemoi checkpoint?)")
        meta = json.loads(z.read(member))
    out = {"path": str(path.resolve()), "uuid": meta.get("uuid"), "run_id": meta.get("run_id"),
           "size_bytes": path.stat().st_size}
    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


# ------------------------------------------------------------------------------------------------- metric


def diff_metric(ref: torch.Tensor, other: torch.Tensor) -> dict:
    """The contract metric (see module docstring)."""
    ref, other = ref.detach().to("cpu", torch.float64), other.detach().to("cpu", torch.float64)
    max_abs = (other - ref).abs().max().item()
    ref_absmax = ref.abs().max().item()
    if math.isnan(max_abs) or math.isnan(ref_absmax):
        rel = float("nan")
    elif ref_absmax == 0.0:
        rel = 0.0 if max_abs == 0.0 else float("inf")
    else:
        rel = max_abs / ref_absmax
    return {"max_abs": max_abs, "ref_absmax": ref_absmax, "rel_absmax": rel,
            "bitwise_equal": bool(torch.equal(ref, other))}


def load_activations(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_dir():
        path = path / "activations.pt"
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)  # Varda sets are 40 GB


def compare_activations(ref: dict, other: dict) -> dict:
    res = {"keys": {}, "only_in_ref": [], "only_in_other": []}
    for k in contract_order(set(ref) | set(other)):
        if k not in other:
            res["only_in_ref"].append(k)
            continue
        if k not in ref:
            res["only_in_other"].append(k)
            continue
        r, o = ref[k], other[k]
        entry = {"shape": list(r.shape), "dtype": str(r.dtype)}
        if r.shape != o.shape:
            entry["error"] = f"shape mismatch {tuple(r.shape)} vs {tuple(o.shape)}"
        else:
            if r.dtype != o.dtype:
                entry["dtype_other"] = str(o.dtype)
            entry.update(diff_metric(r, o))
        res["keys"][k] = entry
    return res


def print_activation_report(res: dict) -> None:
    fmt = "{:<16s} {:<22s} {:>11s} {:>11s} {:>11s}  {}"
    print(fmt.format("key", "shape", "max_abs", "max|ref|", "rel_absmax", "bitwise"))
    for k, e in res["keys"].items():
        if "error" in e:
            print(f"{k:<16s} ERROR {e['error']}")
            continue
        print(fmt.format(k, "x".join(map(str, e["shape"])), f"{e['max_abs']:.3e}", f"{e['ref_absmax']:.3e}",
                         f"{e['rel_absmax']:.3e}", "yes" if e["bitwise_equal"] else "no"))
    for side in ("only_in_ref", "only_in_other"):
        if res[side]:
            print(f"{side}: {', '.join(res[side])}")


# ---------------------------------------------------------------- legacy mode (run_reference.py dirs)

def stats(ref: torch.Tensor, other: torch.Tensor) -> dict:
    ref, other = ref.double(), other.double()
    d = (ref - other).abs()
    amax = ref.abs().max().item()
    mask = ref.abs() > 1e-3 * amax
    return {
        "max_abs": d.max().item(),
        "ref_absmax": amax,
        "rel_absmax": d.max().item() / amax if amax else float("nan"),
        "max_rel": (d[mask] / ref.abs()[mask]).max().item() if mask.any() else float("nan"),
        "rms_diff": d.pow(2).mean().sqrt().item(),
    }


def group_of(name: str) -> str:
    if name in ("tp",):
        return "tp"
    if name in ("hcc", "mcc", "lcc", "tcc"):
        return "clouds"
    if "_" in name and name.split("_")[-1].isdigit():
        return name.split("_")[0] + "_pl"
    return "surface"


def compare_legacy_dirs(ref: Path, oth: Path) -> dict:
    """Old mode: max_rel = max |a-b|/|ref| over elements with |ref| > 1e-3 * max|ref|; per-variable output."""
    man = json.loads((ref / "manifest.json").read_text())
    res = {"ref": str(ref), "other": str(oth), "tensors": {}, "output_per_variable": {}, "output_per_group": {}}
    for k in sorted(man["tensors"]):
        if not (oth / k).exists():
            continue
        r = torch.load(ref / k, weights_only=True)
        o = torch.load(oth / k, weights_only=True)
        res["tensors"][k] = stats(r, o)

    r = torch.load(ref / "output.pt", weights_only=True)
    o = torch.load(oth / "output.pt", weights_only=True)
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(man["output_variables"]):
        res["output_per_variable"][n] = stats(r[..., j], o[..., j])
        groups.setdefault(group_of(n), []).append(j)
    for g, idx in groups.items():
        res["output_per_group"][g] = stats(r[..., idx], o[..., idx])

    fmt = "{:<40s} {:>11s} {:>11s} {:>11s} {:>11s}"
    print(fmt.format("tensor", "max_abs", "ref_absmax", "rel_absmax", "max_rel"))
    for title, d in (("", res["tensors"]), ("output group ", res["output_per_group"])):
        for k, s in d.items():
            print(fmt.format(title + k, *(f"{s[x]:.3e}" for x in ("max_abs", "ref_absmax", "rel_absmax", "max_rel"))))
    return res


def is_contract(p: Path) -> bool:
    return p.is_file() or (p / "activations.pt").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="REF OTHER, or with --contract: one or more golden sets")
    ap.add_argument("--contract", metavar="MODEL", help="check golden sets against this converted model's contract")
    ap.add_argument("--json")
    ap.add_argument("--fail-above", type=float, default=None,
                    help="contract mode: exit 1 if any key has rel_absmax above this (0 = bitwise)")
    a = ap.parse_args()
    if a.contract:
        results = [check_contract(a.contract, p) for p in a.paths]
        for r in results:
            print(f"{r['golden']}: {'OK' if r['ok'] else 'FAIL'} ({len(r['shapes'])} keys; finite: "
                  f"{sum(r['finite'].values())}/{len(r['finite'])} checked)"
                  + "".join(f"\n  {e}" for e in r["errors"]))
            for k in ("dec.out", "out.y"):
                if k in r["shapes"]:
                    print(f"  {k} {r['shapes'][k]}")
        if a.json:
            Path(a.json).write_text(json.dumps(results, indent=1) + "\n")
        sys.exit(0 if all(r["ok"] for r in results) else 1)
    if len(a.paths) != 2:
        ap.error("expected REF OTHER")
    ref, oth = Path(a.paths[0]), Path(a.paths[1])

    if is_contract(ref) and is_contract(oth):
        res = compare_activations(load_activations(ref), load_activations(oth))
        res = {"ref": str(ref), "other": str(oth), **res}
        print_activation_report(res)
        status = 0
        if a.fail_above is not None:
            bad = [k for k, e in res["keys"].items()
                   if "error" in e or math.isnan(e["rel_absmax"]) or e["rel_absmax"] > a.fail_above]
            if bad:
                print(f"FAIL: rel_absmax > {a.fail_above:g} (or error) for: {', '.join(bad)}")
                status = 1
            else:
                print(f"OK: all {len(res['keys'])} common keys have rel_absmax <= {a.fail_above:g}")
    else:
        res = compare_legacy_dirs(ref, oth)
        status = 0
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    sys.exit(status)


if __name__ == "__main__":
    main()
