"""Export a real-data reference run's dumps into one per-stage, per-variable file for the C++ checks.

Reads <run>/dumps/*.safetensors (run_reference_realdata.py) and writes <run>/stages.safetensors (atomically, via
python/safetensors_io.py; F32 as the reference holds it, coordinates F64, masks U8). Names (docs/realdata-reference.md):

    lam.<var>            [2, 1147980]  KENDA after its pre_processors (prognostics and constants), dates (t-6h, t)
    lam.latitudes/longitudes  [1147980] F64
    global.<var>         [2, 542080]   IFS after MIR regrid and gh -> z (before the cutout mask), prognostics+constants
    global.latitudes/longitudes  [542080] F64
    mask.lam_0, mask.global  [G] U8   the cutout's _mask (where each source's points sit in G)
    latitudes, longitudes [G] F64      merged coordinates
    x0                   [1, 2, G, 79] the tensor passed to predict_step at step 0 (= golden in.x layout)
    window.<s>           [1, 2, G, 79] the tensor passed at step s (s >= 1)
    forcings.<s>         [F, G]        computed dynamic forcings inserted for step s (s >= 1; date = valid date of
                                       output s-1), rows in metadata forcing_vars order
    y.<s>                [G, 67]       model output of step s (as the runner state holds it, before post-processors)
    y.<s>.post           [G, 67]       after accumulate_from_start_of_forecast + rescale (tp)
    init.<var>           [G]           initial (step-0) state after the top-level post-processors (written to the
                                       step-0 GRIB files), all 70 fields
Metadata `stages` (JSON): in_vars, out_vars, dates, forcing_vars, source of each tensor, and the checks run here
(x0 == input_tensor transposed, y.<s> == model output cast, NaN counts).

    $PY -I python/reference/realdata/export_stages.py RUN
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetensors_io import read_safetensors, write_safetensors  # noqa: E402

CONVERTED = (Path(os.environ["VARDA_SCRATCH"]) / "converted" / "varda" / "model.safetensors"  # cutout masks
             if os.environ.get("VARDA_SCRATCH") else None)


def load(run: Path, name: str):
    meta, t = read_safetensors(run / "dumps" / f"{name}.safetensors")
    return json.loads(meta["meta"]), t


def fields(t: dict) -> dict:
    return {k.removeprefix("field/"): v for k, v in t.items() if k.startswith("field/")}


def T(a) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def main() -> None:
    run = Path(sys.argv[1])
    out: dict[str, torch.Tensor] = {}
    info: dict = {"run": str(run.resolve()), "checks": {}}

    for src, pre in (("lam_0", "lam"), ("global", "global")):
        for purpose in ("prognostics", "constant_forcings"):
            m, t = load(run, f"source.{purpose}.{src}")
            for k, v in fields(t).items():
                out[f"{pre}.{k}"] = T(np.asarray(v))
            info.setdefault("dates", m.get("dates"))
            for c in ("latitudes", "longitudes"):
                if f"{pre}.{c}" in out:
                    if not np.array_equal(out[f"{pre}.{c}"].numpy(), np.asarray(t[c])):
                        raise SystemExit(f"{src}: {c} differ between purposes")
                else:
                    out[f"{pre}.{c}"] = T(np.asarray(t[c]))

    m, t = load(run, "cutout.prognostics")
    for k in ("mask/lam_0", "mask/global"):
        out[k.replace("/", ".")] = T(np.asarray(t[k]).astype(np.uint8))
    out["latitudes"], out["longitudes"] = T(np.asarray(t["latitudes"])), T(np.asarray(t["longitudes"]))
    # The KENDA state carries no usable coordinates (unstructured GRIB: earthkit falls back to the checkpoint's
    # full 1688650-point latitudes); the cutout then loads source0/* (= the lam slice of the merged coordinates,
    # lam_0/cutout_mask being all True). Export the coordinates the cutout used.
    n_lam = out["lam.2t"].shape[-1]
    info["lam_coords_in_source_state"] = list(out["lam.latitudes"].shape)
    lam = np.asarray(t["mask/lam_0"]).astype(bool)
    if int(lam.sum()) != n_lam or not lam[:n_lam].all():
        raise SystemExit("lam_0 is not the leading slice of the merged state")
    out["lam.latitudes"], out["lam.longitudes"] = out["latitudes"][:n_lam].clone(), out["longitudes"][:n_lam].clone()

    mi, ti = load(run, "input_tensor")
    in_vars = mi["in_vars"]
    info["in_vars"] = in_vars
    steps = sorted(int(p.name.split(".")[0][4:]) for p in (run / "dumps").glob("step*.window.safetensors"))
    for s in steps:
        mw, tw = load(run, f"step{s}.window")
        out["x0" if s == 0 else f"window.{s}"] = T(np.asarray(tw["x"]))
    x0 = out["x0"].numpy()
    info["checks"]["x0_equals_input_tensor"] = bool(np.array_equal(
        x0[0], np.swapaxes(np.asarray(ti["x"]), -2, -1), equal_nan=True))
    info["checks"]["x0_nan"] = int(np.isnan(x0).sum())
    # C1 -> C2: the merged columns of x0 are lam.<var> followed by global.<var>[:, global/cutout_mask]
    if CONVERTED is not None and CONVERTED.exists():
        cmeta, conv = read_safetensors(CONVERTED)
        cvars = json.loads(cmeta["config"])["variables"]
        info["checks"]["in_vars_equal_engine_input_order"] = cvars["input"] == in_vars
        info["engine_output_vars"] = cvars["output"]
        keep = np.asarray(conv["grid.global.cutout_mask"]).astype(bool)
        lam_keep = np.asarray(conv["grid.lam_0.cutout_mask"]).astype(bool)
        n_ok, n_vars = 0, 0
        for i, v in enumerate(in_vars):
            if f"lam.{v}" in out:
                n_vars += 1
                sel = np.concatenate([out[f"lam.{v}"].numpy()[:, lam_keep], out[f"global.{v}"].numpy()[:, keep]], -1)
                n_ok += int(np.array_equal(sel, x0[0, :, :, i], equal_nan=True))
        info["checks"]["x0_columns_equal_merged_sources"] = {"vars": n_vars, "bitwise": n_ok,
                                                             "masks_from": str(CONVERTED)}

    fvars = None
    for p in sorted((run / "dumps").glob("forcings.dynamic.*.safetensors"), key=lambda p: int(p.name.split(".")[2])):
        i = int(p.name.split(".")[2])
        mf, tf = load(run, f"forcings.dynamic.{i}")
        if i == 0:  # the initial window's forcings (dates t-6h, t): part of x0
            info["forcings_initial"] = {"variables": mf["variables"], "dates": mf["dates"]}
            continue
        fvars = mf["variables"]
        v = np.asarray(tf["values"])  # [vars, 1, G]
        out[f"forcings.{i}"] = T(v[:, 0, :])
        info.setdefault("forcing_dates", {})[str(i)] = mf["dates"]
    info["forcing_vars"] = fvars

    out_vars = None
    for s in steps:
        my, ty = load(run, f"step{s}.y")
        out_vars = my["out_vars"]
        y = np.asarray(ty["y"]).reshape(-1, len(out_vars))
        mp, tp = load(run, f"state{s + 1}.pre")
        f = fields(tp)
        pre = np.stack([np.asarray(f[v]) for v in out_vars], axis=1)
        info["checks"][f"y.{s}_state_equals_model_output"] = bool(np.array_equal(pre, y, equal_nan=True))
        info["checks"][f"y.{s}_dtype_model"] = my["dtype"]
        info["checks"][f"y.{s}_dtype_state"] = str(np.asarray(f[out_vars[0]]).dtype)
        out[f"y.{s}"] = T(pre)
        post = dict(f)
        for q in sorted((run / "dumps").glob(f"state{s + 1}.*.safetensors")):
            if q.name.endswith(".pre.safetensors"):
                continue
            _, tq = load(run, q.name.removesuffix(".safetensors"))
            post.update(fields(tq))
        out[f"y.{s}.post"] = T(np.stack([np.asarray(post[v]) for v in out_vars], axis=1))
        info.setdefault("dates_out", {})[str(s)] = mp.get("date")
    info["out_vars"] = out_vars
    if "engine_output_vars" in info:
        info["checks"]["out_vars_equal_engine_output_order"] = info.pop("engine_output_vars") == out_vars

    mp, tp = load(run, "state0.pre")
    init = fields(tp)
    for q in sorted((run / "dumps").glob("state0.*.safetensors")):
        if not q.name.endswith(".pre.safetensors"):
            init.update(fields(load(run, q.name.removesuffix(".safetensors"))[1]))
    for k, v in init.items():
        out[f"init.{k}"] = T(np.asarray(v))
    info["init_vars"] = list(init)

    size = write_safetensors(run / "stages.safetensors", out, {"varda_realdata_stages": "1",
                                                                "stages": json.dumps(info, default=str)})
    print(json.dumps(info["checks"], indent=1))
    print(f"wrote {run / 'stages.safetensors'}: {len(out)} tensors, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
