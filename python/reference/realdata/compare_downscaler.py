"""Spreads between the temporal-downscaler reference runs (run_reference_downscaler.py), and the packing loss of
its GRIB-decoded inputs. Runs in the reference env ($VARDA_SCRATCH/ref-oper-td/.venv: torch, eccodes).

  $PY -I python/reference/realdata/compare_downscaler.py REF_DIR [--runs fp16,fp32,cpu-fp32] [--forecaster-root R]

REF_DIR = $VARDA_SCRATCH/downscale/ref/<YYYYmmddTHH> (holds one directory per tag; the date is taken from its name,
or --date). Metric (docs/format.md section 2),
float64, per key: max_abs = max|a - r|, absmax = max|r|, rel_absmax = max_abs / absmax. NaN positions must agree.

Writes REF_DIR/spreads.json (atomically) with
  pairs.<a>_vs_<b>.model_input.<k>.<slot>.<var>        x of window k (77 channels), a vs b
  pairs.<a>_vs_<b>.model_output.<k>.<h>.<var>          raw y, hour h = 1..6 of window k, 67 variables
  pairs.<a>_vs_<b>.grib.<step>.<name>                  decoded hourly GRIB messages, step 0..24 h
  pairs.<a>_vs_<b>.summary                             worst variable per (window, hour) / per GRIB step
  packing.<tag>.<k>.<slot>.<var>                       decoded GRIB input of the downscaler vs the forecaster's raw
                                                       state (dumps of the forecaster reference run), 66 prognostics
                                                       + z, lsm of slot 0
  copy_check.<tag>                                     hourly GRIB steps 0/6/12/18/24 vs the forecaster's own files
  tp_check.<tag>                                       accumulate / rescale / written tp, recomputed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from eccodes_cosmo_resources import get_definitions_path  # noqa: E402

os.environ["ECCODES_DEFINITION_PATH"] = str(get_definitions_path())
os.environ["ECCODES_VERSION_CHECK_OFF"] = "1"

import eccodes as ec  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(HERE.parents[1]))
from safetensors_io import read_safetensors  # noqa: E402

CHUNK = 1 << 18


def metric(a, r) -> dict:
    """a, r: arrays of any shape, compared over all points (float64, chunked)."""
    a = np.asarray(a).reshape(-1)
    r = np.asarray(r).reshape(-1)
    max_abs, absmax, nanmis, nan = 0.0, 0.0, 0, 0
    for i in range(0, a.size, CHUNK):
        x, y = a[i:i + CHUNK].astype(np.float64), r[i:i + CHUNK].astype(np.float64)
        nx, ny = np.isnan(x), np.isnan(y)
        nanmis += int((nx != ny).sum())
        nan += int(nx.sum())
        ok = ~(nx | ny)
        if ok.any():
            max_abs = max(max_abs, float(np.abs(x[ok] - y[ok]).max()))
            absmax = max(absmax, float(np.abs(y[ok]).max()))
    rel = max_abs / absmax if absmax else (0.0 if max_abs == 0 else float("inf"))
    return {"rel_absmax": rel, "max_abs": max_abs, "absmax": absmax, "nan_mismatch": nanmis}


def metric_cols(a, r) -> np.ndarray:
    """a, r: [N, V]; per-column (max_abs, absmax, nan_mismatch) in float64, chunked over N."""
    V = a.shape[1]
    ma, am, nm = np.zeros(V), np.zeros(V), np.zeros(V, np.int64)
    for i in range(0, a.shape[0], CHUNK):
        x, y = np.asarray(a[i:i + CHUNK], np.float64), np.asarray(r[i:i + CHUNK], np.float64)
        nx, ny = np.isnan(x), np.isnan(y)
        nm += (nx != ny).sum(0)
        d = np.where(nx | ny, 0.0, np.abs(x - y))
        ma = np.maximum(ma, d.max(0))
        am = np.maximum(am, np.where(ny, 0.0, np.abs(y)).max(0))
    return ma, am, nm


def cols_to_dict(names, ma, am, nm) -> dict:
    out = {}
    for j, n in enumerate(names):
        rel = ma[j] / am[j] if am[j] else (0.0 if ma[j] == 0 else float("inf"))
        out[n] = {"rel_absmax": float(rel), "max_abs": float(ma[j]), "absmax": float(am[j]), "nan_mismatch": int(nm[j])}
    return out


def worst(d: dict) -> dict:
    n = max(d, key=lambda k: d[k]["rel_absmax"])
    return {"var": n, "rel_absmax": d[n]["rel_absmax"], "max_abs": d[n]["max_abs"],
            "median_rel_absmax": float(np.median([v["rel_absmax"] for v in d.values()]))}


def grib_messages(path: Path) -> dict:
    """{name: float64 values}; name = '<shortName>@<typeOfLevel>:<level>'."""
    out = {}
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            name = f"{ec.codes_get(h, 'shortName')}@{ec.codes_get(h, 'typeOfLevel')}:{ec.codes_get(h, 'level')}"
            out[name] = ec.codes_get_double_array(h, "values")
            ec.codes_release(h)
    return out


def load_io(run: Path):
    meta, t = read_safetensors(run / "model_io.safetensors")
    return meta, t


def pair_model(a_run: Path, b_run: Path) -> dict:
    ma, ta = load_io(a_run)
    mb, tb = load_io(b_run)
    in_vars, out_vars = json.loads(ma["in_vars"]), json.loads(ma["out_vars"])
    res = {"model_input": {}, "model_output": {}}
    summ_in, summ_out = {}, {}
    for key in sorted(k for k in ta if k.startswith("x.")):
        k = key.split(".")[1]
        if key not in tb:
            continue
        for slot in (0, 1):
            d = cols_to_dict(in_vars, *metric_cols(ta[key][slot], tb[key][slot]))
            res["model_input"][f"{k}.{slot}"] = d
            summ_in[f"w{k}.slot{slot}"] = worst(d)
        yk = f"y.{k}"
        if yk in ta and yk in tb:
            for h in range(6):
                d = cols_to_dict(out_vars, *metric_cols(ta[yk][h], tb[yk][h]))
                res["model_output"][f"{k}.{h + 1}"] = d
                summ_out[f"w{k}.h{h + 1}"] = worst(d)
    res["summary"] = {"model_input": summ_in, "model_output": summ_out}
    return res


def pair_grib(a_run: Path, b_run: Path, steps) -> dict:
    res, summ = {}, {}
    for s in steps:
        pa, pb = a_run / "outputs" / f"varda-single_{DATE_STAMP}_{s}.grib", b_run / "outputs" / f"varda-single_{DATE_STAMP}_{s}.grib"
        if not (pa.exists() and pb.exists()):
            continue
        A, B = grib_messages(pa), grib_messages(pb)
        d = {n: metric(A[n], B[n]) for n in sorted(B) if n in A}
        d["_only_a"], d["_only_b"] = sorted(set(A) - set(B)), sorted(set(B) - set(A))
        res[str(s)] = d
        summ[str(s)] = worst({k: v for k, v in d.items() if not k.startswith("_")})
    return {"grib": res, "grib_summary": summ}


def packing(tag_run: Path, fc_run: Path) -> dict:
    """Decoded GRIB inputs (x) against the forecaster's raw states."""
    meta, t = load_io(tag_run)
    in_vars = json.loads(meta["in_vars"])
    out_vars = json.loads(meta["out_vars"])
    d0 = read_safetensors(fc_run / "dumps" / "state0.pre.safetensors")[1]
    res = {}
    n_windows = len([k for k in t if k.startswith("x.")])
    for k in range(n_windows):
        x = t[f"x.{k}"]
        for slot in (0, 1):
            s = 6 * k + 6 * slot  # forecast hour of the slot
            if s == 0:
                raw = {n: d0[f"field/{n}"] for n in out_vars if n != "tp"}
                raw.update({"z": d0["field/z"], "lsm": d0["field/lsm"]})
            else:
                yy = read_safetensors(fc_run / "dumps" / f"step{s // 6 - 1}.y.safetensors")[1]["y"][0, 0, 0]  # [G, 67]
                raw = {n: yy[:, out_vars.index(n)] for n in out_vars if n != "tp"}
                if slot == 0:  # constants of the slot are the t=0 ones (see the doc); only compare the prognostics
                    pass
            r = {}
            for n, v in raw.items():
                if slot == 1 and n in ("z", "lsm"):
                    continue
                if n in ("z", "lsm") and slot == 0 and k > 0:
                    continue
                r[n] = metric(x[slot][:, in_vars.index(n)], v)
            res[f"{k}.{slot}"] = r
    return res


def tp_check(run: Path, fc_run: Path) -> dict:
    meta, t = read_safetensors(run / "tp_stages.safetensors")
    calls = json.loads(meta["calls"])
    written = json.loads(meta["written"])
    mask = read_safetensors(run / "dumps" / "cutout.prognostics.w0.safetensors")[1]["mask/lam_0"].astype(bool)
    out = {"calls": [], "written": []}
    prev_acc = None
    for c in calls:
        n = c["n"]
        if not c["has_tp"]:
            continue
        pre, acc, post = t[f"call{n}.pre"], t[f"call{n}.acc"], t[f"call{n}.post"]
        want = np.maximum(pre, 0) + (prev_acc if prev_acc is not None else 0)
        want = want.astype(np.float32)
        e = {"call": n, "step": c["step"], "date": c["date"],
             "acc_equals_cumsum_relu": bool(np.array_equal(acc, want)),
             "post_equals_acc_x1000_f32": bool(np.array_equal(post, (acc * np.float32(1000)).astype(np.float32))),
             "post_equals_acc_x1000_f64": bool(np.array_equal(post, (acc.astype(np.float64) * 1000).astype(np.float32))),
             "pre_min": float(pre.min()), "pre_max": float(pre.max()), "acc_max": float(acc.max()),
             "post_max_kgm2": float(post.max()), "pre_negative_points": int((pre < 0).sum())}
        out["calls"].append(e)
        prev_acc = acc
    for w in written:
        n = w["n"]
        if f"written.{n}" not in t:
            continue
        e = {"n": n, "step": w["step"], "tp_dtype": str(t[f"written.{n}"].dtype)}
        # the call whose post equals the written tp inside the LAM mask
        for c in out["calls"]:
            post = t[f"call{c['call']}.post"]
            if np.array_equal(t[f"written.{n}"], post[mask]):
                e["equals_post_of_call"] = c["call"]
                break
        else:
            e["equals_post_of_call"] = None
        out["written"].append(e)
    return out


def copy_check(run: Path, fc_run: Path, steps=(0, 6, 12, 18, 24)) -> dict:
    res = {}
    for s in steps:
        pa = run / "outputs" / f"varda-single_{DATE_STAMP}_{s}.grib"
        pb = fc_run / "outputs" / f"{DATE_STAMP}_{s}.grib"
        if not (pa.exists() and pb.exists()):
            continue
        A, B = grib_messages(pa), grib_messages(pb)
        same = [n for n in A if n in B and np.array_equal(A[n], B[n], equal_nan=True)]
        diff = {n: metric(A[n], B[n]) for n in A if n in B and n not in same}
        res[str(s)] = {"n_downscaler": len(A), "n_forecaster": len(B), "identical": len(same),
                       "different": {n: v for n, v in diff.items()}, "only_downscaler": sorted(set(A) - set(B)),
                       "only_forecaster": sorted(set(B) - set(A))}
    return res


DATE_STAMP = "202610221500"  # YYYYmmddHHMM of the run, set in main() from the reference directory name or --date


def main() -> None:
    global DATE_STAMP
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref_dir")
    ap.add_argument("--runs", default="fp16,fp32,cpu-fp32,fp16-fp32in")
    ap.add_argument("--date", default=None, help="YYYY-mm-ddTHH or YYYYmmddTHH (default: the name of REF_DIR)")
    ap.add_argument("--forecaster-root", default=None, help="default $VARDA_SCRATCH/realdata/<YYYYmmddTHH>")
    ap.add_argument("--steps", default=",".join(str(i) for i in range(25)))
    ap.add_argument("--skip-grib", action="store_true")
    ap.add_argument("--pairs", default=None, help="a:b,c:d (default: all known pairs of the available runs; '' = none)")
    ap.add_argument("--no-extras", action="store_true", help="skip packing / copy / tp checks")
    ap.add_argument("--out-name", default="spreads.json", help="file name under REF_DIR (parts are merged by --merge)")
    ap.add_argument("--merge", default=None, help="comma-separated part files to merge into spreads.json, then exit")
    args = ap.parse_args()
    ref = Path(args.ref_dir)
    if args.merge:
        out = {"pairs": {}, "packing": {}, "copy_check": {}, "tp_check": {}}
        for part in args.merge.split(","):
            d = json.loads((ref / part).read_text())
            for key in ("pairs", "packing", "copy_check", "tp_check"):
                out[key].update(d.get(key, {}))
            out.update({k: v for k, v in d.items() if k in ("metric",)})
            out["runs"] = sorted(set(out.get("runs", [])) | set(d.get("runs", [])))
        tmp = ref / f".spreads.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, ref / "spreads.json")
        print("merged", args.merge)
        return
    tag = re.sub(r"[-:T]", "", args.date or ref.resolve().name)[:10]  # YYYYmmddHH
    if not re.fullmatch(r"\d{10}", tag):
        ap.error(f"cannot read a date from {args.date or ref.resolve().name!r}: give --date YYYY-mm-ddTHH")
    tag = f"{tag[:8]}T{tag[8:]}"
    DATE_STAMP = tag.replace("T", "") + "00"
    froot = Path(args.forecaster_root or Path(os.environ["VARDA_SCRATCH"]) / "realdata" / tag)
    tags = [t for t in args.runs.split(",") if (ref / t / "model_io.safetensors").exists()]
    fc_of = {"fp16": froot / "ref-fp16", "fp32": froot / "ref-fp32", "cpu-fp32": froot / "ref-fp32",
             "fp16-fp32in": froot / "ref-fp32"}
    steps = [int(s) for s in args.steps.split(",")]
    out = {"runs": tags, "metric": "float64; rel_absmax = max|a-r| / max|r|, r = second name of the pair", "pairs": {}}
    pairs = [("fp16", "fp32"), ("cpu-fp32", "fp32"), ("fp16", "cpu-fp32"), ("fp16-fp32in", "fp32"),
             ("fp16-fp32in", "cpu-fp32")]
    if args.pairs is not None:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    for a, b in pairs:
        if a in tags and b in tags:
            print(f"[pair] {a} vs {b}", flush=True)
            r = pair_model(ref / a, ref / b)
            if not args.skip_grib:
                r.update(pair_grib(ref / a, ref / b, steps))
            out["pairs"][f"{a}_vs_{b}"] = r
    out["packing"], out["copy_check"], out["tp_check"] = {}, {}, {}
    for t in ([] if args.no_extras else tags):
        print(f"[packing/copy/tp] {t}", flush=True)
        out["packing"][t] = packing(ref / t, fc_of[t])
        if (ref / t / "tp_stages.safetensors").exists():
            out["tp_check"][t] = tp_check(ref / t, fc_of[t])
        if not args.skip_grib:
            out["copy_check"][t] = copy_check(ref / t, fc_of[t])
    tmp = ref / f".{args.out_name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, ref / args.out_name)
    print("wrote", ref / args.out_name)


if __name__ == "__main__":
    main()
