"""Compare the dumps (and GRIB outputs) of two real-data reference runs tensor by tensor.

Used to show the offline replay is exact (stub run --online vs offline: every input-side dump bitwise equal) and to
summarise fp16 vs fp32. NaN-aware: NaN == NaN at the same place counts as equal; a NaN on one side only is reported.

    $PY -I python/reference/realdata/compare_runs.py RUN_A RUN_B [--only source.,cutout.,input_] [--json OUT]
Exit code 1 unless every compared tensor is bitwise equal (and the GRIB files are byte-identical) when --expect-equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetensors_io import read_safetensors  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmp(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "equal": False}
    if a.dtype == np.bool_ or a.dtype.kind in "iu":
        n = int((a != b).sum())
        return {"equal": n == 0, "differ": n}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.isnan(a64), np.isnan(b64)
    nan_mismatch = int((na != nb).sum())
    both = ~(na | nb)
    d = np.abs(a64[both] - b64[both])
    absmax = float(np.abs(b64[both]).max()) if both.any() else 0.0
    max_abs = float(d.max()) if d.size else 0.0
    return {"equal": nan_mismatch == 0 and max_abs == 0.0, "max_abs": max_abs, "absmax": absmax,
            "rel_absmax": max_abs / absmax if absmax else (0.0 if max_abs == 0 else float("inf")),
            "differ": int((d != 0).sum()), "nan_mismatch": nan_mismatch, "nan": int(na.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--only", default="", help="comma-separated dump-name prefixes")
    ap.add_argument("--json", default=None)
    ap.add_argument("--expect-equal", action="store_true")
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    prefixes = [p for p in args.only.split(",") if p]
    names = sorted({p.name for p in (A / "dumps").glob("*.safetensors")} & {p.name for p in (B / "dumps").glob("*.safetensors")})
    only_a = sorted({p.name for p in (A / "dumps").glob("*.safetensors")} - set(names))
    only_b = sorted({p.name for p in (B / "dumps").glob("*.safetensors")} - set(names))
    out, all_equal = {"dumps": {}, "only_a": only_a, "only_b": only_b}, True
    for n in names:
        if prefixes and not any(n.startswith(p) for p in prefixes):
            continue
        _, ta = read_safetensors(A / "dumps" / n)
        _, tb = read_safetensors(B / "dumps" / n)
        res = {}
        for k in sorted(set(ta) | set(tb)):
            if k not in ta or k not in tb:
                res[k] = {"equal": False, "missing": "a" if k not in ta else "b"}
            else:
                res[k] = cmp(np.asarray(ta[k]), np.asarray(tb[k]))
            all_equal &= res[k]["equal"]
        eq = sum(r["equal"] for r in res.values())
        worst = max((r.get("max_abs", 0.0) for r in res.values()), default=0.0)
        print(f"{n:48s} {eq}/{len(res)} tensors bitwise, worst max_abs {worst:.3e}")
        out["dumps"][n] = res
    if not prefixes:
        ga = {p.name: sha(p) for p in (A / "outputs").glob("*.grib")}
        gb = {p.name: sha(p) for p in (B / "outputs").glob("*.grib")}
        out["grib"] = {n: ga.get(n) == gb.get(n) for n in sorted(set(ga) | set(gb))}
        for n, e in out["grib"].items():
            print(f"grib {n:40s} {'identical' if e else 'DIFFERENT'}")
            all_equal &= e
    out["all_equal"] = all_equal
    print("all bitwise equal:", all_equal)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    if args.expect_equal and not all_equal:
        sys.exit(1)


if __name__ == "__main__":
    main()
