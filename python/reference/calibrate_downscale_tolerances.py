"""Per-window, per-hour tolerances of the temporal downscaler's loop (tolerances/downscale/<model>.json) from the
reference runs' own spreads (docs/downscaler-reference.md, `spreads.json` of python/reference/realdata/compare_downscaler.py).

    python -I python/reference/calibrate_downscale_tolerances.py SPREADS.json OUT.json [--model varda-td] [--uuid U]

Metric (docs/format.md section 2, float64) per window k and output hour h of the raw model output (G, V_out):
  y      rel_absmax over the whole tensor (max_abs over all variables / absmax over all variables; dominated by z)
  y_var  the worst per-variable rel_absmax (each variable against its own absmax; the informative number)
Rules, the one-step rules of tolerances/<model>.json applied per (k, h), floors 1e-5 (fp32) / 1e-6, rounded up to
2 significant digits:
  fast-fp32-vs-ref       engine fp32 (CPU or CUDA) vs the original's fp32 run (cpu-fp32 or cuda fp32, same inputs):
                         max(1e-5, 3 x the original's cpu-fp32 vs cuda-fp32 spread)
  fast-fp16-vs-ref       engine fp16 vs the original's fp32 run, same inputs: 1.75 x the original's fp16-vs-fp32 spread
                         on the same inputs (`fp16-fp32in` vs `fp32`)
  fast-fp16-vs-ref-fp16  engine fp16 vs the original's fp16 run, same inputs: 2 x that spread (two roundings add)
Standard library only; provenance paths are written with $VARDA_SCRATCH.
"""
import argparse
import json
import math
import os
import sys


def round_up_2sig(x: float) -> float:
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    m = x / 10 ** (e - 1)
    return float(f"{math.ceil(m - 1e-9) * 10 ** (e - 1):.2g}")


def neutral(p: str) -> str:
    s = os.environ.get("VARDA_SCRATCH")
    return p.replace(s, "$VARDA_SCRATCH") if s and s in p else p


def spreads_of(pair: dict):
    """{(k, h): (overall rel_absmax, worst-variable rel_absmax)} from the per-variable entries."""
    out = {}
    for key, vars_ in pair["model_output"].items():
        k, h = (int(v) for v in key.split("."))
        max_abs = max(v["max_abs"] for v in vars_.values())
        absmax = max(v["absmax"] for v in vars_.values())
        worst = max(v["rel_absmax"] for v in vars_.values())
        out[(k, h)] = (max_abs / absmax if absmax > 0 else 0.0, worst)
    return out


def table(sp, factor, floor):
    ks = sorted({k for k, _ in sp})
    hs = sorted({h for _, h in sp})
    y = [[round_up_2sig(max(floor, factor * sp[(k, h)][0])) for h in hs] for k in ks]
    yv = [[round_up_2sig(max(floor, factor * sp[(k, h)][1])) for h in hs] for k in ks]
    return {"y": y, "y_var": yv}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spreads")
    ap.add_argument("out")
    ap.add_argument("--model", default="varda-td")
    ap.add_argument("--uuid", default="00e40f03-2f7e-4c1c-8e6d-91ed9062249f")
    a = ap.parse_args()
    d = json.load(open(a.spreads))
    P = d["pairs"]
    fp32 = spreads_of(P["cpu-fp32_vs_fp32"])
    fp16 = spreads_of(P["fp16-fp32in_vs_fp32"])
    doc = {
        "method": __doc__.strip(),
        "model": a.model,
        "checkpoint": {"uuid": a.uuid},
        "windows": len({k for k, _ in fp32}),
        "hours_per_window": len({h for _, h in fp32}),
        "keys": ["y", "y_var"],
        "fast-fp32-vs-ref": table(fp32, 3.0, 1e-5),
        "fast-fp16-vs-ref": table(fp16, 1.75, 1e-6),
        "fast-fp16-vs-ref-fp16": table(fp16, 2.0, 1e-6),
        "reference_spread": {
            "cpu-fp32_vs_fp32": {f"w{k}.h{h}": {"y": v[0], "y_var": v[1]} for (k, h), v in sorted(fp32.items())},
            "fp16-fp32in_vs_fp32": {f"w{k}.h{h}": {"y": v[0], "y_var": v[1]} for (k, h), v in sorted(fp16.items())},
        },
        "sources": {"spreads": neutral(os.path.abspath(a.spreads)), "runs": d.get("runs"), "metric": d.get("metric")},
    }
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, a.out)
    w = doc["fast-fp32-vs-ref"]["y_var"]
    print(f"wrote {a.out}: {doc['windows']} windows x {doc['hours_per_window']} hours; fast-fp32-vs-ref y_var "
          f"{min(map(min, w)):.1e}..{max(map(max, w)):.1e}, fast-fp16-vs-ref y_var "
          f"{min(map(min, doc['fast-fp16-vs-ref']['y_var'])):.1e}..{max(map(max, doc['fast-fp16-vs-ref']['y_var'])):.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
