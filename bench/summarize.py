#!/usr/bin/env python3
"""Turn a bench job directory ($VARDA_SCRATCH/jobs/bench/<jobid>) into one markdown table (stdlib only).

    python3 bench/summarize.py JOBDIR [JOBDIR ...]

Reads JOBDIR/meta.json and JOBDIR/results/<model>.<impl>.<precision>.json[l] (impl: original | pyengine |
varda). Python results are the JSON lines of bench_reference.py / bench.py (median_s, min_s, first_call_s,
peak_alloc_*_gb); varda-bench JSON is read leniently (median_ms|median_s, min_ms|min_s, first_call_s|first_call_ms,
peak_mem_gb|peak_memory_gb|peak_mem_mb|..., or a "times_ms" list). Speedup = original warm median / this median
(same model, precision, same job).
"""
import json
import statistics
import sys
from pathlib import Path

NAMES = {"original": "original anemoi (triton)", "pyengine": "Python engine fast (compile + CUDA graph)",
         "varda": "varda.cpp"}
ORDER = {"original": 0, "pyengine": 1, "varda": 2}


def first(d, keys, scale=1.0):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return d[k] * scale
    return None


def norm(r):
    ms = r.get("times_ms") or ([t * 1e3 for t in r["times_s"]] if r.get("times_s") else None)
    med = first(r, ["median_ms", "warm_ms_median"]) or first(r, ["median_s"], 1e3) or (statistics.median(ms) if ms else None)
    mn = first(r, ["min_ms", "warm_ms_min"]) or first(r, ["min_s"], 1e3) or (min(ms) if ms else None)
    fc = first(r, ["first_call_s"]) or first(r, ["first_call_ms"], 1e-3)
    pk = [v for v in (first(r, ["peak_mem_gb", "peak_memory_gb", "peak_device_gb", "peak_gb"]),
                      first(r, ["peak_mem_mb", "peak_memory_mb"], 1e-3),
                      first(r, ["peak_mem_bytes", "peak_memory_bytes"], 1e-9),
                      first(r, ["peak_alloc_forward_gb"]), first(r, ["peak_alloc_load_warmup_gb"])) if v]
    n = r.get("repeats") or r.get("iters") or (len(ms) if ms else None)
    return dict(med=med, mn=mn, fc=fc, peak=max(pk) if pk else None, n=n)


def load(path):
    txt = path.read_text().strip()
    if not txt:
        return None
    try:
        recs = [json.loads(txt)]
    except json.JSONDecodeError:
        recs = [json.loads(l) for l in txt.splitlines() if l.startswith("{")]
    return norm(recs[-1]) if recs else None


def fmt(v, f):
    return "-" if v is None else f % v


def table(job):
    job = Path(job)
    meta = json.loads((job / "meta.json").read_text()) if (job / "meta.json").exists() else {}
    codes = {}
    if (job / "exit_codes.txt").exists():
        codes = dict(l.split() for l in (job / "exit_codes.txt").read_text().splitlines() if len(l.split()) == 2)
    rows = []
    for p in (job / "results").glob("*"):
        parts = p.name.split(".")
        if len(parts) < 4 or parts[-1] not in ("json", "jsonl"):
            continue
        m, impl, prec = parts[0], parts[1], parts[2]
        try:
            r = load(p)
        except (ValueError, KeyError) as e:
            print(f"warning: {p}: {e}", file=sys.stderr)
            continue
        if r:
            rows.append((m, prec, impl, r))
    base = {(m, prec): r["med"] for m, prec, impl, r in rows if impl == "original"}
    rows.sort(key=lambda x: (x[0], x[1], ORDER.get(x[2], 9)))
    out = [f"### Benchmark, job {meta.get('job_id', job.name)}", "",
           f"GPU: {meta.get('gpu', '?')} ({meta.get('mem_total', '?')}), driver {meta.get('driver', '?')}, "
           f"node {meta.get('node', '?')}, job {meta.get('job_id', job.name)}. One forward pass (predict_step), "
           "hook-free, warm, input on the device, TF32 off.", "",
           "| model | precision | implementation | warm median ms | warm min ms | iters | first call s | peak device GB | speedup vs original |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m, prec, impl, r in rows:
        b = base.get((m, prec))
        sp = f"{b / r['med']:.2f}x" if b and r["med"] else "-"
        out.append(f"| {m} | {prec} | {NAMES.get(impl, impl)} | {fmt(r['med'], '%.1f')} | {fmt(r['mn'], '%.1f')} | "
                   f"{r['n'] or '-'} | {fmt(r['fc'], '%.1f')} | {fmt(r['peak'], '%.1f')} | {sp} |")
    bad = {k: v for k, v in codes.items() if v not in ("0",)}
    if bad:
        out += ["", "Steps with non-zero exit or skipped: " + ", ".join(f"{k}={v}" for k, v in bad.items())]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("\n".join(table(j) for j in sys.argv[1:]))
