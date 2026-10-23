"""Write the frozen-cache manifest: $VARDA_SCRATCH/realdata/<YYYYmmddTHH>/manifest.json (atomically).

Records the date, the pinned ECMWF runs/steps (pins.json), the sha256 and size of every cached file (KENDA raw
and constants, STAC documents, ECMWF full files and indexes), the HF templates and config, the checkpoint
sha256, and the sha256 of the reference env lock (requirements-ref-oper.lock.txt).

    $PY -I python/reference/realdata/make_manifest.py --date 2026-10-22T15
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "$VARDA_SCRATCH")
HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def tree(base: Path, rel_to: Path) -> dict:
    return {str(p.relative_to(rel_to)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(base.rglob("*")) if p.is_file() and not p.name.endswith((".lock", ".part"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    date = dt.datetime.fromisoformat(args.date)
    root = Path(VARDA_SCRATCH) / "realdata" / date.strftime("%Y%m%dT%H")
    hf = Path(VARDA_SCRATCH) / "hf"
    lock = HERE / "requirements-ref-oper.lock.txt"
    m = {
        "kind": "varda realdata frozen cache",
        "date": date.isoformat(),
        "input_dates": [(date - dt.timedelta(hours=6)).isoformat(), date.isoformat()],
        "pins": json.loads((root / "pins.json").read_text()),
        "cache": tree(root / "cache", root),
        "hf": {**tree(hf / "templates", hf),
               "varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml": None,
               "varda-forecaster-det-sgm/metadata.yaml": None},
        "checkpoint": {"path": str(hf / "varda-forecaster-det-sgm" / "model.ckpt")},
        "env_lock": {"path": str(lock), "sha256": sha256(lock)},
        "written_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    for k in ("varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml", "varda-forecaster-det-sgm/metadata.yaml",
              "varda-forecaster-det-sgm/pyproject.toml"):
        p = hf / k
        m["hf"][k] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    ck = hf / "varda-forecaster-det-sgm" / "model.ckpt"
    m["checkpoint"].update({"bytes": ck.stat().st_size, "sha256": sha256(ck)})
    m["totals"] = {"files": len(m["cache"]), "bytes": sum(v["bytes"] for v in m["cache"].values())}
    out = root / "manifest.json"
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out}: {m['totals']}, checkpoint {m['checkpoint']['sha256']}")


if __name__ == "__main__":
    main()
