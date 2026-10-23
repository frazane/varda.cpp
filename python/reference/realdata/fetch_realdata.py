"""Freeze everything a Varda real-data run needs into a permanent offline cache.

Runs on the login node (network). Layout under ``$VARDA_SCRATCH/realdata/<YYYYmmddTHH>/``::

    cache/kenda/raw/<YYYYmmddTHHMMSS>/<VAR>.grib2   # same layout as oper-kenda-opendata's cache_dir
    cache/kenda/constants/<asset>.grib2             # ICON-CH1 collection-level static assets
    cache/kenda/stac/<YYYYmmddTHHMMSS>.json         # the STAC items (to replay _stac_items offline)
    cache/kenda/stac/collection_icon-ch1.json       # the collection document (static asset names)
    cache/ecmwf/files/<run>-<step>h-<stream>-fc.grib2 (+ .index)   # full ECMWF open-data files
    pins.json                                        # date, ECMWF latest run, per-date run/step
    manifest.json (written by make_manifest.py)

Subcommands::

    fetch_realdata.py pick                      # print the notebook's choice of date
    fetch_realdata.py kenda  --date 2026-10-22T15
    fetch_realdata.py ecmwf  --date 2026-10-22T15 [--latest 2026-10-22T18]
    fetch_realdata.py templates

Every file is written to ``<name>.part`` and renamed. Existing files are kept.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from datetime import datetime, timedelta

# Shared with python/fetch_inputs.py: HTTP (stdlib), STAC queries, the notebook's date rule, the ECMWF run rule.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # python/ (-I safe)
from opendata import (  # noqa: E402
    HF_REPO, atomic_write_json, download, get_json, icon_collection, pick_latest_kenda, resolve, stac_items,
)

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "$VARDA_SCRATCH")


def root(date: datetime) -> str:
    return os.path.join(VARDA_SCRATCH, "realdata", date.strftime("%Y%m%dT%H"))


def cmd_pick(args) -> None:
    """The notebook's rule: latest KENDA-CH1 analysis t (t.hour % 3 == 0) with t-6h published."""
    print(pick_latest_kenda().isoformat())


def cmd_kenda(args) -> None:
    date = datetime.fromisoformat(args.date)
    base = os.path.join(root(date), "cache", "kenda")
    jobs = []
    for valid in (date - timedelta(hours=6), date):
        tag = valid.strftime("%Y%m%dT%H%M%S")
        items = stac_items(valid)
        atomic_write_json(os.path.join(base, "stac", f"{tag}.json"), items)
        for it in items:
            var = it["properties"]["forecast:variable"]
            assets = list(it["assets"].values())
            assert len(assets) == 1, it["id"]
            jobs.append((assets[0]["href"], os.path.join(base, "raw", tag, f"{var}.grib2")))
    coll = icon_collection()
    atomic_write_json(os.path.join(base, "stac", "collection_icon-ch1.json"), coll)
    for name, asset in coll.get("assets", {}).items():
        if asset.get("type") == "application/grib":
            jobs.append((asset["href"], os.path.join(base, "constants", name)))
    with cf.ThreadPoolExecutor(8) as ex:
        for msg in ex.map(lambda j: download(*j), jobs):
            print(msg, flush=True)


def cmd_ecmwf(args) -> None:
    from ecmwf.opendata import Client

    date = datetime.fromisoformat(args.date)
    client = Client(source="google")
    if args.latest:
        latest = datetime.fromisoformat(args.latest)
    else:
        latest = client.latest(type="fc")  # what oper-ecmwf-opendata's _latest_published_run returns now
    pins = {
        "date": date.isoformat(),
        "ecmwf_source": "google",
        "ecmwf_latest_published_run": latest.isoformat(),
        "ecmwf_latest_probed_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "input_dates": {},
        "constant_forcings_run": {"init": latest.isoformat(), "step": 0},
    }
    wanted = {(latest, 0)}
    for valid in (date - timedelta(hours=6), date):
        init, lead = resolve(valid, latest, allow_forecast_fallback=True)
        pins["input_dates"][valid.isoformat()] = {"init": init.isoformat(), "step": lead}
        wanted.add((init, lead))
    base = os.path.join(root(date), "cache", "ecmwf", "files")
    files = {}
    for init, step in sorted(wanted):
        res = client._get_urls(date=init.strftime("%Y%m%d"), time=init.hour, step=step, type="fc", stream="oper")
        assert len(res.urls) == 1, res.urls
        url = res.urls[0]
        name = os.path.basename(url)
        print(download(url, os.path.join(base, name)), flush=True)
        print(download(os.path.splitext(url)[0] + ".index", os.path.join(base, os.path.splitext(name)[0] + ".index")))
        files[f"{init.isoformat()}+{step}h"] = {"url": url, "file": name}
    pins["ecmwf_files"] = files
    path = os.path.join(root(date), "pins.json")
    if os.path.exists(path) and not args.force:
        old = json.load(open(path))
        if old["ecmwf_latest_published_run"] != pins["ecmwf_latest_published_run"]:
            sys.exit(f"pins.json exists with a different latest run: {old['ecmwf_latest_published_run']}")
        print("pins.json unchanged")
        return
    atomic_write_json(path, pins)
    print(json.dumps(pins, indent=2))


def cmd_templates(args) -> None:
    """Fetch only templates/ of the HF repo (no huggingface_hub in the ref env: plain HTTP API)."""
    info = get_json(f"https://huggingface.co/api/models/{HF_REPO}", timeout=30)
    rev = info["sha"]
    tree = get_json(f"https://huggingface.co/api/models/{HF_REPO}/tree/{rev}/templates?recursive=true", timeout=30)
    dest = os.path.join(VARDA_SCRATCH, "hf")
    for x in tree:
        if x["type"] != "file":
            continue
        print(download(f"https://huggingface.co/{HF_REPO}/resolve/{rev}/{x['path']}", os.path.join(dest, x["path"])))
    atomic_write_json(os.path.join(dest, "templates", "REVISION.json"), {"repo": HF_REPO, "sha": rev})


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick")
    k = sub.add_parser("kenda")
    k.add_argument("--date", required=True)
    e = sub.add_parser("ecmwf")
    e.add_argument("--date", required=True)
    e.add_argument("--latest", default=None, help="pin ECMWF latest run instead of probing")
    e.add_argument("--force", action="store_true")
    sub.add_parser("templates")
    args = p.parse_args()
    {"pick": cmd_pick, "kenda": cmd_kenda, "ecmwf": cmd_ecmwf, "templates": cmd_templates}[args.cmd](args)


if __name__ == "__main__":
    main()
