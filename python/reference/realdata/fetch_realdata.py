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

import requests

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "$VARDA_SCRATCH")
STAC = "https://data.geo.admin.ch/api/stac/v1"
KENDA = "ch.meteoschweiz.ogd-analysis-kenda-ch1"
ICON = "ch.meteoschweiz.ogd-forecasting-icon-ch1"
HF_REPO = "MeteoSwiss/Varda-single-1.0"

# ECMWF open-data facts, copied from anemoi_plugins_meteoswiss/inference/inputs/oper_ecmwf_opendata.py
FREQUENCY_H, STEP_H, MAX_LEAD_TIME_H, STORED_RUNS = 6, 3, 144, 12


def root(date: datetime) -> str:
    return os.path.join(VARDA_SCRATCH, "realdata", date.strftime("%Y%m%dT%H"))


def atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: str, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode())


def download(href: str, path: str) -> str:
    if os.path.exists(path):
        return f"keep {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part"
    with requests.get(href, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    os.replace(tmp, path)
    return f"got  {path} ({os.path.getsize(path) / 1e6:.1f} MB)"


def stac_items(valid_time: datetime) -> list[dict]:
    """Same query as oper_kenda_opendata._stac_items."""
    url = f"{STAC}/collections/{KENDA}/items"
    params = {"datetime": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 100}
    out = []
    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        page = r.json()
        out += page.get("features", [])
        nxt = [link["href"] for link in page.get("links", []) if link.get("rel") == "next"]
        url, params = (nxt[0], None) if nxt else (None, None)
    return out


def cmd_pick(args) -> None:
    """The notebook's rule: latest KENDA-CH1 analysis t (t.hour % 3 == 0) with t-6h published."""
    features = requests.post(
        f"{STAC}/search",
        json={
            "collections": [KENDA],
            "forecast:variable": "P",
            "forecast:horizon": "P0DT00H00M00S",
            "forecast:perturbed": False,
            "limit": 100,
        },
        timeout=60,
    ).json()["features"]
    available = {
        datetime.fromisoformat(f["properties"]["forecast:reference_datetime"].split("/")[0].removesuffix("Z"))
        for f in features
    }
    latest = max(t for t in available if t.hour % 3 == 0 and t - timedelta(hours=6) in available)
    print(latest.isoformat())


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
    coll = requests.get(f"{STAC}/collections/{ICON}", timeout=30)
    coll.raise_for_status()
    coll = coll.json()
    atomic_write_json(os.path.join(base, "stac", "collection_icon-ch1.json"), coll)
    for name, asset in coll.get("assets", {}).items():
        if asset.get("type") == "application/grib":
            jobs.append((asset["href"], os.path.join(base, "constants", name)))
    with cf.ThreadPoolExecutor(8) as ex:
        for msg in ex.map(lambda j: download(*j), jobs):
            print(msg, flush=True)


def resolve(target: datetime, latest: datetime, allow_forecast_fallback: bool) -> tuple[datetime, int]:
    """Copy of OperEcmwfOpenDataInput._resolve_init_time_and_lead_hours (same checks)."""
    hour = timedelta(hours=1)
    offset = latest - target
    assert offset % hour == timedelta(0)
    hours_ahead = offset // hour
    n_back = -(-max(hours_ahead, 0) // FREQUENCY_H)
    assert n_back <= STORED_RUNS
    init = latest - timedelta(hours=FREQUENCY_H * n_back)
    lead = (target - init) // hour
    assert 0 <= lead <= MAX_LEAD_TIME_H and lead % STEP_H == 0
    assert lead == 0 or allow_forecast_fallback
    return init, lead


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
    info = requests.get(f"https://huggingface.co/api/models/{HF_REPO}", timeout=30).json()
    rev = info["sha"]
    tree = requests.get(f"https://huggingface.co/api/models/{HF_REPO}/tree/{rev}/templates?recursive=true", timeout=30)
    tree.raise_for_status()
    dest = os.path.join(VARDA_SCRATCH, "hf")
    for x in tree.json():
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
