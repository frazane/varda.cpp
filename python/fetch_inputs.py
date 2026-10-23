"""Download the open data one Varda forecast needs, into a per-date input cache that `varda-forecast --realdata`
reads. Standard library only, Python >= 3.9 (any python3; on this cluster `.venv/bin/python`). Runs where there is
network (the login node here). Tools only: never imported by src/, apps/ or tests/.

    fetch_inputs.py DATE|latest [--root DIR] [--io-config varda-io.json] [--source google]
                    [--ecmwf-latest RUN] [--constants-run RUN] [--full-ecmwf] [--jobs N]

DATE is the initial time t in UTC (2026-10-22T15; t.hour % 3 == 0). `latest` applies the Hugging Face notebook's rule:
the latest KENDA-CH1 analysis t with t - 6h also published. The input directory is printed on stdout (one line);
progress goes to stderr.

Layout (the one `docs/realdata-reference.md` describes, so `--realdata DIR` reads it)::

    DIR = <root>/<YYYYmmddTHH>/
      cache/kenda/raw/<YYYYmmddTHHMMSS>/<VAR>.grib2      t-6h and t: only the variables the run reads (11)
      cache/kenda/constants/<asset>.grib2                ICON-CH1 horizontal and vertical constants
      cache/kenda/stac/<YYYYmmddTHHMMSS>.json, collection_icon-ch1.json   the STAC documents (provenance)
      cache/ecmwf/files/<run>-<step>h-oper-fc.subset.grib2 (+ .index)    only the messages the run reads, as
                                                          byte ranges of the published file (exact bytes)
      pins.json                                          the choices (written last: present = complete)

ECMWF choices, as anemoi-plugins-meteoswiss' oper-ecmwf-opendata makes them at start time:
  * `latest`: the latest published run (HEAD probe, as `ecmwf.opendata.Client.latest`), at fetch time; --ecmwf-latest
    pins it instead;
  * prognostics at t-6h and t: the run and step `resolve(valid, latest)` gives: the analysis when its run is
    published, else a forecast of the latest run (always at 03/09/15/21Z: the 3 h forecast of the run before);
  * constants (z lsm sdor slor, `constant: true`): step 0 of `latest` (anemoi takes the latest run at start time).
    --constants-run RUN names another run. The choice is recorded in pins.json["constant_forcings_run"].

A directory that already has pins.json is complete: it is reused as is, without network (an explicit DATE), so a
forecast can be rerun on exactly the same inputs. `latest` needs the network to resolve the date.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not add the script dir
import opendata as od  # noqa: E402
from opendata import utcnow  # noqa: E402

SCHEMA = "varda-inputs/1"
STEP_H = 6  # the model's step: inputs at t - 6h and t (checked against the io config when given)

# What the run reads when no io config is given (python/export_io_config.py's varda-io.json of Varda-single-1.0
# gives the same sets; derive_wanted() derives them from it).
KENDA_VARS = ["P", "PMSL", "PS", "QV", "T", "T_2M", "TD_2M", "U", "U_10M", "V", "V_10M"]
IFS_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
IFS_PROGNOSTIC = [(p, "pl", lev) for p in ("gh", "q", "t", "u", "v") for lev in IFS_LEVELS] + \
    [(p, "sfc", None) for p in ("10u", "10v", "2d", "2t", "msl", "sp")]
IFS_CONSTANT = [(p, "sfc", None) for p in ("z", "lsm", "sdor", "slor")]
# KENDA auxiliaries the vertical interpolation reads (src/varda/io/input.cpp kenda_prognostics: P, T_2M, PS); FI
# is computed from HHL (vertical constants), never read.
KENDA_AUX = ["P", "T_2M", "PS"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(msg: str):
    print(f"fetch_inputs: {msg}", file=sys.stderr)
    sys.exit(1)


def derive_wanted(io_config: str):
    """KENDA variables and IFS messages from varda-io.json, by the rules of src/varda/io/input.cpp."""
    cfg = json.load(open(io_config))
    if cfg.get("schema") != "varda-io/1":
        fail(f"{io_config}: schema {cfg.get('schema')!r}, expected varda-io/1")
    if cfg["timestep_hours"] != STEP_H:
        fail(f"{io_config}: timestep {cfg['timestep_hours']} h; this tool fetches t-{STEP_H}h and t")
    kenda, prog, const = set(KENDA_AUX), [], []
    for v in cfg["variables"]:
        cats = v.get("categories") or []
        if not v.get("in_model") or v.get("computed_forcing"):
            continue
        m = v.get("mars") or {}
        lt = m.get("levtype")
        if "prognostic" in cats:
            if m.get("param") != "FI":
                kenda.add(m["param"])
            if lt == "pl":
                lev = int(m["levelist"])
                base = v["name"][: -len(f"_{lev}")]
                prog.append(("gh" if base == "z" else base, "pl", lev))
            else:
                prog.append((v["name"], "sfc", None))
        elif "constant" in cats:
            const.append((v["name"], "sfc", None))
    return sorted(kenda), prog, const


def kenda_tag(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M%S")


def check_complete(d: str) -> list[str]:
    """Files a complete input directory must have (from its pins.json); returns the missing ones."""
    pins = json.load(open(os.path.join(d, "pins.json")))
    t = od.parse_date(pins["date"])
    need = [os.path.join(d, "cache", "kenda", "constants", n) for n in
            ("horizontal_constants_icon-ch1-eps.grib2", "vertical_constants_icon-ch1-eps.grib2")]
    kvars = pins.get("kenda", {}).get("variables", KENDA_VARS)
    for valid in (t - timedelta(hours=STEP_H), t):
        need += [os.path.join(d, "cache", "kenda", "raw", kenda_tag(valid), f"{v}.grib2") for v in kvars]
    need += [os.path.join(d, "cache", "ecmwf", "files", e["file"]) for e in pins["ecmwf_files"].values()]
    return [p for p in need if not os.path.exists(p)]


def kenda_items(t: datetime, variables: list[str]) -> dict:
    """The STAC items of t-6h and t; fails (writing nothing) unless every needed variable is published."""
    out = {}
    for valid in (t - timedelta(hours=STEP_H), t):
        items = od.stac_items(valid)
        by_var = {}
        for it in items:
            var = it["properties"]["forecast:variable"]
            assets = list(it["assets"].values())
            if len(assets) != 1 or var in by_var:
                fail(f"KENDA-CH1 {valid:%Y-%m-%dT%H}: unexpected STAC item layout for {var} ({it['id']})")
            by_var[var] = assets[0]["href"]
        missing = [v for v in variables if v not in by_var]
        if missing:
            if not items:
                why = "nothing is published for it: KENDA-CH1 open data are kept about 24 h"
            else:
                why = (f"{len(items)} of the analysis' variables are published, not {', '.join(missing)}: "
                       "either being published (retry in a few minutes) or expiring (open data are kept about 24 h)")
            fail(f"KENDA-CH1 analysis {valid:%Y-%m-%dT%H}Z is incomplete: {why}. "
                 "`latest` picks the newest complete pair; an older date can only be run from an existing cache.")
        out[valid] = (items, by_var)
    return out


def fetch_kenda(t: datetime, base: str, variables: list[str], found: dict, jobs: int) -> dict:
    """The KENDA-CH1 files of t-6h and t (only `variables`, from kenda_items) and the ICON-CH1 constants."""
    out = {"variables": variables, "items": {}, "constants": {}}
    todo = []
    for valid, (items, by_var) in found.items():
        tag = kenda_tag(valid)
        od.atomic_write_json(os.path.join(base, "stac", f"{tag}.json"), items)
        out["items"][tag] = {v: by_var[v] for v in variables}
        todo += [(by_var[v], os.path.join(base, "raw", tag, f"{v}.grib2")) for v in variables]
    coll = od.icon_collection()
    od.atomic_write_json(os.path.join(base, "stac", "collection_icon-ch1.json"), coll)
    consts = od.icon_constant_assets(coll)
    for n in ("horizontal_constants_icon-ch1-eps.grib2", "vertical_constants_icon-ch1-eps.grib2"):
        if n not in consts:
            fail(f"ICON-CH1 collection has no asset {n} (assets: {sorted(consts)})")
    out["constants"] = consts
    todo += [(href, os.path.join(base, "constants", n)) for n, href in consts.items()]
    with cf.ThreadPoolExecutor(jobs) as ex:
        for msg in ex.map(lambda j: od.download(*j), todo):
            log(msg)
    return out


def select_messages(index: list[dict], wanted: list[tuple], what: str) -> list[dict]:
    """The index lines of `wanted` (param, levtype, level): exactly one each, returned in file order."""
    out = []
    for p, lt, lev in wanted:
        hit = [e for e in index if e.get("param") == p and e.get("levtype") == lt
               and (lt != "pl" or str(e.get("levelist")) == str(lev))]
        if len(hit) != 1:
            fail(f"{what}: {len(hit)} index entries for {p} {lt} {lev or ''} (expected 1)")
        out.append(hit[0])
    return sorted(out, key=lambda e: e["_offset"])


def merge_ranges(lines: list[dict]) -> list[tuple[int, int]]:
    out = []
    for e in lines:
        off, n = int(e["_offset"]), int(e["_length"])
        if out and out[-1][0] + out[-1][1] == off:
            out[-1] = (out[-1][0], out[-1][1] + n)
        else:
            out.append((off, n))
    return out


def check_grib_messages(path: str, lines: list[dict]) -> None:
    with open(path, "rb") as f:
        for e in lines:
            b = f.read(int(e["_length"]))
            if b[:4] != b"GRIB" or b[-4:] != b"7777":
                fail(f"{path}: message {e['param']} {e.get('levelist', '')} is not a whole GRIB message")
        if f.read(1):
            fail(f"{path}: trailing bytes")


def fetch_ecmwf(t: datetime, base: str, latest: datetime, constants_run: datetime, prog: list, const: list,
                source: str, subset: bool, jobs: int) -> dict:
    """The IFS open-data files (or their needed messages) for t-6h, t and the constants run."""
    input_dates, roles = {}, {}
    for valid in (t - timedelta(hours=STEP_H), t):
        init, lead = od.resolve(valid, latest, allow_forecast_fallback=True)
        input_dates[valid.isoformat()] = {"init": init.isoformat(), "step": lead}
        roles.setdefault((init, lead), []).extend(prog)
    roles.setdefault((constants_run, 0), []).extend(const)
    files = {}

    def one(item):
        (init, step), wanted = item
        url = od.ecmwf_url(init, step, source)
        name = os.path.basename(url)
        stem = os.path.splitext(name)[0]
        key = f"{init.isoformat()}+{step}h"
        if not subset:
            try:
                msg = od.download(url, os.path.join(base, name))
            except od.urllib.error.HTTPError as e:
                fail(f"ECMWF {url}: HTTP {e.code} (run {init:%Y-%m-%dT%H} step {step} h not (or no longer) "
                     "published; open data keep about 4 days)")
            log(msg)
            return key, {"url": url, "file": name, "subset": False, "bytes": os.path.getsize(os.path.join(base, name))}
        try:
            text = od.get_bytes(os.path.splitext(url)[0] + ".index").decode()
        except od.urllib.error.HTTPError as e:
            fail(f"ECMWF {url}: index HTTP {e.code} (run {init:%Y-%m-%dT%H} step {step} h not (or no longer) "
                 "published; open data keep about 4 days)")
        od.atomic_write_bytes(os.path.join(base, f"{stem}.index"), text.encode())
        lines = select_messages(od.read_index(text), list(dict.fromkeys(wanted)), name)
        for e in lines:
            if int(e["step"]) != step or e["date"] != f"{init:%Y%m%d}" or e["time"] != f"{init:%H}00":
                fail(f"{name}: index line {e} is not run {init} step {step}")
        out = os.path.join(base, f"{stem}.subset.grib2")
        if os.path.exists(out):
            log(f"keep {out}")
        else:
            n = od.download_ranges(url, merge_ranges(lines), out)
            log(f"got  {out} ({len(lines)} messages, {n / 1e6:.1f} MB of {name})")
        check_grib_messages(out, lines)
        return key, {"url": url, "file": os.path.basename(out), "subset": True, "messages": len(lines),
                     "index": f"{stem}.index", "bytes": os.path.getsize(out),
                     "fields": [f"{e['param']}@{e.get('levelist', e['levtype'])}" for e in lines]}

    with cf.ThreadPoolExecutor(max(1, min(jobs, len(roles)))) as ex:
        for key, entry in ex.map(one, sorted(roles.items())):
            files[key] = entry
    return {"input_dates": input_dates, "ecmwf_files": files}


def dir_bytes(d: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("date", help="YYYY-MM-DDTHH (UTC) or 'latest'")
    default_root = os.environ.get("VARDA_INPUTS") or os.path.join(
        os.environ.get("VARDA_SCRATCH") or os.path.expanduser("~/.cache/varda"), "inputs")
    ap.add_argument("--root", default=default_root, help=f"input cache root (default $VARDA_INPUTS = {default_root})")
    ap.add_argument("--io-config", help="varda-io.json: derive the variables from it (default: Varda-single-1.0's)")
    ap.add_argument("--source", default="google", choices=sorted(od.ECMWF_URLS), help="ECMWF open-data mirror")
    ap.add_argument("--ecmwf-latest", help="pin the 'latest published run' (YYYY-MM-DDTHH) instead of probing")
    ap.add_argument("--constants-run", help="run of the global constants (default: the latest run)")
    ap.add_argument("--full-ecmwf", action="store_true", help="download whole IFS files, not only the messages used")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    a = ap.parse_args()

    try:
        if a.date == "latest":
            t = od.pick_latest_kenda()
            log(f"latest KENDA-CH1 analysis with t-6h published: {t:%Y-%m-%dT%H}Z")
        else:
            t = od.parse_date(a.date)
    except (ValueError, od.OpenDataError) as e:
        fail(str(e))
    if t.minute or t.second or t.hour % 3:
        fail(f"{t:%Y-%m-%dT%H:%M}: the initial time must be a whole hour with hour % 3 == 0 (00, 03, ..., 21 UTC)")
    d = os.path.join(a.root, t.strftime("%Y%m%dT%H"))
    if os.path.exists(os.path.join(d, "pins.json")):
        missing = check_complete(d)
        if missing:
            fail(f"{d} has pins.json but misses {len(missing)} file(s), e.g. {missing[0]}; "
                 "remove that directory and fetch again (if the date is still in open data)")
        log(f"inputs for {t:%Y-%m-%dT%H}Z already complete in {d}")
        print(d)
        return

    kenda_vars, prog, const = KENDA_VARS, IFS_PROGNOSTIC, IFS_CONSTANT
    if a.io_config:
        kenda_vars, prog, const = derive_wanted(a.io_config)
    t0 = time.time()
    try:
        found = kenda_items(t, kenda_vars)  # before anything is written: an unpublished date leaves no directory
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free
        if free < 4e9:
            fail(f"{d}: only {free / 1e9:.1f} GB free; a date needs about 2.5 GB")
        kenda = fetch_kenda(t, os.path.join(d, "cache", "kenda"), kenda_vars, found, a.jobs)
        probed_at = utcnow()
        latest = od.parse_date(a.ecmwf_latest) if a.ecmwf_latest else od.ecmwf_latest_run(a.source)
        crun = od.parse_date(a.constants_run) if a.constants_run else latest
        log(f"ECMWF latest published run: {latest:%Y-%m-%dT%H}Z ({'pinned' if a.ecmwf_latest else 'probed'}); "
            f"constants from {crun:%Y-%m-%dT%H}Z+0h")
        ec = fetch_ecmwf(t, os.path.join(d, "cache", "ecmwf", "files"), latest, crun, prog, const, a.source,
                         not a.full_ecmwf, a.jobs)
    except od.OpenDataError as e:
        fail(str(e))
    pins = {
        "schema": SCHEMA,
        "date": t.isoformat(),
        "input_dates": ec["input_dates"],
        "ecmwf_files": ec["ecmwf_files"],
        "ecmwf_source": a.source,
        "ecmwf_latest_published_run": latest.isoformat(),
        "ecmwf_latest_probed_at_utc": None if a.ecmwf_latest else probed_at.isoformat(timespec="seconds"),
        "constant_forcings_run": {
            "init": crun.isoformat(), "step": 0,
            "rule": "--constants-run" if a.constants_run else
                    "latest published run at fetch time (anemoi-inference takes the latest run at start time)",
        },
        "kenda": kenda,
        "fetched_at_utc": utcnow().isoformat(timespec="seconds"),
        "fetch_seconds": round(time.time() - t0, 1),
        "tool": "python/fetch_inputs.py",
    }
    pins["bytes"] = dir_bytes(d)
    od.atomic_write_json(os.path.join(d, "pins.json"), pins)
    log(f"inputs for {t:%Y-%m-%dT%H}Z complete in {d}: {pins['bytes'] / 1e9:.2f} GB, {pins['fetch_seconds']:.0f} s")
    print(d)


if __name__ == "__main__":
    main()
