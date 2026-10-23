"""Open-data access shared by python/fetch_inputs.py (users) and python/reference/realdata/fetch_realdata.py
(the frozen reference cache). Standard library only (urllib), Python >= 3.8. Tools only: never imported by
src/, apps/ or tests/.

Sources:
  * KENDA-CH1 analyses: the data.geo.admin.ch STAC API (collection ch.meteoschweiz.ogd-analysis-kenda-ch1),
    with the ICON-CH1 collection's static assets (horizontal and vertical constants);
  * ECMWF IFS open data (0.25 deg, stream oper, type fc) on a mirror (default: google, as the reference run).

The rules copied from the reference stack, with where they come from:
  * `pick_latest_kenda`: the Hugging Face notebook's choice of the initial date (the latest KENDA-CH1 analysis t with
    t.hour % 3 == 0 and t - 6h also published);
  * `ecmwf_latest_run`: `ecmwf.opendata.Client(source).latest(type="fc")`, which anemoi-plugins-meteoswiss'
    oper-ecmwf-opendata calls at start time (HEAD on the step-0 file, from today 18z back in 6 h steps, 2 days);
  * `resolve`: OperEcmwfOpenDataInput._resolve_init_time_and_lead_hours (run and step for a valid time).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

STAC = "https://data.geo.admin.ch/api/stac/v1"
KENDA = "ch.meteoschweiz.ogd-analysis-kenda-ch1"
ICON = "ch.meteoschweiz.ogd-forecasting-icon-ch1"
HF_REPO = "MeteoSwiss/Varda-single-1.0"

# ECMWF open-data facts, copied from anemoi_plugins_meteoswiss/inference/inputs/oper_ecmwf_opendata.py
FREQUENCY_H, STEP_H, MAX_LEAD_TIME_H, STORED_RUNS = 6, 3, 144, 12
# ecmwf/opendata/urls.py
ECMWF_URLS = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "ecmwf": "https://data.ecmwf.int/forecasts",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
}

USER_AGENT = "varda.cpp-opendata/1 (python-urllib)"


def utcnow() -> datetime:
    """Naive UTC now (the dates here are naive UTC, as in anemoi-inference)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class OpenDataError(RuntimeError):
    """A failure the user can act on (data not published, expired, network)."""


# ------------------------------------------------------------------------------------------------ HTTP


def _request(url: str, *, method: str = "GET", params: dict | None = None, body=None, headers: dict | None = None,
             timeout: float = 60, retries: int = 4):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    h = {"User-Agent": USER_AGENT}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h, method=method),
                                          timeout=timeout)
        except urllib.error.HTTPError as e:
            # 404/403/416 are answers, not glitches; retry only 429 and 5xx
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise OpenDataError(f"network error for {url}: {e}") from e
        time.sleep(delay)
        delay *= 2


def get_json(url: str, params: dict | None = None, timeout: float = 60):
    with _request(url, params=params, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, body, timeout: float = 60):
    with _request(url, method="POST", body=body, timeout=timeout) as r:
        return json.loads(r.read())


def head_status(url: str, timeout: float = 30) -> int:
    try:
        with _request(url, method="HEAD", timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def get_bytes(url: str, timeout: float = 120) -> bytes:
    with _request(url, timeout=timeout) as r:
        return r.read()


# ------------------------------------------------------------------------------------------------ files


def atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: str, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode())


def download(href: str, path: str) -> str:
    """Whole file to `path` (via `<path>.part`, then rename). An existing file is kept."""
    if os.path.exists(path):
        return f"keep {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part"
    with _request(href, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(tmp, path)
    return f"got  {path} ({os.path.getsize(path) / 1e6:.1f} MB)"


def download_ranges(href: str, ranges: list[tuple[int, int]], path: str) -> int:
    """The byte ranges (offset, length) of `href`, concatenated in the given order, to `path` (atomic).

    One HTTP Range request per range; each answer must be 206 with exactly `length` bytes. Returns the size."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part"
    total = 0
    with open(tmp, "wb") as f:
        for off, n in ranges:
            with _request(href, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=120) as r:
                if r.status != 206:
                    raise OpenDataError(f"{href}: range request answered {r.status}, not 206")
                b = r.read()
            if len(b) != n:
                raise OpenDataError(f"{href}: range {off}+{n} returned {len(b)} bytes")
            f.write(b)
            total += n
    os.replace(tmp, path)
    return total


# ------------------------------------------------------------------------------------------------ KENDA-CH1


def parse_date(s: str) -> datetime:
    """'2026-10-22T15', '2026-10-22T15:00[:00][Z]', '20261022T15', '2026102215' -> naive UTC datetime."""
    s = s.strip().removesuffix("Z")
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H", "%Y%m%dT%H%M%S", "%Y%m%d%H"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot read date {s!r}: use YYYY-MM-DDTHH (UTC), e.g. 2026-10-22T15")


def stac_items(valid_time: datetime) -> list[dict]:
    """Same query as oper_kenda_opendata._stac_items: every KENDA-CH1 item valid at `valid_time`."""
    url = f"{STAC}/collections/{KENDA}/items"
    params = {"datetime": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 100}
    out = []
    while url:
        page = get_json(url, params=params, timeout=30)
        out += page.get("features", [])
        nxt = [link["href"] for link in page.get("links", []) if link.get("rel") == "next"]
        url, params = (nxt[0], None) if nxt else (None, None)
    return out


def kenda_published_dates() -> set[datetime]:
    """The notebook's query: reference times of the published KENDA-CH1 P analyses (first page, limit 100)."""
    features = post_json(
        f"{STAC}/search",
        {
            "collections": [KENDA],
            "forecast:variable": "P",
            "forecast:horizon": "P0DT00H00M00S",
            "forecast:perturbed": False,
            "limit": 100,
        },
    )["features"]
    return {
        datetime.fromisoformat(f["properties"]["forecast:reference_datetime"].split("/")[0].removesuffix("Z"))
        for f in features
    }


def pick_latest_kenda(available: set[datetime] | None = None) -> datetime:
    """The notebook's rule: latest KENDA-CH1 analysis t (t.hour % 3 == 0) with t-6h published."""
    available = kenda_published_dates() if available is None else available
    ok = [t for t in available if t.hour % 3 == 0 and t - timedelta(hours=6) in available]
    if not ok:
        raise OpenDataError("no KENDA-CH1 analysis t with t-6h published was found (data.geo.admin.ch STAC)")
    return max(ok)


def icon_collection() -> dict:
    return get_json(f"{STAC}/collections/{ICON}", timeout=30)


def icon_constant_assets(coll: dict) -> dict[str, str]:
    """name -> href of the ICON-CH1 collection's GRIB assets (horizontal and vertical constants)."""
    return {name: a["href"] for name, a in coll.get("assets", {}).items() if a.get("type") == "application/grib"}


# ------------------------------------------------------------------------------------------------ ECMWF


def ecmwf_url(init: datetime, step: int, source: str = "google", ext: str = "grib2") -> str:
    """HOURLY_PATTERN of ecmwf.opendata for model ifs, resol 0p25, stream oper, type fc (runs from 2026-05-12 on:
    06/18z stay under oper)."""
    if init < datetime(2026, 5, 12):
        raise OpenDataError(f"run {init:%Y-%m-%dT%H}: dates before 2026-05-12 use another stream layout (not supported)")
    base = ECMWF_URLS[source]
    return f"{base}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/oper/{init:%Y%m%d%H%M%S}-{step}h-oper-fc.{ext}"


def ecmwf_latest_run(source: str = "google", now: datetime | None = None) -> datetime:
    """Client(source).latest(type="fc"): today 18z, then back in 6 h steps for 2 days, the first run whose step-0
    file answers HEAD 200."""
    now = utcnow() if now is None else now
    date = datetime(now.year, now.month, now.day, 18)
    stop = date - timedelta(days=2)
    while date > stop:
        if head_status(ecmwf_url(date, 0, source)) == 200:
            return date
        date -= timedelta(hours=6)
    raise OpenDataError(f"cannot establish the latest ECMWF open-data run on {source}")


def resolve(target: datetime, latest: datetime, allow_forecast_fallback: bool) -> tuple[datetime, int]:
    """Copy of OperEcmwfOpenDataInput._resolve_init_time_and_lead_hours (same checks; errors explained)."""
    hour = timedelta(hours=1)
    offset = latest - target
    if offset % hour != timedelta(0):
        raise OpenDataError(f"{target}: not a whole hour")
    hours_ahead = offset // hour
    n_back = -(-max(hours_ahead, 0) // FREQUENCY_H)
    if n_back > STORED_RUNS:
        raise OpenDataError(f"{target:%Y-%m-%dT%H}: older than the {STORED_RUNS} ECMWF runs kept in open data "
                            f"(latest run {latest:%Y-%m-%dT%H})")
    init = latest - timedelta(hours=FREQUENCY_H * n_back)
    lead = (target - init) // hour
    if not (0 <= lead <= MAX_LEAD_TIME_H and lead % STEP_H == 0):
        raise OpenDataError(f"{target:%Y-%m-%dT%H}: run {init:%Y-%m-%dT%H} step {lead} h is not in open data "
                            f"(steps are multiples of {STEP_H} h up to {MAX_LEAD_TIME_H} h)")
    if lead != 0 and not allow_forecast_fallback:
        raise OpenDataError(f"{target:%Y-%m-%dT%H}: no analysis, and the forecast fallback is off")
    return init, lead


def read_index(text: str) -> list[dict]:
    """An open-data `.index` file: one JSON object per line (keys of the message, `_offset`, `_length`)."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]
