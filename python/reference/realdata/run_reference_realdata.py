"""Run the ORIGINAL Varda forecaster on real data exactly as anemoi-inference does with the Hugging Face config,
offline from the frozen cache, and dump every stage the C++ real-data I/O will be checked at.

Runs in the reference env built from the HF pyproject ($VARDA_SCRATCH/ref-oper/.venv; lock:
requirements-ref-oper.lock.txt next to this file). Never import this from src/ or tests/.

What runs: anemoi-inference's own `create_runner(RunConfiguration.load(<HF inference_opendata.yaml>, overrides))`
and `runner.execute()`, i.e. the `anemoi-inference run inference_opendata.yaml date=<t>` of the HF notebook, in a
run directory that holds symlinks to the HF checkpoint folder and templates/ (the config's relative paths
resolve exactly as in the HF repo root). The environment is the notebook's: ECCODES_DEFINITION_PATH from
eccodes_cosmo_resources, ECCODES_VERSION_CHECK_OFF=1, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and python -O (the notebook's PYTHONOPTIMIZE=1). Differences to
the notebook, all deliberate: the default (single-device) runner instead of `runner.parallel.cluster.manual=N`;
cache dirs and `precision` set by overrides; the network replaced by the frozen cache (below).

Offline replay (default; --online only for the cross-check):
  * oper-ecmwf-opendata: `_latest_published_run` returns the run pinned in pins.json at download time (it calls
    ecmwf.opendata.Client.latest() otherwise, which also selects the run of the `constant: true` source).
  * ecmwf.opendata.Client: the .index of a data file is read from the frozen cache (the real get_parts() then
    selects the byte ranges, preserve_request_order=True as in earthkit), and `_download` is redirected to
    file://<frozen full file> with the same parts (multiurl's PartFileDownloader), so earthkit's ecmwf-open-data
    source receives the same bytes it would have downloaded.
  * oper-kenda-opendata: `_stac_items` / `_collection_static_assets` return the STAC documents saved at download
    time; every asset is already in the frozen cache_dir, so nothing is fetched.
  * socket connect() raises: any other network access fails loudly.
In --online mode only the ECMWF pin is applied and the caches are fresh (under the run dir): run it with
--stub-model on a CPU node and compare the dumps with an offline stub run (compare_runs.py) to show the replay is
exact.

Dumps (<out>/dumps/*.safetensors, F32 as the runner holds them unless noted; metadata JSON in each file; index in
<out>/manifest.json):
  source.<purpose>.<src>     per cutout source (lam_0 = KENDA, global = ECMWF), the state its input returns
                             (after its pre_processors): field/<name> [n_dates, n_points], latitudes/longitudes F64
  regrid.<k>                 each MIR call of oper-ecmwf-opendata: out/<name>@<level> F64 [542080] (decoded MIR
                             output), with the request; the MIR input and output GRIB bytes next to it
                             (regrid.<k>.in.grib2 / .out.grib2)
  cutout.<purpose>           the cutout-combined state: field/<name> [2, G], latitudes, longitudes, mask/<src> (BOOL)
  input_state                the state given to TensorHandler.prepare_input_tensor, after the initial computed
                             forcings were added: field/<name> [2, G]
  input_tensor               prepare_input_tensor's result [2, V_in, G] (numpy layout of the runner)
  forcings.<kind>.<i>        each ComputedForcings.load_forcings_array call: values [vars, dates, G]
  step<s>.window / step<s>.y the tensor passed to model.predict_step [1, 2, G, V_in] (= golden in.x layout) and its
                             output (upcast to F32 if autocast made it fp16; dtype in metadata)
  state<k>.pre / state<k>.<i>.<processor>   the state before the top-level post_processors (k = 0: initial state)
                             and the fields each processor changed
  grib.<output>.<k>          the state each grib output encodes (after its own post_processors)
GRIB files: <out>/outputs/ (the config's two grib outputs).

Usage (GPU node, sbatch; see docs/realdata-reference.md):
    $PY -O -I python/reference/realdata/run_reference_realdata.py --date 2026-10-22T15 --precision checkpoint \
        --out $VARDA_SCRATCH/realdata/20261022T15/ref-fp16
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

# --- the notebook's environment, before anything imports eccodes -------------------------------------------
from eccodes_cosmo_resources import get_definitions_path  # noqa: E402

os.environ["ECCODES_DEFINITION_PATH"] = str(get_definitions_path())
os.environ["ECCODES_VERSION_CHECK_OFF"] = "1"
os.environ["ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER"] = "16"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
from safetensors_io import write_safetensors  # noqa: E402

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "$VARDA_SCRATCH")
HF = Path(VARDA_SCRATCH) / "hf"
HF_CONFIG = HF / "varda-forecaster-det-sgm" / "configs" / "inference" / "inference_opendata.yaml"
GOOGLE = "https://storage.googleapis.com/ecmwf-open-data/"


def sha256(path: str | Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def tt(a) -> torch.Tensor:
    """numpy/torch -> contiguous CPU torch tensor without changing the dtype."""
    if isinstance(a, torch.Tensor):
        return a.detach().to("cpu").contiguous()
    return torch.from_numpy(np.ascontiguousarray(a))


class Recorder:
    def __init__(self, out: Path):
        self.dir = out / "dumps"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, dict] = {}
        self.counters: dict[str, int] = {}
        self.nan: dict[str, dict] = {}

    def next(self, key: str) -> int:
        i = self.counters.get(key, 0)
        self.counters[key] = i + 1
        return i

    def save(self, name: str, tensors: dict, meta: dict) -> None:
        tensors = {k: tt(v) for k, v in tensors.items()}
        size = write_safetensors(self.dir / f"{name}.safetensors", tensors,
                                 {"varda_realdata": "1", "name": name, "meta": json.dumps(meta, default=str)})
        self.index[name] = {"file": f"dumps/{name}.safetensors", "bytes": size, "meta": meta,
                            "tensors": {k: [str(v.dtype).removeprefix("torch."), list(v.shape)] for k, v in tensors.items()}}
        print(f"[dump] {name}: {len(tensors)} tensors, {size / 1e6:.1f} MB", flush=True)

    def save_fields(self, name: str, state: dict, extra: dict | None = None, meta: dict | None = None) -> None:
        fields = state["fields"]
        tensors = {f"field/{k}": v for k, v in fields.items()}
        nan = {}
        for k, v in fields.items():
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            n = int(np.isnan(a).sum())
            if n:
                nan[k] = {"count": n, "per_date": np.isnan(a).reshape(a.shape[0], -1).sum(1).tolist()
                          if a.ndim == 2 else None}
        self.nan[name] = nan
        for key in ("latitudes", "longitudes"):
            if state.get(key) is not None:
                tensors[key] = np.asarray(state[key])
        tensors.update(extra or {})
        m = {"names": list(fields), "date": state.get("date"), "step": state.get("step"), "nan": nan}
        m.update(meta or {})
        for key in ("_geography",):
            if key in state:
                m[key] = state[key]
        self.save(name, tensors, m)


# ------------------------------------------------------------------------------------------------ offline replay


def block_network() -> None:
    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        raise RuntimeError(f"offline run: network access attempted ({address})")

    socket.socket.connect = connect
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"offline run: {a}"))


def pin_ecmwf_latest(pins: dict) -> None:
    from anemoi_plugins_meteoswiss.inference.inputs import oper_ecmwf_opendata as m

    latest = dt.datetime.fromisoformat(pins["ecmwf_latest_published_run"])
    calls = []

    def _latest_published_run(source: str = "ecmwf", **params):
        if source != pins["ecmwf_source"]:
            raise RuntimeError(f"pinned source {pins['ecmwf_source']}, asked {source}")
        calls.append(params)
        return latest

    m._latest_published_run = _latest_published_run
    print(f"[pin] oper-ecmwf-opendata latest published run = {latest}", flush=True)


def replay_ecmwf(files_dir: Path, log: list) -> None:
    from ecmwf.opendata import client as C

    class _IndexResponse:
        def __init__(self, path):
            self.path, self.status_code = path, 200

        def raise_for_status(self):
            pass

        def iter_lines(self):
            with open(self.path, "rb") as f:
                for line in f:
                    yield line.rstrip(b"\n")

    def local(url: str) -> Path:
        if not url.startswith(GOOGLE):
            raise RuntimeError(f"offline: unexpected ECMWF url {url}")
        p = files_dir / os.path.basename(url)
        if not p.exists():
            raise RuntimeError(f"offline: {p} not in the frozen cache")
        return p

    orig_init = C.Client.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        real_get = self.session.get

        def get(url, *aa, **kk):
            if url.endswith(".index"):
                return _IndexResponse(local(url))
            raise RuntimeError(f"offline: GET {url}")

        self.session.get = get
        self.session.head = lambda url, *aa, **kk: (_ for _ in ()).throw(RuntimeError(f"offline: HEAD {url}"))
        self._real_get = real_get

    orig_download = C.Client._download

    def _download(self, result):
        urls = []
        for u in result.urls:
            if isinstance(u, tuple):
                url, parts = u
                urls.append(("file://" + str(local(url)), parts))
                log.append({"url": url, "parts": [list(p) for p in parts], "target": result.target})
            else:
                urls.append("file://" + str(local(u)))
                log.append({"url": u, "parts": None, "target": result.target})
        result.urls = urls
        return orig_download(self, result)

    C.Client.__init__ = __init__
    C.Client._download = _download


def replay_kenda(kenda_dir: Path) -> None:
    from anemoi_plugins_meteoswiss.inference.inputs import oper_kenda_opendata as m

    def _stac_items(valid_time, *, limit=100):
        p = kenda_dir / "stac" / f"{valid_time.strftime('%Y%m%dT%H%M%S')}.json"
        if not p.exists():
            raise RuntimeError(f"offline: no frozen STAC items for {valid_time} ({p})")
        yield from json.loads(p.read_text())

    def _collection_static_assets():
        coll = json.loads((kenda_dir / "stac" / "collection_icon-ch1.json").read_text())
        return {n: a["href"] for n, a in coll.get("assets", {}).items() if a.get("type") == "application/grib"}

    m._stac_items = _stac_items
    m._collection_static_assets = _collection_static_assets


# ------------------------------------------------------------------------------------------------------- hooks


def install_mir_hook(rec: Recorder) -> None:
    import io

    import earthkit.data as ekd
    from anemoi.plugins.ecmwf.transform.regrid import backend

    orig = backend._mir_regrid_grib

    def _mir_regrid_grib(fields, grid, area, packing, accuracy):
        k = rec.next("regrid")
        buf = io.BytesIO()
        fields.to_target("file", buf)
        (rec.dir / f"regrid.{k}.in.grib2").write_bytes(buf.getvalue())
        t0 = time.perf_counter()
        out = orig(fields, grid, area, packing, accuracy)
        seconds = time.perf_counter() - t0
        obuf = io.BytesIO()
        out.to_target("file", obuf)
        (rec.dir / f"regrid.{k}.out.grib2").write_bytes(obuf.getvalue())
        keys = ("shortName", "paramId", "levtype", "level", "typeOfLevel", "dataDate", "dataTime", "step", "gridType",
                "numberOfDataPoints", "packingType", "bitsPerValue", "binaryScaleFactor", "decimalScaleFactor",
                "referenceValue", "missingValue", "bitmapPresent")

        def md(f):
            d = {}
            for key in keys:
                try:
                    d[key] = f.metadata(key)
                except Exception:  # noqa: BLE001
                    d[key] = None
            return d

        tensors, names = {}, []
        for f in out:
            m = md(f)
            name = f"{m['shortName']}@{m['level']}"
            names.append(name)
            tensors[f"out/{name}"] = f.to_numpy(dtype=np.float64, flatten=True)
        lat, lon = out[0].grid_points()
        tensors["out_latitudes"], tensors["out_longitudes"] = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        rec.save(f"regrid.{k}", tensors, {
            "grid": grid, "area": area, "packing": packing, "accuracy": accuracy, "seconds": seconds,
            "mir_job": {"grid": grid, "area": area, "edition": 2, "packing": packing, "accuracy": accuracy,
                        "truncation": "auto"},
            "in": [md(f) for f in fields], "out": [md(f) for f in out], "out_names": names,
            "in_grib": f"dumps/regrid.{k}.in.grib2", "out_grib": f"dumps/regrid.{k}.out.grib2"})
        return out

    backend._mir_regrid_grib = _mir_regrid_grib


def install_runner_hooks(runner, rec: Recorder, stub_model: bool) -> dict:
    from anemoi.inference.inputs.cutout import Cutout
    from anemoi.inference.outputs.grib import BaseGribOutput

    info: dict = {"predict": []}

    # 1-2. per-source states and the cutout-combined state, for every input purpose
    for purpose, inputs in (("prognostics", runner.prognostics_inputs),
                            ("constant_forcings", runner.constant_forcings_inputs),
                            ("dynamic_forcings", runner.dynamic_forcings_inputs),
                            ("boundary_forcings", runner.boundary_forcings_inputs)):
        for ds, inp in inputs.items():
            if not isinstance(inp, Cutout):
                continue
            for src, sub in inp.sources.items():
                orig = sub.create_input_state

                def create_input_state(*a, _o=orig, _p=purpose, _s=src, _sub=sub, **k):
                    st = _o(*a, **k)
                    rec.save_fields(f"source.{_p}.{_s}", st, meta={"input": repr(_sub), "dates": k.get("dates")})
                    return st

                sub.create_input_state = create_input_state
            orig_c = inp.create_input_state

            def create_cutout_state(*a, _o=orig_c, _p=purpose, _inp=inp, **k):
                st = _o(*a, **k)
                masks = {f"mask/{s}": np.asarray(m) for s, m in st.get("_mask", {}).items()}
                rec.save_fields(f"cutout.{_p}", st, extra=masks, meta={"dates": k.get("dates"),
                                                                        "sources": list(_inp.sources)})
                return st

            inp.create_input_state = create_cutout_state

    # 3. input state / tensor and computed forcings
    for ds, h in runner.tensor_handlers.items():
        orig_p = h.prepare_input_tensor

        def prepare_input_tensor(input_state, *a, _o=orig_p, _h=h, **k):
            x = _o(input_state, *a, **k)
            rec.save_fields("input_state", input_state, meta={"lagged": [str(d) for d in _h.metadata.lagged]})
            names = [None] * x.shape[1]
            for v, i in _h.metadata.variable_to_input_tensor_index.items():
                names[i] = v
            rec.save("input_tensor", {"x": x}, {"layout": "[multi_step_input, variables, values]", "in_vars": names,
                                                "nan_count": int(np.isnan(x).sum())})
            return x

        h.prepare_input_tensor = prepare_input_tensor
        for kind, providers in (("constant", h.constant_forcings_providers), ("dynamic", h.dynamic_forcings_providers)):
            for p in providers:
                orig_l = p.load_forcings_array

                def load_forcings_array(dates, state, _o=orig_l, _p=p, _kind=kind):
                    a = _o(dates, state)
                    i = rec.next(f"forcings.{_kind}")
                    rec.save(f"forcings.{_kind}.{i}", {"values": np.asarray(a)},
                             {"provider": repr(_p), "variables": list(_p.variables), "mask": [int(m) for m in _p.mask],
                              "dates": [d.isoformat() for d in dates], "layout": "[variables, dates, values]"})
                    return a

                p.load_forcings_array = load_forcings_array

    # 4. the window each predict_step sees and its output
    cls = type(runner)
    orig_predict = cls.predict_step
    out_vars = None

    def predict_step(self, model, input_tensors_torch, **kwargs):
        s = kwargs.get("fcstep")
        for ds, x in input_tensors_torch.items():
            rec.save(f"step{s}.window", {"x": x.float()}, {"layout": "[batch, multi_step_input, values, variables]",
                                                          "dtype": str(x.dtype), "kwargs": kwargs, "dataset": ds})
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = orig_predict(self, model, input_tensors_torch, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        for ds, t in y.items():
            h = self.tensor_handlers[ds]
            ov = [h.metadata.output_tensor_index_to_variable[j] for j in range(t.shape[-1])]
            rec.save(f"step{s}.y", {"y": t.float()}, {"layout": "[batch, (time), ensemble, values, variables]",
                                                      "dtype": str(t.dtype), "out_vars": ov, "seconds": seconds,
                                                      "nan_count": int(torch.isnan(t).sum())})
        info["predict"].append({"step": s, "seconds": seconds, "autocast": str(torch.get_autocast_dtype(self.device.type))
                                if torch.is_autocast_enabled(self.device.type) else None})
        return y

    cls.predict_step = predict_step

    # 5. top-level post_processors (accumulate_from_start_of_forecast, rescale)
    for ds, procs in runner.post_processors.items():
        for i, p in enumerate(procs):
            orig_pp = p.process

            def process(state, _o=orig_pp, _i=i, _p=p, _ds=ds):
                if _i == 0:
                    k = rec.next(f"state.{_ds}")
                    rec.save_fields(f"state{k}.pre", state)
                before = {n: np.array(v, copy=True) for n, v in state["fields"].items()}
                out = _o(state)
                changed = {n: v for n, v in out["fields"].items()
                           if n not in before or not np.array_equal(np.asarray(v), before[n], equal_nan=True)}
                k = rec.counters[f"state.{_ds}"] - 1
                rec.save_fields(f"state{k}.{_i}.{type(_p).__name__}", {**out, "fields": changed},
                                meta={"processor": repr(_p), "added": [n for n in out["fields"] if n not in before],
                                      "removed": [n for n in before if n not in out["fields"]]})
                return out

            p.process = process

    # 6. what each grib output encodes
    orig_write = BaseGribOutput.write_step

    def write_step(self, state):
        path = str(getattr(self, "out", ""))
        name = "global" if "global-" in path else "lam"
        k = rec.next(f"grib.{name}")
        rec.save_fields(f"grib.{name}.{k}", state, meta={"output": repr(self), "path": path})
        return orig_write(self, state)

    BaseGribOutput.write_step = write_step

    if stub_model:
        class Stub(torch.nn.Module):
            def predict_step(self, xs, **kwargs):  # multi-dataset checkpoint: {dataset: [1, 2, G, V_in]}
                out = {}
                for ds, x in xs.items():
                    n_out = len(runner.tensor_handlers[ds].metadata.output_tensor_index_to_variable)
                    out[ds] = torch.zeros(x.shape[0], 1, 1, x.shape[2], n_out, dtype=torch.float32, device=x.device)
                return out

        runner.__dict__["model"] = Stub()
    return info


# --------------------------------------------------------------------------------------------------------- main


def cpu_info() -> dict:
    """numpy's SIMD dispatch decides np.log(float32) (docs/spec-io.md 2.5): record it with the CPU model."""
    from numpy._core import _multiarray_umath as mu

    model = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    feats = getattr(mu, "__cpu_features__", {})
    return {"model": model, "nproc": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
            "numpy_cpu_baseline": list(getattr(mu, "__cpu_baseline__", [])),
            "numpy_cpu_dispatch": list(getattr(mu, "__cpu_dispatch__", [])),
            "numpy_cpu_features_on": sorted(k for k, v in feats.items() if v)}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for p in ("torch", "anemoi-inference", "anemoi-models", "anemoi-utils", "anemoi-transform", "anemoi-datasets",
              "anemoi-plugins-meteoswiss", "anemoi-plugins-ecmwf-inference", "earthkit-data", "earthkit-regrid",
              "eccodes", "eccodeslib", "eccodes-cosmo-resources-python", "mir-python", "mirlib", "ecmwf-opendata",
              "multiurl", "numpy", "triton"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="initial date t (newest input slot), e.g. 2026-10-22T15")
    ap.add_argument("--precision", default="checkpoint", help="'checkpoint' (runner default) or a runner value: 32, 16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lead-time", default="1d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="frozen cache root (default $VARDA_SCRATCH/realdata/<date>)")
    ap.add_argument("--online", action="store_true", help="network on, fresh caches; only the ECMWF run is pinned")
    ap.add_argument("--stub-model", action="store_true", help="zeros instead of the network (I/O cross-checks)")
    ap.add_argument("--no-dumps", action="store_true",
                    help="no hooks, no dumps: the plain runner on the frozen inputs (timing baseline)")
    ap.add_argument("--ecmwf-cache", default=None,
                    help="earthkit/MIR cache directory to use (and keep) instead of <out>/cache/ecmwf, e.g. a warm one")
    args = ap.parse_args()

    date = dt.datetime.fromisoformat(args.date)
    root = Path(args.cache or Path(VARDA_SCRATCH) / "realdata" / date.strftime("%Y%m%dT%H"))
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads((root / "pins.json").read_text())
    if dt.datetime.fromisoformat(pins["date"]) != date:
        sys.exit(f"pins.json is for {pins['date']}, not {date}")

    # run dir = a stand-in for the HF repo root: the config's relative paths resolve as there
    for name, target in (("varda-forecaster-det-sgm", HF / "varda-forecaster-det-sgm"), ("templates", HF / "templates")):
        os.symlink(target, out / name)
    os.chdir(out)

    cfg = yaml.safe_load(HF_CONFIG.read_text())
    overrides: dict = {"date": date.isoformat(), "lead_time": args.lead_time, "device": args.device}
    if args.precision != "checkpoint":
        overrides["precision"] = args.precision
    if args.online:
        kenda_cache, ecmwf_cache = out / "cache" / "kenda", out / "cache" / "ecmwf"
    else:
        kenda_cache, ecmwf_cache = root / "cache" / "kenda", out / "cache" / "ecmwf"  # ecmwf: earthkit + MIR caches
    if args.ecmwf_cache:
        ecmwf_cache = Path(args.ecmwf_cache).resolve()
    for section in ("constant_forcings", "input"):
        for src in cfg[section]["cutout"]:
            (name, sub), = src.items()
            (kind, opts), = sub.items()
            opts["cache_dir"] = str(kenda_cache if kind == "oper_kenda_opendata" else ecmwf_cache)
    (out / "config.yaml").write_text(yaml.safe_dump({"base": str(HF_CONFIG), "resolved": cfg,
                                                     "overrides": overrides}, sort_keys=False))

    pin_ecmwf_latest(pins)
    fetch_log: list = []
    if not args.online:
        replay_kenda(root / "cache" / "kenda")
        replay_ecmwf(root / "cache" / "ecmwf" / "files", fetch_log)

    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners import create_runner

    rec = Recorder(out)
    if not args.no_dumps:
        install_mir_hook(rec)
    if not args.online:
        block_network()

    t0 = time.perf_counter()
    config = RunConfiguration.load(cfg, [overrides])
    runner = create_runner(config)
    info = {"predict": None} if args.no_dumps else install_runner_hooks(runner, rec, args.stub_model)
    t1 = time.perf_counter()
    runner.execute()
    t2 = time.perf_counter()

    gribs = sorted(p for p in (out / "outputs").glob("*.grib"))
    manifest = {
        "kind": "varda realdata reference run",
        "date": date.isoformat(),
        "argv": sys.argv,
        "online": args.online,
        "stub_model": args.stub_model,
        "no_dumps": args.no_dumps,
        "ecmwf_cache": str(ecmwf_cache),
        "precision_arg": args.precision,
        "runner_precision": runner.precision,
        "checkpoint_precision": runner.checkpoint.precision,
        "autocast": str(runner.autocast),
        "device": str(runner.device),
        "device_name": torch.cuda.get_device_name(0) if runner.device.type == "cuda" else platform.processor(),
        "runner_class": type(runner).__mro__[1].__name__,
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "python_flags_optimize": sys.flags.optimize,
        "env": {k: os.environ.get(k) for k in ("ECCODES_DEFINITION_PATH", "ECCODES_VERSION_CHECK_OFF",
                                                 "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", "PYTORCH_CUDA_ALLOC_CONF",
                                                 "MIR_CACHE_PATH", "SLURM_JOB_ID")},
        "host": socket.gethostname(),
        "cpu": cpu_info(),
        "versions": versions(),
        "pins": pins,
        "frozen_cache": str(root),
        "ecmwf_replayed_downloads": fetch_log,
        "seconds": {"setup": t1 - t0, "execute": t2 - t1},
        "predict": info["predict"],
        "nan": rec.nan,
        "dumps": rec.index,
        "grib_outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in gribs},
    }
    atomic_json(out / "manifest.json", manifest)
    print(f"done: {len(gribs)} grib files, {len(rec.index)} dumps, execute {t2 - t1:.1f} s", flush=True)


if __name__ == "__main__":
    main()
