"""Export the real-data I/O configuration of the Varda forecaster as one JSON file.

The C++ engine must not parse YAML or read the checkpoint pickle. This tool reads, once:
  * the checkpoint metadata, from the zip (anemoi-inference's own `Checkpoint` class; nothing is
    unpickled) with the Hugging Face `patch_metadata` applied;
  * the Hugging Face inference config (`configs/inference/inference_opendata.yaml`);
  * the two GRIB template index files and the template files (sha256 only),
and writes `varda-io.json` (schema: docs/io-output.md). It is strict, like python/convert.py: anything it
does not understand is an error, never a silent default.

Run in the reference environment (it has anemoi-inference, yaml, eccodes):
    $VARDA_SCRATCH/ref-oper/.venv/bin/python -I python/export_io_config.py \
        [--hf $VARDA_SCRATCH/hf] [--out $VARDA_SCRATCH/realdata/assets/varda-io.json]
Tools only: never imported by src/, apps/ or tests/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import yaml

SCHEMA = "varda-io/1"
VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "$VARDA_SCRATCH")

# Top-level keys of the inference config this tool understands.
KNOWN_TOP = {
    "checkpoint", "lead_time", "constant_forcings", "input", "post_processors", "output",
    "verbosity", "allow_nans", "write_initial_state", "patch_metadata",
}
# mars keys a model variable may carry (anything else would change the encoded keys, see
# anemoi/inference/grib/encoding.py:grib_keys, and is not implemented in the C++).
MARS_KEYS = {"date", "time", "step", "param", "levtype", "levelist"}
# keys a template-index match may use (the C++ builds exactly these lookup entries).
MATCH_KEYS = {"levtype", "param", "levelist", "name", "time_processing"}
GRIB_OUTPUT_KEYS = {"path", "encoding", "templates", "post_processors", "modifiers"}


def fail(msg: str):
    raise SystemExit(f"export_io_config: {msg}")


def sha256(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def hours(td, what: str) -> int:
    s = td.total_seconds() / 3600
    if s != int(s):
        fail(f"{what}: {td} is not a whole number of hours")
    return int(s)


def plain(v, what: str):
    """JSON-able scalar or None; fails on anything else (bool is kept as bool)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    fail(f"{what}: unsupported value {v!r}")


def convert_post_processor(p, where: str) -> dict:
    """One entry of a `post_processors` list."""
    if isinstance(p, str):
        fail(f"{where}: bare post-processor {p!r} not understood")
    if not isinstance(p, dict) or len(p) != 1:
        fail(f"{where}: post-processor must be a one-key dict, got {p!r}")
    (kind, arg), = p.items()
    if kind == "accumulate_from_start_of_forecast":
        extra = set(arg) - {"accumulations", "allow_negative"}
        if extra:
            fail(f"{where}: accumulate: unknown keys {sorted(extra)}")
        if not isinstance(arg.get("accumulations"), list):
            fail(f"{where}: accumulate needs an explicit accumulations list")
        return {"type": "accumulate_from_start_of_forecast", "accumulations": list(arg["accumulations"]),
                "allow_negative": bool(arg.get("allow_negative", False))}
    if kind == "forward_transform_filter":
        if not (isinstance(arg, dict) and len(arg) == 1 and "rescale" in arg):
            fail(f"{where}: forward_transform_filter: only `rescale` is implemented, got {arg!r}")
        r = arg["rescale"]
        if set(r) != {"scale", "offset", "param"}:
            fail(f"{where}: rescale keys must be scale, offset, param, got {sorted(r)}")
        for k in ("scale", "offset"):
            if not isinstance(r[k], (int, float)) or isinstance(r[k], bool):
                fail(f"{where}: rescale {k} must be a number")
        return {"type": "rescale", "scale": r["scale"], "offset": r["offset"], "param": str(r["param"])}
    if kind == "extract_from_state":
        if not isinstance(arg, str):
            fail(f"{where}: extract_from_state expects the source name")
        return {"type": "extract_from_state", "source": arg}
    if kind == "extract_mask":
        extra = set(arg) - {"mask", "as_slice", "inverse"}
        if extra or "mask" not in arg:
            fail(f"{where}: extract_mask keys {sorted(arg)}")
        return {"type": "extract_mask", "mask": arg["mask"], "as_slice": bool(arg.get("as_slice", False)),
                "inverse": bool(arg.get("inverse", False))}
    if kind == "assign_mask":
        if isinstance(arg, str):
            arg = {"mask": arg}
        extra = set(arg) - {"mask"}
        if extra or "mask" not in arg:
            fail(f"{where}: assign_mask supports `mask` only (fill value NaN), got {sorted(arg)}")
        return {"type": "assign_mask", "mask": arg["mask"], "fill_value": "nan"}
    fail(f"{where}: post-processor {kind!r} not implemented")


def convert_modifiers(mods, where: str) -> list:
    if mods is None:
        return []
    out = []
    for m in mods:
        if not (isinstance(m, dict) and len(m) == 1 and "patches" in m):
            fail(f"{where}: only the `patches` modifier is implemented, got {m!r}")
        patches = []
        for patch in m["patches"]:
            if not (isinstance(patch, dict) and list(patch) == ["variable"]):
                fail(f"{where}: patches entries must be {{variable: {{...}}}}")
            rules = []
            for regex, kv in patch["variable"].items():
                if not isinstance(kv, dict):
                    fail(f"{where}: patch {regex!r}: expected a dict")
                for k, v in kv.items():
                    if isinstance(v, str) and v.startswith("\\"):
                        fail(f"{where}: patch {regex!r}: back-references ({v!r}) not implemented")
                    plain(v, f"{where}: patch {regex!r}.{k}")
                    if v is None or isinstance(v, (bool, float)):
                        fail(f"{where}: patch {regex!r}.{k}: only int and str values are implemented")
                rules.append({"regex": str(regex), "keys": dict(kv)})
            patches.append({"variable": rules})
        # PatchOutput keeps only the LAST patch (anemoi/inference/outputs/grib.py:107-115, the
        # `append` is outside the loop). Record all, and say which one is used.
        out.append({"type": "patches", "patches": patches, "effective_patch": len(patches) - 1})
    return out


def default_format_path(path: str) -> str:
    """The `format_dataset_name` decorator (anemoi/inference/decorators.py:270-279): every
    `{key}` is replaced by the literal "{key}" and the format spec is applied to that string
    (width/zero-fill on a longer string does nothing)."""
    import string

    class DefaultFormat(dict):
        def __missing__(self, key):
            return f"{{{key}}}"

    return path.format_map(DefaultFormat(dataset="data", dataset_name="data"))


def load_index(hf: Path, rel: str) -> tuple[list, dict]:
    p = hf / rel
    if not p.is_file():
        fail(f"template index {p} not found")
    idx = yaml.safe_load(p.read_text())
    if not isinstance(idx, list):
        fail(f"{p}: index must be a list")
    entries, files = [], {}
    for e in idx:
        if not (isinstance(e, list) and len(e) == 2 and isinstance(e[0], dict) and isinstance(e[1], str)):
            fail(f"{p}: bad index element {e!r}")
        match, grib = e
        bad = set(match) - MATCH_KEYS
        if bad:
            fail(f"{p}: match keys {sorted(bad)} not implemented")
        if "{" in grib or not grib.startswith("templates/"):
            fail(f"{p}: template path {grib!r} must be templates/<file> without placeholders")
        f = hf / grib
        if not f.is_file():
            fail(f"template file {f} not found")
        norm = {k: (list(v) if isinstance(v, list) else v) for k, v in match.items()}
        entries.append({"match": norm, "file": grib[len("templates/"):]})
        files[grib[len("templates/"):]] = sha256(f)
    return entries, files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf", default=f"{VARDA_SCRATCH}/hf")
    ap.add_argument("--out", default=f"{VARDA_SCRATCH}/realdata/assets/varda-io.json")
    a = ap.parse_args()
    hf = Path(a.hf)
    cfg_path = hf / "varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml"
    patch_path = hf / "varda-forecaster-det-sgm/metadata.yaml"
    ckpt_path = hf / "varda-forecaster-det-sgm/model.ckpt"
    for p in (cfg_path, patch_path, ckpt_path, hf / "templates/REVISION.json"):
        if not p.is_file():
            fail(f"{p} not found")

    cfg = yaml.safe_load(cfg_path.read_text())
    patch = yaml.safe_load(patch_path.read_text())
    if set(cfg) - KNOWN_TOP:
        fail(f"inference config has unknown top-level keys {sorted(set(cfg) - KNOWN_TOP)}")
    if cfg.get("patch_metadata") != "varda-forecaster-det-sgm/metadata.yaml":
        fail("patch_metadata is not the expected file")
    if cfg.get("allow_nans") is not True or cfg.get("write_initial_state") is not True:
        fail("this exporter encodes allow_nans: true and write_initial_state: true (NaN bitmap path, step 0)")
    lead = cfg["lead_time"]
    if lead != "1d":
        fail(f"lead_time {lead!r} not understood")

    # ---- checkpoint metadata, patched, via anemoi-inference's own classes (zip read only) ----
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    os.environ.setdefault("ECCODES_VERSION_CHECK_OFF", "1")
    from anemoi.inference.checkpoint import Checkpoint
    from anemoi.utils.dates import as_timedelta

    ck = Checkpoint(str(ckpt_path), patch_metadata=patch)
    m = ck._metadata
    vm = m.variables_metadata
    categories = m.variable_categories()
    model_vars = list(m.variables)
    out_index = m.output_tensor_index_to_variable
    output_names = [out_index[i] for i in sorted(out_index)]
    typed = m.typed_variables

    variables = []
    for name, md in vm.items():
        mars = dict(md.get("mars", {}))
        bad = set(mars) - MARS_KEYS
        if bad:
            fail(f"variable {name}: mars keys {sorted(bad)} not implemented")
        for k, v in mars.items():
            plain(v, f"variable {name}.mars.{k}")
        process = md.get("process")
        if process not in (None, "accumulation"):
            fail(f"variable {name}: process {process!r} not implemented")
        period = None
        if process is not None:
            per = md.get("period")
            if not (isinstance(per, list) and len(per) == 2):
                fail(f"variable {name}: accumulation without a [start, end] period")
            period = hours(as_timedelta(per[1]) - as_timedelta(per[0]), f"{name}.period")
            if typed[name].period is None:
                fail(f"variable {name}: reference sees no period")
        extra = set(md) - {"mars", "constant_in_time", "computed_forcing", "process", "period"}
        if extra:
            fail(f"variable {name}: metadata keys {sorted(extra)} not understood")
        variables.append({
            "name": name,
            "in_model": name in model_vars,
            "mars": mars,
            "constant_in_time": bool(md.get("constant_in_time", False)),
            "computed_forcing": bool(md.get("computed_forcing", False)),
            "process": process,
            "period_hours": period,
            "categories": list(categories[name]) if name in model_vars else None,
        })
    for name in output_names:
        if name not in vm or "mars" not in vm[name]:
            fail(f"model output {name} has no patched mars entry")

    prog = m.select_variables(include=["prognostic"], exclude=["computed", "diagnostic", "forcing"],
                              has_mars_requests=True)
    consts = m.select_variables(include=["constant+forcing"], exclude=["computed", "diagnostic", "prognostic"],
                                has_mars_requests=True)
    if len(prog) != 66 or consts != ["z", "lsm", "slor", "sdor"]:
        fail(f"unexpected initial-state variable sets: {len(prog)} prognostics, constants {consts}")
    if m.grid != "N320":
        fail(f"unexpected grid {m.grid!r}")

    # ---- KENDA namer (identical in constant_forcings and input) ----
    def kenda_namer(section: str) -> dict:
        lam = [d for d in cfg[section]["cutout"] if "lam_0" in d]
        if len(lam) != 1:
            fail(f"{section}: expected exactly one lam_0 source")
        src = lam[0]["lam_0"]
        if list(src) != ["oper_kenda_opendata"]:
            fail(f"{section}.lam_0: expected oper_kenda_opendata, got {list(src)}")
        return src["oper_kenda_opendata"]["namer"]

    ns = kenda_namer("input")
    if kenda_namer("constant_forcings") != ns:
        fail("the two KENDA namers differ")
    if list(ns) != ["rules"]:
        fail(f"namer keys {list(ns)} not understood")
    rules = []
    for r in ns["rules"]:
        if not (isinstance(r, list) and len(r) == 2 and isinstance(r[0], dict) and isinstance(r[1], str)):
            fail(f"namer rule {r!r} not understood")
        if set(r[0]) != {"shortName"}:
            fail(f"namer rule {r!r}: only shortName matches are implemented")
        rules.append({"match": {"shortName": r[0]["shortName"]}, "name": r[1]})

    # ---- outputs ----
    if cfg.get("post_processors") is None:
        fail("no top-level post_processors")
    top_pp = [convert_post_processor(p, "post_processors") for p in cfg["post_processors"]]
    out_cfg = cfg["output"]
    if list(out_cfg) != ["tee"] or list(out_cfg["tee"]) != ["outputs"]:
        fail("output must be tee: {outputs: [...]}")
    index_cache: dict[str, tuple] = {}
    template_files: dict[str, str] = {}
    outputs, names_seen = [], []
    for k, o in enumerate(out_cfg["tee"]["outputs"]):
        if not (isinstance(o, dict) and len(o) == 1 and "grib" in o):
            fail(f"output {k}: only `grib` is implemented")
        g = o["grib"]
        bad = set(g) - GRIB_OUTPUT_KEYS
        if bad:
            fail(f"output {k}: keys {sorted(bad)} not implemented")
        enc = g.get("encoding", {})
        for key, v in enc.items():
            if not isinstance(v, int) or isinstance(v, bool):
                fail(f"output {k}: encoding {key}={v!r}: only int values are implemented")
        tpl = g["templates"]
        if not (isinstance(tpl, dict) and list(tpl) == ["samples"] and isinstance(tpl["samples"], str)):
            fail(f"output {k}: templates must be {{samples: <index file>}}")
        entries, files = load_index(hf, tpl["samples"])
        template_files.update(files)
        pps = [convert_post_processor(p, f"output {k}") for p in g.get("post_processors", [])]
        is_lam = any(p["type"] == "extract_from_state" for p in pps)
        name = "lam" if is_lam else "global"
        names_seen.append(name)
        outputs.append({
            "name": name,
            "kind": "grib",
            "path": g["path"],
            "path_effective": default_format_path(g["path"]),
            "encoding": dict(enc),
            "template_index": {"source": tpl["samples"], "entries": entries},
            "post_processors": pps,
            "modifiers": convert_modifiers(g.get("modifiers"), f"output {k}"),
            "check_encoding": True,
            "missing_value": -9999,
        })
    if names_seen != ["lam", "global"]:
        fail(f"expected outputs [lam, global], got {names_seen}")

    rev = json.loads((hf / "templates/REVISION.json").read_text())
    doc = {
        "schema": SCHEMA,
        "source": {
            "checkpoint": "varda-forecaster-det-sgm/model.ckpt",
            "checkpoint_uuid": m._metadata.get("uuid"),
            "inference_config": "varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml",
            "inference_config_sha256": sha256(cfg_path),
            "patch_metadata_sha256": sha256(patch_path),
            "templates_revision": rev,
            "template_sha256": template_files,
            "anemoi_inference": __import__("importlib.metadata").metadata.version("anemoi-inference"),
        },
        "grid": m.grid,
        "area": list(m.area),
        "number_of_grid_points": int(m.number_of_grid_points),
        "timestep_hours": hours(m.timestep, "timestep"),
        "lead_time_hours": 24,
        "write_initial_state": True,
        "allow_nans": True,
        "cutout_sources": ["lam_0", "global"],
        "variables": variables,
        "model_variables": model_vars,
        "output_variables": output_names,
        "initial_state_fields": prog + consts,
        "kenda_namer_rules": rules,
        "post_processors": top_pp,
        "outputs": outputs,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out}: {len(variables)} variables ({len(model_vars)} in the model, {len(output_names)} outputs), "
          f"{len(prog + consts)} initial-state fields, {len(rules)} namer rules, "
          f"{sum(len(o['template_index']['entries']) for o in outputs)} template index entries")


if __name__ == "__main__":
    main()
