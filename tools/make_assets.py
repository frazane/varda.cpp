"""The Varda assets bundle: everything derived once per model release, in one directory that
`varda-forecast --assets DIR` (and `./varda`) reads. Standard library only, Python >= 3.8 (`.venv/bin/python`).

    DIR/model.safetensors                 python/convert.py of the HF checkpoint (with the cutout masks)
    DIR/varda-io.json                     python/export_io_config.py (variables, GRIB output, templates index)
    DIR/regrid-ifs0p25-n320.safetensors   python/export_regrid_matrix.py (MIR's IFS 0.25 deg -> N320 linear matrix)
    DIR/templates/                        templates/ of the HF repo (ICON and IFS GRIB samples)
    DIR/definitions/                      ecCodes definitions of eccodes-cosmo-resources (KENDA-CH1 short names)
    DIR/manifest.json                     sha256 and size of every file, sources, checkpoint sha256, tool versions

Subcommands:
    make_assets.py check [--dest DIR]           verify DIR against its manifest (exit 1 with what is missing)
    make_assets.py link  [--dest DIR] [--copy]  assemble DIR from existing, verified files (hard links, or copies
                                                with --copy or across file systems); defaults: under $VARDA_SCRATCH
    make_assets.py build --hf HF [--dest DIR] [--grib F] [--py PY] [--ref-py PY]
                                                from scratch: run the converter (PY: torch env) and the two exporters
                                                (REF-PY: the reference env with anemoi-inference, mir, eccodes);
                                                use a compute node (the conversion reads the 1.5 GB checkpoint)

DIR defaults to $VARDA_ASSETS, else $VARDA_SCRATCH/assets/varda-single-1.0. A bundle is assembled under
DIR.partial-<pid> and renamed into place when complete and verified, so DIR is either absent or whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAME = "varda-single-1.0"
SCHEMA = "varda-assets/1"
FILES = ("model.safetensors", "varda-io.json", "regrid-ifs0p25-n320.safetensors")
TREES = ("templates", "definitions")
SCRATCH = os.environ.get("VARDA_SCRATCH", "")


def default_dest() -> str:
    if os.environ.get("VARDA_ASSETS"):
        return os.environ["VARDA_ASSETS"]
    if SCRATCH:
        return os.path.join(SCRATCH, "assets", NAME)
    return os.path.expanduser(f"~/.cache/varda/assets/{NAME}")


def fail(msg: str):
    print(f"make_assets: {msg}", file=sys.stderr)
    sys.exit(1)


_SHA: dict = {}


def sha256(path) -> str:
    """sha256 of a file, computed once per inode and version in this process (a hard link is its source's inode)."""
    st = os.stat(path)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while b := f.read(1 << 24):
                h.update(b)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def tree_files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def tree_digest(d: Path) -> dict:
    """sha256 over "relpath<TAB>sha256<LF>" of every file, sorted by path."""
    h, n, size = hashlib.sha256(), 0, 0
    for rel in tree_files(d):
        h.update(f"{rel}\t{sha256(d / rel)}\n".encode())
        n += 1
        size += (d / rel).stat().st_size
    return {"sha256": h.hexdigest(), "files": n, "bytes": size}


def safetensors_meta(path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


def place(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return "hard link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def describe(d: Path) -> dict:
    """Content checks and provenance read from the files themselves."""
    out = {}
    meta = safetensors_meta(d / "model.safetensors")
    if "cutout" not in meta:
        fail(f"{d / 'model.safetensors'} has no cutout masks: convert with the current python/convert.py")
    cfg = json.loads(meta["config"])
    src = cfg.get("source", {})
    out["checkpoint"] = {k: src.get(k) for k in ("sha256", "uuid", "size_bytes", "run_id", "checkpoint")}
    out["checkpoint"]["hf_repo"] = "MeteoSwiss/Varda-single-1.0"
    out["checkpoint"]["varda_format"] = meta.get("varda_format")
    out["cutout"] = json.loads(meta["cutout"])
    io = json.loads((d / "varda-io.json").read_text())
    if io.get("schema") != "varda-io/1":
        fail(f"{d / 'varda-io.json'}: schema {io.get('schema')!r}")
    if io["source"].get("checkpoint_uuid") != src.get("uuid"):
        fail(f"varda-io.json is for checkpoint {io['source'].get('checkpoint_uuid')}, model is {src.get('uuid')}")
    rg = json.loads(safetensors_meta(d / "regrid-ifs0p25-n320.safetensors")["regrid"])
    rev = json.loads((d / "templates" / "REVISION.json").read_text()) if (d / "templates/REVISION.json").exists() \
        else None
    if rev and io["source"].get("templates_revision") != rev:
        fail(f"templates/REVISION.json {rev} != varda-io.json's {io['source'].get('templates_revision')}")
    for name, want in io["source"].get("template_sha256", {}).items():
        if sha256(d / "templates" / name) != want:
            fail(f"templates/{name}: sha256 differs from varda-io.json's")
    out["tools"] = {
        "convert": f"python/convert.py (varda_format {meta.get('varda_format')}, config format "
                   f"{cfg.get('format_version')})",
        "export_io_config": f"anemoi-inference {io['source'].get('anemoi_inference')}",
        "export_regrid_matrix": rg.get("mir_version"),
        "regrid_mat_sha256": rg.get("mat_sha256"),
        "templates_revision": rev,
    }
    return out


def write_manifest(d: Path, sources: dict, how: str, extra: dict | None = None) -> dict:
    files = {}
    for f in FILES:
        files[f] = {"sha256": sha256(d / f), "bytes": (d / f).stat().st_size, "source": sources.get(f)}
    for rel in tree_files(d / "templates"):
        files[f"templates/{rel}"] = {"sha256": sha256(d / "templates" / rel), "bytes": (d / "templates" / rel).stat()
                                     .st_size, "source": sources.get("templates")}
    m = {
        "schema": SCHEMA,
        "name": NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": f"tools/make_assets.py {how}",
        "varda_cpp_commit": git_commit(),
        "files": files,
        "trees": {"definitions": dict(tree_digest(d / "definitions"), source=sources.get("definitions"))},
        **describe(d),
        **(extra or {}),
    }
    tmp = d / f"manifest.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, d / "manifest.json")
    return m


def check(d: Path, quiet: bool = False) -> list[str]:
    """Problems of bundle `d` against its manifest (empty: complete and intact)."""
    if not d.is_dir():
        return [f"{d}: no such directory"]
    if not (d / "manifest.json").exists():
        return [f"{d}/manifest.json: missing"] + [f"{d}/{f}: missing" for f in FILES + TREES
                                                 if not (d / f).exists()]
    m = json.loads((d / "manifest.json").read_text())
    bad = []
    for rel, e in m["files"].items():
        p = d / rel
        if not p.exists():
            bad.append(f"{p}: missing")
        elif p.stat().st_size != e["bytes"] or sha256(p) != e["sha256"]:
            bad.append(f"{p}: differs from the manifest")
    for rel, e in m.get("trees", {}).items():
        p = d / rel
        if not p.is_dir():
            bad.append(f"{p}/: missing")
        elif tree_digest(p)["sha256"] != e["sha256"]:
            bad.append(f"{p}/: differs from the manifest")
    if not bad and not quiet:
        print(f"{d}: complete ({len(m['files'])} files + definitions/ ({m['trees']['definitions']['files']} files), "
              f"checkpoint sha256 {m['checkpoint']['sha256'][:12]}..., created {m['created_utc']})")
    return bad


def finish(work: Path, dest: Path) -> None:
    if dest.exists():
        fail(f"{dest} appeared while building (another make_assets?); kept {work}")
    os.replace(work, dest)
    print(f"bundle ready: {dest}")


def cmd_check(a) -> None:
    bad = check(Path(a.dest))
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)


def cmd_link(a) -> None:
    dest = Path(a.dest)
    if dest.exists():
        bad = check(dest)
        if bad:
            fail(f"{dest} exists but is not intact:\n  " + "\n  ".join(bad) + "\nmove it away and rerun")
        return
    srcs = {
        "model.safetensors": Path(a.model),
        "varda-io.json": Path(a.io_config),
        "regrid-ifs0p25-n320.safetensors": Path(a.regrid_matrix),
        "templates": Path(a.templates),
        "definitions": Path(a.definitions),
    }
    for k, p in srcs.items():
        if not p.exists():
            fail(f"source for {k} not found: {p} (pass it explicitly, or build from scratch: make_assets.py build)")
    work = dest.with_name(f"{dest.name}.partial-{os.getpid()}")
    work.mkdir(parents=True)
    how = set()
    for f in FILES:
        how.add(place(srcs[f], work / f, a.copy))
    for t in TREES:
        for rel in tree_files(srcs[t]):
            how.add(place(srcs[t] / rel, work / t / rel, a.copy))
    # verify: every placed file equals its source, byte for byte (sha256)
    for f in FILES:
        if sha256(work / f) != sha256(srcs[f]):
            fail(f"{work / f}: sha256 differs from {srcs[f]}")
    for t in TREES:
        if tree_digest(work / t)["sha256"] != tree_digest(srcs[t])["sha256"]:
            fail(f"{work / t}: differs from {srcs[t]}")
    m = write_manifest(work, {k: str(p.resolve()) for k, p in srcs.items()}, f"link ({', '.join(sorted(how))})")
    print(f"verified: {len(m['files'])} files and definitions/ equal their sources (sha256)")
    finish(work, dest)


MIR_SNIPPET = r"""
import io, os, sys
import eccodes, mir
src = sys.argv[1]
with open(src, "rb") as f:
    while True:
        h = eccodes.codes_grib_new_from_file(f)
        if h is None:
            sys.exit(f"{src}: no message without bitmap")
        ok = not eccodes.codes_get(h, "bitmapPresent")
        msg = eccodes.codes_get_message(h)
        eccodes.codes_release(h)
        if ok:
            break
job = mir.Job(grid="N320", area=[89.785, 0.0, -89.785, 359.719], edition=2, packing="ccsds", accuracy=16,
              truncation="auto")
out = io.BytesIO()
job.execute(io.BytesIO(msg), out)
print(f"MIR regridded one message ({len(out.getvalue())} bytes out); cache {os.environ['MIR_CACHE_PATH']}")
"""


def run(cmd: list, env: dict | None = None) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode != 0:
        fail(f"step failed (exit {r.returncode}): {cmd[0]} {cmd[1:3]}")


def find_grib(a) -> Path:
    if a.grib:
        return Path(a.grib)
    roots = [os.environ.get("VARDA_INPUTS"), os.path.join(SCRATCH, "inputs") if SCRATCH else None]
    for r in filter(None, roots):
        for p in sorted(Path(r).glob("*/cache/ecmwf/files/*.grib2")):
            return p
    fail("no IFS 0.25 deg open-data GRIB file for the regrid geometry: pass --grib F (any file of an input cache, "
         "e.g. after `./varda fetch latest`: $VARDA_INPUTS/<date>/cache/ecmwf/files/*.grib2)")


def cmd_build(a) -> None:
    dest = Path(a.dest)
    if dest.exists():
        fail(f"{dest} exists; build into another --dest, or move it away")
    hf = Path(a.hf)
    ckpt = hf / "varda-forecaster-det-sgm" / "model.ckpt"
    for p in (ckpt, hf / "varda-forecaster-det-sgm/configs/inference/inference_opendata.yaml",
              hf / "varda-forecaster-det-sgm/metadata.yaml", hf / "templates" / "REVISION.json"):
        if not p.exists():
            fail(f"{p} not found: --hf must be a download of {NAME} with varda-forecaster-det-sgm/ and templates/ "
                 "(templates/REVISION.json: python/reference/realdata/fetch_realdata.py templates)")
    grib = find_grib(a)
    work = dest.with_name(f"{dest.name}.partial-{os.getpid()}")
    (work / "tmp").mkdir(parents=True)
    env = dict(os.environ)
    # 1. weights + config + cutout masks (torch env, restricted unpickler)
    run([a.py, "-I", REPO / "python/convert.py", ckpt, work / "model.safetensors"])
    # 2. I/O configuration (reference env: anemoi-inference reads the checkpoint metadata from the zip)
    run([a.ref_py, "-I", REPO / "python/export_io_config.py", "--hf", hf, "--out", work / "varda-io.json"])
    # 3. MIR's interpolation matrix: one MIR call on a real open-data message fills a private MIR cache with
    #    the .mat (the layout export_regrid_matrix.py reads from a run directory), then export it
    mirrun = work / "tmp" / "mir-run"
    # MIR writes <MIR_CACHE_PATH>/mir/weights/..., and does not create MIR_CACHE_PATH itself
    (mirrun / "cache" / "ecmwf").mkdir(parents=True)
    env_mir = dict(env, MIR_CACHE_PATH=str(mirrun / "cache" / "ecmwf"))
    env_mir.pop("ECCODES_DEFINITION_PATH", None)
    run([a.ref_py, "-I", "-c", MIR_SNIPPET, grib], env=env_mir)
    run([a.ref_py, "-I", REPO / "python/export_regrid_matrix.py", "--run", mirrun, "--grib", grib,
         "--out", work / "regrid-ifs0p25-n320.safetensors"])
    # 4. templates and COSMO definitions, copied
    shutil.copytree(hf / "templates", work / "templates")
    defs = subprocess.run([a.ref_py, "-I", "-c",
                           "import eccodes_cosmo_resources as e; print(e.get_definitions_path())"],
                          capture_output=True, text=True, check=True).stdout.strip()
    shutil.copytree(defs, work / "definitions")
    ver = subprocess.run([a.ref_py, "-I", "-c", "import importlib.metadata as m; "
                          "print(m.version('eccodes-cosmo-resources-python'))"],
                         capture_output=True, text=True).stdout.strip() or None
    shutil.rmtree(work / "tmp")  # created above by this run
    write_manifest(work, {"model.safetensors": str(ckpt), "varda-io.json": str(hf),
                          "regrid-ifs0p25-n320.safetensors": f"MIR on {grib}", "templates": str(hf / "templates"),
                          "definitions": defs}, "build",
                   {"definitions_package": ver})
    finish(work, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dest", default=default_dest())
    s = os.path.join(SCRATCH or "/nonexistent", "")
    ln = sub.add_parser("link")
    ln.add_argument("--dest", default=default_dest())
    ln.add_argument("--model", default=f"{s}converted/varda/model.safetensors")
    ln.add_argument("--io-config", default=f"{s}realdata/assets/varda-io.json")
    ln.add_argument("--regrid-matrix", default=f"{s}realdata/assets/regrid-ifs0p25-n320.safetensors")
    ln.add_argument("--templates", default=f"{s}hf/templates")
    ln.add_argument("--definitions", default=os.environ.get("VARDA_COSMO_DEFINITIONS")
                    or f"{s}deps/eccodes-cosmo-resources-2.44.0.1/definitions")
    ln.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    b = sub.add_parser("build")
    b.add_argument("--hf", required=True, help="HF download of Varda-single-1.0 (varda-forecaster-det-sgm/, templates/)")
    b.add_argument("--dest", default=default_dest())
    b.add_argument("--grib", help="an IFS 0.25 deg open-data GRIB2 file (geometry of the regrid source)")
    b.add_argument("--py", default=str(REPO / ".venv/bin/python"), help="python with torch + numpy (converter)")
    b.add_argument("--ref-py", default=os.path.join(SCRATCH, "ref-oper/.venv/bin/python") if SCRATCH else "python3",
                   help="python of the reference env (anemoi-inference, mir-python, eccodes, eccodes-cosmo-resources)")
    a = ap.parse_args()
    {"check": cmd_check, "link": cmd_link, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
