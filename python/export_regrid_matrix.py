"""Export MIR's IFS 0.25 deg -> N320 interpolation matrix (the one the real-data reference run used) for the C++.

Runs in the reference env ($VARDA_SCRATCH/ref-oper/.venv: mir-python, eccodes, eckit). Tools only, never the engine.

Source of the matrix: MIR's own weight cache file (`<run>/cache/ecmwf/mir/weights/<ver>/linear/LL-.../N320-....mat`,
an eckit Stream: rows, cols, nnz, little-endian flag, index/scalar/size item sizes, then the outer, inner and data
blobs). It is read raw (no scipy, no sort): `regrid.indices` keeps MIR's stored column order within each row.

Output ($VARDA_SCRATCH/realdata/assets/regrid-ifs0p25-n320.safetensors, written atomically):
    regrid.indptr  I64 [n_target + 1]     row pointers
    regrid.indices I32 [nnz]              source columns, MIR's stored order (NOT sorted)
    regrid.weights F64 [nnz]
    __metadata__["regrid"] = JSON: method, mir version, n_source, n_target, source geometry (keys of a real
        open-data message), target grid and area, sha256 of the .mat file(s) found in the run's cache,
        verification results (when --verify).

--verify <run dir>: for every MIR call the run dumped (dumps/regrid.<k>.{in,out}.grib2):
  1. y = W x in double, each row summed sequentially in stored order (s = 0; s += w_j * x_j), x =
     codes_get_double_array of the input message;
  2. MIR itself on the same input bytes with 64-bit IEEE output packing (its values before the 16-bit encode):
     compared with y (max_abs, number of points that differ);
  3. MIR on the same bytes with the run's job (ccsds, 16 bits), ECCODES_DEFINITION_PATH unset as in the run:
     compared byte for byte with the run's MIR output;
  4. y set into a clone of the run's output message (ccsds 16 bits) through ecCodes and decoded: compared with
     the run's decoded output (bitwise?).
Also checks that the MIR cache file this process used is byte-identical to the run's.

Usage (compute node):
    $PY -I python/export_regrid_matrix.py --run $VARDA_SCRATCH/realdata/20261022T15/ref-fp16 \
        --grib $VARDA_SCRATCH/realdata/20261022T15/cache/ecmwf/files/20261022120000-3h-oper-fc.grib2 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safetensors_io import write_safetensors  # noqa: E402

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH", "")  # only for the default of --out
OUT = Path(VARDA_SCRATCH) / "realdata" / "assets" / "regrid-ifs0p25-n320.safetensors" if VARDA_SCRATCH else None
GEOMETRY_KEYS = ("gridType", "Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
                 "latitudeOfLastGridPointInDegrees", "longitudeOfLastGridPointInDegrees", "iDirectionIncrementInDegrees",
                 "jDirectionIncrementInDegrees", "iScansNegatively", "jScansPositively", "jPointsAreConsecutive",
                 "numberOfDataPoints", "bitmapPresent")
TARGET_GRID, TARGET_AREA = "N320", [89.785, 0.0, -89.785, 359.719]  # checkpoint metadata.grid / metadata.area


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def read_mat(path):
    """Raw CSR blobs of a MIR/eckit weight cache file, stored order kept."""
    from eckit.stream import Stream

    with open(path, "rb") as f:
        s = Stream(f)
        rows, cols, nnz = s.read_unsigned_long(), s.read_unsigned_long(), s.read_unsigned_long()
        little = s.read_int() != 0
        isz, ssz = s.read_unsigned_long(), s.read_unsigned_long()
        s.read_unsigned_long()
        bo = "<" if little else ">"
        ui = np.dtype({4: np.uint32, 8: np.uint64}[isz]).newbyteorder(bo)
        fl = np.dtype({4: np.float32, 8: np.float64}[ssz]).newbyteorder(bo)
        outer = np.frombuffer(s.read_large_blob(), dtype=ui).astype(np.int64)
        inner = np.frombuffer(s.read_large_blob(), dtype=ui).astype(np.int64)
        data = np.frombuffer(s.read_large_blob(), dtype=fl).astype(np.float64)
    if outer.shape != (rows + 1,) or inner.shape != (nnz,) or data.shape != (nnz,) or outer[-1] != nnz:
        raise ValueError(f"{path}: inconsistent CSR blobs")
    return {"rows": int(rows), "cols": int(cols), "nnz": int(nnz), "index_item_size": int(isz),
            "scalar_item_size": int(ssz), "indptr": outer, "indices": inner, "weights": data}


def find_mats(run: Path) -> list[Path]:
    return sorted((run / "cache" / "ecmwf" / "mir" / "weights").rglob("*.mat"))


def apply_sequential(m, x: np.ndarray) -> np.ndarray:
    """y_i = (((0 + w_i0 x_c0) + w_i1 x_c1) + ...) in double, stored order; numpy: separate mul and add."""
    indptr, idx, w = m["indptr"], m["indices"], m["weights"]
    n = len(indptr) - 1
    length = np.diff(indptr)
    y = np.zeros(n, np.float64)
    for j in range(int(length.max())):
        rows = np.nonzero(length > j)[0]
        k = indptr[rows] + j
        y[rows] = y[rows] + w[k] * x[idx[k]]
    return y


def messages(path: Path):
    import eccodes

    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                return
            yield h


def geometry(h) -> dict:
    import eccodes

    out = {}
    for k in GEOMETRY_KEYS:
        try:
            v = eccodes.codes_get(h, k)
        except Exception:  # noqa: BLE001
            v = None
        out[k] = v
    return out


def mir_run(in_bytes: bytes, **job) -> bytes:
    import mir

    j = mir.Job(**job)
    ib, ob = io.BytesIO(in_bytes), io.BytesIO()
    j.execute(ib, ob)
    return ob.getvalue()


def decode_all(data: bytes) -> list[tuple[str, np.ndarray]]:
    import eccodes

    out = []
    with tempfile.NamedTemporaryFile(suffix=".grib2") as t:
        t.write(data)
        t.flush()
        for h in messages(Path(t.name)):
            out.append((f"{eccodes.codes_get(h, 'shortName')}@{eccodes.codes_get(h, 'level')}",
                        eccodes.codes_get_double_array(h, "values")))
            eccodes.codes_release(h)
    return out


def verify(run: Path, m, mats_run: list[Path]) -> dict:
    import eccodes

    os.environ.pop("ECCODES_DEFINITION_PATH", None)  # as in the run's retrieve+regrid (_without_..._override)
    tmp_cache = tempfile.mkdtemp(prefix="mir-cache-verify-")
    os.environ["MIR_CACHE_PATH"] = tmp_cache
    results, totals = [], {"fields": 0, "points_differ_vs_mir_ieee64": 0, "max_abs_vs_mir_ieee64": 0.0,
                           "fields_reencode_bitwise": 0,
                           "max_abs_reencode_vs_run": 0.0, "calls": 0, "calls_bytes_equal": 0}
    job = {"grid": TARGET_GRID, "area": TARGET_AREA, "edition": 2, "truncation": "auto"}
    for inp in sorted((run / "dumps").glob("regrid.*.in.grib2"), key=lambda p: int(p.name.split(".")[1])):
        k = inp.name.split(".")[1]
        outp = run / "dumps" / f"regrid.{k}.out.grib2"
        in_bytes, run_out = inp.read_bytes(), outp.read_bytes()
        t0 = time.perf_counter()
        ieee = decode_all(mir_run(in_bytes, packing="ieee", accuracy=64, **job))
        ccsds = mir_run(in_bytes, packing="ccsds", accuracy=16, **job)
        seconds = time.perf_counter() - t0
        totals["calls"] += 1
        totals["calls_bytes_equal"] += int(ccsds == run_out)
        run_dec = decode_all(run_out)
        # input values, double decode
        xs = []
        for h in messages(inp):
            xs.append((f"{eccodes.codes_get(h, 'shortName')}@{eccodes.codes_get(h, 'level')}",
                       eccodes.codes_get_double_array(h, "values"), geometry(h)))
            eccodes.codes_release(h)
        # the run's output handles, to re-encode y
        out_handles = list(messages(outp))
        if not (len(xs) == len(ieee) == len(run_dec) == len(out_handles)):
            raise RuntimeError(f"regrid.{k}: message counts differ")
        for (name, x, geo), (n2, yi), (n3, yr), h in zip(xs, ieee, run_dec, out_handles):
            if not (name == n2 == n3):
                raise RuntimeError(f"regrid.{k}: order differs {name} {n2} {n3}")
            y = apply_sequential(m, x)
            d = np.abs(y - yi)
            hc = eccodes.codes_clone(h)
            eccodes.codes_set_double_array(hc, "values", y)
            yre = eccodes.codes_get_double_array(hc, "values")
            eccodes.codes_release(hc)
            eccodes.codes_release(h)
            dre = np.abs(yre - yr)
            r = {"call": int(k), "field": name, "points_differ_vs_mir_ieee64": int((d != 0).sum()),
                 "max_abs_vs_mir_ieee64": float(d.max()), "reencode_bitwise": bool((dre == 0).all()),
                 "reencode_points_differ": int((dre != 0).sum()), "max_abs_reencode_vs_run": float(dre.max()),
                 "max_abs_y_vs_run_decoded": float(np.abs(y - yr).max()), "geometry": geo}
            results.append(r)
            totals["fields"] += 1
            totals["points_differ_vs_mir_ieee64"] += r["points_differ_vs_mir_ieee64"]
            totals["max_abs_vs_mir_ieee64"] = max(totals["max_abs_vs_mir_ieee64"], r["max_abs_vs_mir_ieee64"])
            totals["fields_reencode_bitwise"] += int(r["reencode_bitwise"])
            totals["max_abs_reencode_vs_run"] = max(totals["max_abs_reencode_vs_run"], r["max_abs_reencode_vs_run"])
        print(f"regrid.{k}: {len(xs)} fields, MIR ccsds bytes == run: {ccsds == run_out}, {seconds:.1f} s", flush=True)
    mats_here = sorted(Path(tmp_cache).rglob("*.mat"))
    run_shas = {sha256(p) for p in mats_run}
    here = {str(p.relative_to(tmp_cache)): sha256(p) for p in mats_here}
    totals["mir_cache_files_this_process"] = here
    totals["mir_cache_identical_to_run"] = bool(here) and set(here.values()) <= run_shas
    return {"totals": totals, "per_field": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="reference run dir (its cache/ecmwf/mir holds the matrix)")
    ap.add_argument("--grib", required=True, help="a real open-data file: source geometry")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=str(OUT) if OUT else None, required=OUT is None)
    args = ap.parse_args()
    run = Path(args.run)

    mats = find_mats(run)
    if not mats:
        sys.exit(f"no MIR .mat under {run}/cache/ecmwf/mir")
    shas = {str(p.relative_to(run)): sha256(p) for p in mats}
    if len(set(shas.values())) != 1:
        sys.exit(f"the run used different matrices: {shas}")
    m = read_mat(mats[0])

    import eccodes

    geos = []
    for h in messages(Path(args.grib)):
        geos.append(geometry(h))
        eccodes.codes_release(h)
    # ocean fields (sve, svn, zos, sithick) carry a bitmap; none of the fields the run uses do
    geos = [g for g in geos if not g["bitmapPresent"]]
    geo = geos[0]
    if any(g != geo for g in geos):
        sys.exit("messages of the source file differ in geometry")
    if m["cols"] != geo["numberOfDataPoints"]:
        sys.exit(f"matrix cols {m['cols']} != source points {geo['numberOfDataPoints']}")
    meta = {
        "method": "mir linear (finite-element triangles; MIR default, no interpolation key set)",
        "mir_version": {p: importlib.metadata.version(p) for p in ("mir-python", "mirlib", "eccodeslib")},
        "n_source": m["cols"], "n_target": m["rows"], "nnz": m["nnz"],
        "row_length_hist": np.bincount(np.diff(m["indptr"])).tolist(),
        "mat_index_item_size": m["index_item_size"], "mat_scalar_item_size": m["scalar_item_size"],
        "source_grid": "IFS open data 0.25x0.25 regular_ll", "source_geometry": geo,
        "source_geometry_from": os.path.abspath(args.grib),
        "target_grid": TARGET_GRID, "target_area": TARGET_AREA,
        "mir_job": {"grid": TARGET_GRID, "area": TARGET_AREA, "edition": 2, "packing": "ccsds", "accuracy": 16,
                    "truncation": "auto"},
        "mat_files_sha256": shas, "mat_sha256": next(iter(shas.values())), "run": str(run.resolve()),
        "apply": "y_i = sum_j w_ij x_j in double, sequentially in stored order starting from 0; then 16-bit grid_ccsds "
                 "re-encode (ecCodes) and decode",
    }
    if args.verify:
        v = verify(run, m, mats)
        meta["verification"] = v["totals"]
        vj = Path(args.out).with_suffix(".verify.json")
        tmp = vj.with_name(vj.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(v, indent=1, default=str) + "\n")
        os.replace(tmp, vj)
        print(json.dumps(v["totals"], indent=1, default=str))
    if m["indices"].max() >= 2**31:
        sys.exit("indices do not fit I32")
    tensors = {"regrid.indptr": torch.from_numpy(m["indptr"]),
               "regrid.indices": torch.from_numpy(m["indices"].astype(np.int32)),
               "regrid.weights": torch.from_numpy(m["weights"])}
    size = write_safetensors(args.out, tensors, {"regrid": json.dumps(meta, default=str)})
    print(f"wrote {args.out} ({size / 1e6:.1f} MB), sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
