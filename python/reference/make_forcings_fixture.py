"""Reference dynamic forcings from anemoi's own code, for tests/test_forcings.cpp (docs/forcings.md).

    PY=$REF_VENV/bin/python
    $PY -I python/reference/make_forcings_fixture.py [--out $VARDA_SCRATCH/forcings]

Runs `anemoi.inference.forcings.ComputedForcings.load_forcings_array` (anemoi-inference 9505dd0, imported
from the read-only clone in $VARDA_SCRATCH/src without installing anything) on top of the earthkit-data /
earthkit-meteo / anemoi-transform installed in the Varda reference venv ($REF_VENV), with the latitudes /
longitudes anemoi-inference reads from the checkpoint (`load_metadata(..., supporting_arrays=True)`,
dataset "data"). One file per model: `<out>/<model>.safetensors`, tensor `forcings/<YYYY-MM-DDTHH:MM:SS>`
F32 [G, 5] (columns in `names` order), metadata `names`, `dates`, `versions`, `checkpoint`. `lat` / `lon`
(F64 [G]) are stored too, so the C++ test needs no other input. Also cross-checks insolation against
`earthkit.meteo.solar.cos_solar_zenith_angle`.
"""

import argparse
import datetime
import json
import os
import struct
import sys
import types

import numpy as np

SCRATCH = os.environ.get("VARDA_SCRATCH") or sys.exit("VARDA_SCRATCH is not set: source tools/env.sh")
AI_SRC = os.path.join(SCRATCH, "src", "anemoi-inference", "src")  # read-only clone at 9505dd0
sys.path.insert(0, AI_SRC)
_v = types.ModuleType("anemoi.inference._version")  # generated at install time; absent in a bare clone
_v.__version__ = "0.12.1.dev16+g9505dd092"
sys.modules["anemoi.inference._version"] = _v

from anemoi.inference.forcings import ComputedForcings  # noqa: E402
from anemoi.utils.checkpoints import load_metadata  # noqa: E402

NAMES = ["cos_julian_day", "cos_local_time", "insolation", "sin_julian_day", "sin_local_time"]
MODELS = {
    "varda": "VARDA_CKPT",  # model name -> environment variable with its checkpoint (tools/env.sh)
    "dev": "VARDA_DEV_CKPT",
}


def dates_list():
    d = [
        "2024-06-01T06:00:00", "2024-06-01T12:00:00", "2024-12-31T18:00:00", "2024-02-29T00:00:00",
        "2025-01-01T00:00:00",
        "2024-01-01T00:00:00", "2023-12-31T23:00:00", "2023-03-01T00:00:00", "2024-02-29T23:00:00",
        "2024-07-01T09:00:00", "2024-06-21T15:00:00", "2000-02-29T12:00:00", "2100-03-01T06:00:00",
    ]
    t0 = datetime.datetime(2024, 6, 1)
    d += [(t0 + datetime.timedelta(hours=6 * k)).isoformat() for k in range(1, 9)]  # 6-hourly rollout steps
    d += [(t0 + datetime.timedelta(hours=k)).isoformat() for k in (1, 2, 3)]  # hourly (dev-model timestep)
    return list(dict.fromkeys(d))


def write_safetensors(path, tensors, metadata):
    """Minimal writer (docs/format.md section 1): 8-byte dtypes first, then by name."""
    code = {np.dtype("<f4"): "F32", np.dtype("<f8"): "F64"}
    order = sorted(tensors, key=lambda n: (tensors[n].dtype.itemsize != 8, n))
    header, off = {"__metadata__": metadata}, 0
    for n in order:
        a = tensors[n]
        header[n] = {"dtype": code[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    h = json.dumps(header).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for n in order:
            f.write(np.ascontiguousarray(tensors[n]).tobytes())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCRATCH, "forcings"))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import anemoi.transform
    import earthkit.data
    import earthkit.meteo
    from earthkit.meteo.solar import cos_solar_zenith_angle

    versions = {
        "numpy": np.__version__, "earthkit-data": earthkit.data.__version__,
        "earthkit-meteo": earthkit.meteo.__version__, "anemoi-transform": anemoi.transform.__version__,
        "anemoi-inference": "9505dd0926ec3c5dea9308813cc7859bfdad755a (clone, ComputedForcings used as is)",
        "python": sys.version.split()[0],
    }
    print(versions)
    dates = dates_list()
    for model in args.models:
        ckpt = os.environ.get(MODELS[model]) or sys.exit(f"{MODELS[model]} is not set: source tools/env.sh")
        _, arrays = load_metadata(ckpt, supporting_arrays=True)  # what anemoi-inference's Checkpoint does
        lat, lon = arrays["data"]["latitudes"], arrays["data"]["longitudes"]
        assert lat.dtype == np.float64 and lon.dtype == np.float64 and lat.shape == lon.shape
        print(model, "G =", lat.shape[0])
        cf = ComputedForcings(None, NAMES, None)
        state = {"latitudes": lat, "longitudes": lon}
        out = {"lat": np.ascontiguousarray(lat), "lon": np.ascontiguousarray(lon)}
        for iso in dates:
            when = datetime.datetime.fromisoformat(iso)  # naive, as the runner's dates are
            arr = cf.load_forcings_array([when], state)  # (variables, dates, values) float32
            assert arr.dtype == np.float32 and arr.shape == (len(NAMES), 1, lat.shape[0]), (arr.dtype, arr.shape)
            out[f"forcings/{iso}"] = np.ascontiguousarray(arr[:, 0, :].T)  # [G, n]
            # independent cross-check of the insolation column against earthkit-meteo's own function
            z = cos_solar_zenith_angle(when, lat, lon).astype(np.float32)
            same = np.array_equal(z, arr[2, 0], equal_nan=True)
            print(f"  {iso}: insolation == earthkit.meteo.solar.cos_solar_zenith_angle (f32): {same};"
                  f" max abs {np.abs(arr).max():.4f}")
        meta = {"varda_forcings_fixture": "1", "names": json.dumps(NAMES), "dates": json.dumps(dates),
                "versions": json.dumps(versions), "checkpoint": ckpt,
                "layout": "forcings/<date>: F32 [G, len(names)]; lat, lon: F64 [G] degrees"}
        path = os.path.join(args.out, f"{model}.safetensors")
        write_safetensors(path, out, meta)
        print("wrote", path, os.path.getsize(path) / 1e6, "MB")


if __name__ == "__main__":
    main()
