"""Grid facts for the real-data path: lam_0 vs ICON-CH1 native cells, the cutout masks, N320 coordinates.

Reads the checkpoint's supporting arrays (anemoi-inference Checkpoint metadata, no model load), the frozen
ICON-CH1 horizontal constants (CLAT/CLON), and a run's dumps (source/cutout/regrid coordinates). Writes JSON.

    $PY -I python/reference/realdata/analyse_grids.py RUN OUT.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from eccodes_cosmo_resources import get_definitions_path

os.environ["ECCODES_DEFINITION_PATH"] = str(get_definitions_path())

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetensors_io import read_safetensors  # noqa: E402

VARDA_SCRATCH = os.environ.get("VARDA_SCRATCH") or sys.exit("VARDA_SCRATCH is not set: source tools/env.sh")
HF = Path(VARDA_SCRATCH) / "hf" / "varda-forecaster-det-sgm"


def supporting_arrays() -> dict:
    from anemoi.inference.checkpoint import Checkpoint
    from anemoi.inference.metadata import Metadata

    c = Checkpoint(str(HF / "model.ckpt"), metadata_base=Metadata,
                   patch_metadata=yaml.safe_load((HF / "metadata.yaml").read_text()))
    (ds, m), = c.multi_dataset_metadata.items()
    return {k: np.asarray(v) for k, v in m.supporting_arrays.items()}


def icon_clat_clon(run_cache: Path):
    import eccodes

    out = {}
    with open(run_cache / "constants" / "horizontal_constants_icon-ch1-eps.grib2", "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            n = eccodes.codes_get(h, "shortName")
            if n in ("CLAT", "CLON"):
                out[n] = eccodes.codes_get_double_array(h, "values")
                out[n + "_units"] = eccodes.codes_get(h, "units")
            eccodes.codes_release(h)
    return out


def summary_diff(a, b) -> dict:
    if np.shape(a) != np.shape(b):
        return {"shape_mismatch": [list(np.shape(a)), list(np.shape(b))], "bitwise": False}
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"max_abs": float(d.max()), "n_differ": int((d != 0).sum()), "bitwise": bool((d == 0).all())}


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    sa = supporting_arrays()
    res: dict = {"supporting_arrays": {k: [str(v.dtype), list(v.shape)] for k, v in sa.items()}}
    lam_mask, glob_mask = sa["lam_0/cutout_mask"].astype(bool), sa["global/cutout_mask"].astype(bool)
    res["lam_0/cutout_mask"] = {"n": int(lam_mask.size), "true": int(lam_mask.sum()), "all_true": bool(lam_mask.all())}
    dropped = np.nonzero(~glob_mask)[0]
    res["global/cutout_mask"] = {"n": int(glob_mask.size), "true": int(glob_mask.sum()), "false": int(dropped.size),
                                 "false_index_range": [int(dropped.min()), int(dropped.max())] if dropped.size else None,
                                 "false_lat_range": [float(sa["source1/latitudes"][dropped].min()),
                                                     float(sa["source1/latitudes"][dropped].max())],
                                 "false_lon_range": [float(sa["source1/longitudes"][dropped].min()),
                                                     float(sa["source1/longitudes"][dropped].max())]}
    G = sa["latitudes"].size
    res["G"] = int(G)
    merged_lat = np.concatenate([sa["source0/latitudes"][lam_mask], sa["source1/latitudes"][glob_mask]])
    merged_lon = np.concatenate([sa["source0/longitudes"][lam_mask], sa["source1/longitudes"][glob_mask]])
    res["merged_equals_checkpoint_latlon"] = {"lat": summary_diff(merged_lat, sa["latitudes"]),
                                              "lon": summary_diff(merged_lon, sa["longitudes"])}

    ic = icon_clat_clon(Path(VARDA_SCRATCH) / "realdata" / "20261022T15" / "cache" / "kenda")
    clat, clon = ic["CLAT"], ic["CLON"]
    res["icon_ch1_horizontal_constants"] = {"n": int(clat.size), "units": [ic["CLAT_units"], ic["CLON_units"]],
                                            "lat_range": [float(clat.min()), float(clat.max())],
                                            "lon_range": [float(clon.min()), float(clon.max())]}
    s0lat, s0lon = sa["source0/latitudes"], sa["source0/longitudes"]
    res["lam_vs_icon_native"] = {"same_count": bool(s0lat.size == clat.size),
                                 "lat": summary_diff(s0lat, clat), "lon": summary_diff(s0lon, clon),
                                 "lon_mod360": summary_diff(np.mod(s0lon, 360), np.mod(clon, 360)),
                                 "lat_f32": summary_diff(s0lat.astype(np.float32), clat.astype(np.float32)),
                                 "lon_f32": summary_diff(s0lon.astype(np.float32), clon.astype(np.float32))}

    # the run's states: coordinates actually used
    def coords(name):
        meta, t = read_safetensors(run / "dumps" / f"{name}.safetensors")
        return np.asarray(t["latitudes"]), np.asarray(t["longitudes"])

    lat, lon = coords("source.prognostics.lam_0")
    res["run_lam_coords_vs_source0"] = {"lat": summary_diff(lat, s0lat), "lon": summary_diff(lon, s0lon)}
    lat, lon = coords("source.prognostics.global")
    res["run_global_coords_vs_source1"] = {"lat": summary_diff(lat, sa["source1/latitudes"]),
                                           "lon": summary_diff(lon, sa["source1/longitudes"])}
    meta, t = read_safetensors(run / "dumps" / "regrid.0.safetensors")
    res["mir_out_coords_vs_source1"] = {"lat": summary_diff(t["out_latitudes"], sa["source1/latitudes"]),
                                        "lon": summary_diff(t["out_longitudes"], sa["source1/longitudes"])}
    lat, lon = coords("cutout.prognostics")
    res["run_cutout_coords_vs_checkpoint"] = {"lat": summary_diff(lat, sa["latitudes"]),
                                              "lon": summary_diff(lon, sa["longitudes"])}
    _, t = read_safetensors(run / "dumps" / "cutout.prognostics.safetensors")
    ml, mg = np.asarray(t["mask/lam_0"]), np.asarray(t["mask/global"])
    res["cutout_private_masks"] = {"lam_0_true": int(ml.sum()), "lam_0_is_prefix": bool(ml[:int(ml.sum())].all()),
                                   "global_true": int(mg.sum()), "disjoint_and_cover": bool((ml ^ mg).all())}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
