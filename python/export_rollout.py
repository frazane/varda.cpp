"""Export a reference rollout of the Python project ($IE_ROOT: reference/run_rollout.py, rollout.pt +
manifest.json [+ forcings.pt]) to the varda.cpp rollout file read by apps/varda-rollout.

    .venv/bin/python -I python/export_rollout.py ROLLOUT_DIR OUT.safetensors --model CONVERTED.safetensors [--verify]

Layout: the same as python/reference/export_rollout.py (the Varda references), so the C++ side reads both:

    x0 [2, G, V_in], y [N, G, V_out], window [N, 2, G, V_in] (the window step s sees), forcings [N-1, G, F]
    (forcing s: valid date of output s, written after step s), forcing_index / prog_in / prog_out / kept_in I64,
    mul_in / add_in [V_in], mul_out / add_out [V_out];
    __metadata__: varda_rollout "1", tag, steps, start_date (date of the newest input slot x0[1]), timestep,
    dates (JSON), forcing_vars, in_vars, out_vars, kept_vars (JSON), run (JSON), plus checkpoint_uuid.

The Python project's rollout.pt has no variable names; they come from the converted model's config (--model; the
manifest's checkpoint must be the one the model was converted from: compared by path and, through the tolerances, by
uuid). If ROLLOUT_DIR has a forcings.pt (the recorded ComputedForcings), it must equal rollout.pt's forcings bitwise.
Needs torch + numpy (.venv). Output written atomically (python/safetensors_io.py).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safetensors_io as sio  # noqa: E402


def model_config(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return json.loads(header["__metadata__"]["config"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollout_dir")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, help="converted model.safetensors (variable names, checkpoint uuid)")
    ap.add_argument("--tag", help="default: the directory name")
    ap.add_argument("--verify", action="store_true", help="read the file back and compare every tensor bitwise")
    a = ap.parse_args()
    d = Path(a.rollout_dir)
    r = torch.load(d / "rollout.pt", weights_only=True, mmap=True, map_location="cpu")
    man = json.loads((d / "manifest.json").read_text())
    cfg = model_config(Path(a.model))
    assert r.get("mode", "inference") == "inference", r.get("mode")
    in_vars, out_vars = list(cfg["variables"]["input"]), list(cfg["variables"]["output"])
    n, g = r["y"].shape[0], r["y"].shape[-2]
    v_in, v_out = r["x0"].shape[-1], r["y"].shape[-1]
    assert (len(in_vars), len(out_vars), g) == (v_in, v_out, cfg["dims"]["n_data_nodes"]), "model and rollout differ"
    assert Path(man["checkpoint"]).name == Path(cfg["source"]["checkpoint"]).name or \
        man["checkpoint"] == cfg["source"]["checkpoint"], (man["checkpoint"], cfg["source"]["checkpoint"])
    assert [int(i) for i in r["prog_in"]] == [int(i) for i in cfg["indices"]["residual_input"]], "prog_in"
    assert [int(i) for i in r["prog_out"]] == [int(i) for i in cfg["indices"]["residual_output"]], "prog_out"
    assert [in_vars[i] for i in r["forcing_index"]] == list(r["forcing_vars"]), "forcing names/indices"
    f = r["forcings"]
    if (d / "forcings.pt").exists():
        fr = torch.load(d / "forcings.pt", weights_only=True, mmap=True)
        assert list(fr["dates"]) == list(r["dates"]) and torch.equal(fr["values"], f), "forcings.pt != rollout forcings"
    dates = list(r["dates"])
    assert len(dates) == f.shape[0] == n - 1, (len(dates), tuple(f.shape), n)
    tensors = {
        "x0": r["x0"].reshape(2, g, v_in),
        "y": r["y"].reshape(n, g, v_out),
        "window": r["window"].reshape(n, 2, g, v_in),
        "forcings": f.reshape(f.shape[0], g, f.shape[-1]),
        "forcing_index": torch.tensor([int(i) for i in r["forcing_index"]], dtype=torch.int64),
        "prog_in": torch.tensor([int(i) for i in r["prog_in"]], dtype=torch.int64),
        "prog_out": torch.tensor([int(i) for i in r["prog_out"]], dtype=torch.int64),
        "kept_in": torch.tensor(sorted(in_vars.index(v) for v in r["kept_vars"]), dtype=torch.int64),
    }
    for k in ("mul_in", "add_in", "mul_out", "add_out"):
        tensors[k] = r[k].reshape(-1).contiguous()
    meta = {
        "varda_rollout": "1",
        "tag": a.tag or d.name,
        "steps": str(n),
        "start_date": man["start_date"],
        "timestep": "",  # not recorded by the Python project; the model's forcings.timestep_hours applies
        "dates": json.dumps(dates),
        "forcing_vars": json.dumps(list(r["forcing_vars"])),
        "in_vars": json.dumps(in_vars),
        "out_vars": json.dumps(out_vars),
        "kept_vars": json.dumps(list(r["kept_vars"])),
        "checkpoint_uuid": cfg["source"]["uuid"],
        "run": json.dumps({k: man.get(k) for k in (
            "device", "backend", "precision", "runner_autocast_dtype", "autocast_effective", "forcings_from",
            "host", "command", "checkpoint")} | {"source": str((d / "rollout.pt").resolve())}),
    }
    size = sio.write_safetensors(a.out, tensors, meta)
    print(f"wrote {a.out} ({size / 2**30:.2f} GiB): " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()))
    if a.verify:
        m, arrays = sio.read_safetensors(a.out)
        assert m == meta, "metadata"
        for k, t in tensors.items():
            assert torch.equal(torch.from_numpy(arrays[k].copy()), t), k
        print("verify: every tensor bitwise equal to rollout.pt, metadata equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
