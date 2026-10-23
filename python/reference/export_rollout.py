"""Export a reference rollout directory (rollout.pt + manifest.json from run_rollout.py) to one safetensors file.

    python -I python/reference/export_rollout.py ROLLOUT_DIR [OUT.safetensors]      # default ROLLOUT_DIR/rollout.safetensors

Container: docs/format.md section 1 (safetensors with __metadata__; written here, no safetensors package; 8-byte dtypes
first, then by name, packed with no gaps, data section 8-byte aligned, the same rule as python/safetensors_io.py). The
tensors are streamed from the memory-mapped rollout.pt in 64 MB pieces, the output goes to a temporary name and is
renamed. Layout (N = steps, G = data nodes, V_in/V_out = model input/output variables, F = dynamic forcings):
docs/rollout.md section "Files".

Needs only torch + numpy (any venv).
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch

CHUNK = 64 << 20
DT = {torch.float32: ("F32", 4), torch.int64: ("I64", 8)}


def build(rollout: dict, manifest: dict, tag: str) -> tuple[dict, dict]:
    """-> (tensors, metadata) with the rollout.pt shapes squeezed as documented."""
    n, g = rollout["y"].shape[0], rollout["y"].shape[-2]
    v_in, v_out = rollout["x0"].shape[-1], rollout["y"].shape[-1]
    in_vars, out_vars = list(rollout["in_vars"]), list(rollout["out_vars"])
    assert len(in_vars) == v_in and len(out_vars) == v_out
    f = rollout["forcings"]
    tensors = {
        "x0": rollout["x0"].reshape(2, g, v_in),
        "y": rollout["y"].reshape(n, g, v_out),
        "window": rollout["window"].reshape(n, 2, g, v_in),
        "forcings": f.reshape(f.shape[0], g, f.shape[-1]),
        "forcing_index": torch.tensor(rollout["forcing_index"], dtype=torch.int64),
        "prog_in": torch.tensor(rollout["prog_in"], dtype=torch.int64),
        "prog_out": torch.tensor(rollout["prog_out"], dtype=torch.int64),
        "kept_in": torch.tensor(sorted(in_vars.index(v) for v in rollout["kept_vars"]), dtype=torch.int64),
    }
    for k in ("mul_in", "add_in", "mul_out", "add_out"):
        tensors[k] = rollout[k].reshape(-1).contiguous()
    dates = list(rollout["dates"])  # valid dates of the forcings: output step s (s = 0..N-2) is dated dates[s]
    assert len(dates) == f.shape[0] == n - 1, (len(dates), f.shape, n)
    meta = {
        "varda_rollout": "1",
        "tag": tag,
        "steps": str(n),
        "start_date": manifest["start_date"],  # date of the newest input slot (x0[1])
        "timestep": manifest.get("checkpoint_timestep", ""),
        "dates": json.dumps(dates),  # forcing s is for the valid date of output s; the last step has no forcing
        "forcing_vars": json.dumps(rollout["forcing_vars"]),
        "in_vars": json.dumps(in_vars),
        "out_vars": json.dumps(out_vars),
        "kept_vars": json.dumps(rollout["kept_vars"]),
        "run": json.dumps({k: manifest.get(k) for k in (
            "device", "backend", "precision", "runner_autocast_dtype", "autocast_effective", "perturb_ulp_seed",
            "chunk_env", "forcings_from", "slurm_job_id", "host", "command", "determinism", "rollout_s", "checkpoint")}),
    }
    return tensors, meta


def write(path: Path, tensors: dict, meta: dict) -> int:
    order = sorted(tensors, key=lambda n: (DT[tensors[n].dtype][1] != 8, n))
    header, off = {"__metadata__": meta}, 0
    for n in order:
        t = tensors[n]
        name, size = DT[t.dtype]
        header[n] = {"dtype": name, "shape": list(t.shape), "data_offsets": [off, off + t.numel() * size]}
        off += t.numel() * size
    h = json.dumps(header).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<Q", len(h)))
            fh.write(h)
            for n in order:
                flat = tensors[n].contiguous().reshape(-1).view(torch.uint8)
                for i in range(0, flat.numel(), CHUNK):
                    fh.write(memoryview(flat[i:i + CHUNK].numpy()))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return 8 + len(h) + off


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollout_dir")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--verify", action="store_true",
                    help="read the file back with python/safetensors_io.py (layout check) and compare every tensor "
                         "bitwise with rollout.pt")
    a = ap.parse_args()
    d = Path(a.rollout_dir)
    out = Path(a.out) if a.out else d / "rollout.safetensors"
    rollout = torch.load(d / "rollout.pt", weights_only=True, mmap=True, map_location="cpu")
    manifest = json.loads((d / "manifest.json").read_text())
    tensors, meta = build(rollout, manifest, d.name)
    size = write(out, tensors, meta)
    print(f"wrote {out} ({size / 2**30:.2f} GiB): " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()))
    if a.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import safetensors_io as sio  # the repository's reader: validates offsets/alignment/size

        m, arrays = sio.read_safetensors(out)
        assert m["varda_rollout"] == "1" and json.loads(m["dates"]) == list(rollout["dates"])
        for k, t in tensors.items():
            assert arrays[k].shape == tuple(t.shape), k
            for i in range(0, t.shape[0], 1):  # row by row (rows are at most 0.5 GB)
                assert torch.equal(torch.from_numpy(arrays[k][i].copy() if t.dim() > 1 else arrays[k][:]), t[i] if t.dim() > 1 else t), k
                if t.dim() <= 1:
                    break
        print("verify: header valid, every tensor bitwise equal to rollout.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
