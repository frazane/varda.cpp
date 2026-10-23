"""Export a reference golden directory (anemoi, $REF_SCRATCH/golden) to one safetensors file (docs/format.md §3).

    python -I python/export_golden.py GOLDEN_DIR OUT.safetensors [--tag TAG]

GOLDEN_DIR holds `activations.pt` (a dict[str, Tensor] written by torch.save) and
`manifest.json`. Every key is written as F32 with its shape, one tensor at a time (the
input is memory-mapped), and checked against the manifest's per-key sha256 when it has
one. The output is written to a temporary name and renamed.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import zipfile

import torch


def tensor_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.detach().contiguous().view(torch.uint8).numpy()).cast("B")


def checkpoint_uuid(manifest: dict) -> str:
    """The golden data's checkpoint uuid: manifest checkpoint_identity, else the uuid in the
    embedded anemoi.json of the checkpoint file the manifest names (plain JSON, no unpickling)."""
    ident = manifest.get("checkpoint_identity") or {}
    if ident.get("uuid"):
        return ident["uuid"]
    with zipfile.ZipFile(manifest["checkpoint"]) as z:
        member = next(n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json"))
        return json.loads(z.read(member))["uuid"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("golden_dir")
    ap.add_argument("out")
    ap.add_argument("--tag", help="default: manifest tag")
    args = ap.parse_args()

    src = os.path.join(args.golden_dir, "activations.pt")
    manifest_text = open(os.path.join(args.golden_dir, "manifest.json")).read()
    manifest = json.loads(manifest_text)
    acts = torch.load(src, weights_only=True, mmap=True, map_location="cpu")
    if not isinstance(acts, dict):
        raise SystemExit(f"{src}: expected a dict of tensors, got {type(acts).__name__}")

    names = list(acts)  # keep the dump's key order
    for n in names:
        if acts[n].dtype != torch.float32:
            raise SystemExit(f"{src}: key {n} has dtype {acts[n].dtype}, contract requires float32")

    header, off = {}, 0
    header["__metadata__"] = {
        "varda_golden": "1",
        "tag": args.tag or manifest.get("tag", ""),
        "manifest": manifest_text,
        "source": os.path.abspath(src),
        "source_size": str(os.path.getsize(src)),
        "checkpoint_uuid": checkpoint_uuid(manifest),
    }
    for n in names:
        nb = acts[n].numel() * 4
        header[n] = {"dtype": "F32", "shape": list(acts[n].shape), "data_offsets": [off, off + nb]}
        off += nb
    h = json.dumps(header).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)

    tensors_meta = manifest.get("tensors", {})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp.{os.getpid()}"
    checked = 0
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for n in names:
                b = tensor_bytes(acts[n])
                want = tensors_meta.get(n, {}).get("sha256")
                if want:
                    got = hashlib.sha256(b).hexdigest()
                    if got != want:
                        raise SystemExit(f"{n}: sha256 {got} != manifest {want}")
                    checked += 1
                f.write(b)
        os.replace(tmp, args.out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"{args.out}: {len(names)} keys, {off / 1e9:.2f} GB, {checked} sha256-checked against the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
