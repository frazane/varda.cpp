"""Minimal safetensors reader/writer (numpy + torch + stdlib only; docs/format.md section 1).

    [u64 LE N][N bytes JSON header, space-padded so 8 + N is a multiple of 8][data]

Tensors are packed contiguously with no gaps; 8-byte dtypes come first, then by name, so every
tensor starts at a multiple of its element size (the data section starts 8-byte aligned).
"""

import json
import os
import struct

import numpy as np
import torch

DTYPES = {  # safetensors name -> (torch dtype, numpy dtype, itemsize)
    "F64": (torch.float64, np.dtype("<f8"), 8),
    "F32": (torch.float32, np.dtype("<f4"), 4),
    "F16": (torch.float16, np.dtype("<f2"), 2),
    "I64": (torch.int64, np.dtype("<i8"), 8),
    "I32": (torch.int32, np.dtype("<i4"), 4),
    "I16": (torch.int16, np.dtype("<i2"), 2),
    "I8": (torch.int8, np.dtype("i1"), 1),
    "U8": (torch.uint8, np.dtype("u1"), 1),
    "BOOL": (torch.bool, np.dtype("?"), 1),
}
_BY_TORCH = {v[0]: k for k, v in DTYPES.items()}


def _raw(t):
    """Contiguous little-endian bytes of a tensor as a uint8 numpy array (no copy if possible)."""
    t = t.detach().cpu().contiguous()
    return t.reshape(-1).view(torch.uint8).numpy() if t.numel() else np.zeros(0, np.uint8)


def write_safetensors(path, tensors, metadata):
    """Write {name: torch.Tensor} + {str: str} metadata atomically (temporary name, then rename).

    Order: 8-byte dtypes first, then by name. Returns the file size in bytes."""
    path = os.fspath(path)
    for k, v in metadata.items():
        if not (isinstance(k, str) and isinstance(v, str)):
            raise TypeError("metadata must map str to str")
    for n, t in tensors.items():
        if t.dtype not in _BY_TORCH:
            raise TypeError(f"{n}: unsupported dtype {t.dtype}")
    order = sorted(tensors, key=lambda n: (DTYPES[_BY_TORCH[tensors[n].dtype]][2] != 8, n))
    header, off = {"__metadata__": dict(metadata)}, 0
    for n in order:
        t = tensors[n]
        nb = t.numel() * DTYPES[_BY_TORCH[t.dtype]][2]
        header[n] = {"dtype": _BY_TORCH[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
        off += nb
    h = json.dumps(header).encode()
    h += b" " * ((8 - (8 + len(h)) % 8) % 8)
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for n in order:
                f.write(memoryview(_raw(tensors[n])))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return 8 + len(h) + off


def read_header(path):
    """(header dict, data section start offset). Validates the container layout."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if 8 + n > size:
            raise ValueError("header length exceeds file")
        header = json.loads(f.read(n))
    start = 8 + n
    if start % 8:
        raise ValueError(f"data section not 8-byte aligned (starts at {start})")
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items() if k != "__metadata__")
    pos = 0
    for b, e, k in spans:
        if b != pos:
            raise ValueError(f"{k}: offset {b}, expected {pos} (gap or overlap)")
        info = header[k]
        dt = DTYPES[info["dtype"]]
        if e - b != int(np.prod(info["shape"], dtype=np.int64)) * dt[2]:
            raise ValueError(f"{k}: byte span does not match shape/dtype")
        if b % dt[2]:
            raise ValueError(f"{k}: offset {b} not aligned to {dt[2]}")
        pos = e
    if start + pos != size:
        raise ValueError(f"file size {size} != header + data {start + pos}")
    return header, start


def read_safetensors(path):
    """(metadata, {name: numpy array}) with the arrays memory-mapped read-only from the file."""
    header, start = read_header(path)
    meta = header.get("__metadata__", {})
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dt = DTYPES[v["dtype"]][1]
        b, e = v["data_offsets"]
        out[k] = (np.memmap(path, dtype=dt, mode="r", offset=start + b, shape=tuple(v["shape"]))
                  if e > b else np.zeros(tuple(v["shape"]), dt))
    return meta, out
