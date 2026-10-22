"""Compare the C++ forcings (tests/test_forcings.cpp with VARDA_FORCINGS_DUMP) with the forcings recorded
by the rollout reference (rollout/forcings/forcings.pt, anemoi's ComputedForcings inside the real runner loop).

    .venv/bin/python -I python/reference/compare_forcings_recorded.py DUMP.safetensors forcings.pt
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import safetensors_io as sio  # noqa: E402

meta, t = sio.read_safetensors(sys.argv[1])
rec = torch.load(sys.argv[2], mmap=True, weights_only=True)
names, dates = json.loads(meta["names"]), json.loads(meta["dates"])
assert names == rec["forcing_vars"] and dates == rec["dates"], "names / dates differ"
a = torch.from_numpy(np.array(t["values"]))  # [N, G, 5]
r = rec["values"][:, 0]  # [N, G, 5]
assert a.shape == r.shape and a.dtype == r.dtype == torch.float32, (a.shape, r.shape)
neq = a.view(torch.int32) != r.view(torch.int32)
print(f"{len(dates)} dates x {a.shape[1]} points x {len(names)} forcings: {int(neq.sum())} of {a.numel()} elements not bitwise equal")
for k, n in enumerate(names):
    print(f"  {n:16s} max_abs {float((a[..., k] - r[..., k]).abs().max()):.3e}  not_bitwise {int(neq[..., k].sum())}")
sys.exit(0 if not neq.any() else 1)
