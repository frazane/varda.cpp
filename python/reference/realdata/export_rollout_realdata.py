"""Turn a real-data reference run's stages.safetensors into a rollout file (docs/format.md section 5).

    python -I python/reference/realdata/export_rollout_realdata.py RUN_DIR TEMPLATE_ROLLOUT [OUT]

RUN_DIR is a run of run_reference_realdata.py (e.g. $VARDA_SCRATCH/realdata/20261022T15/ref-fp16) with its
stages.safetensors (export_stages.py); OUT defaults to RUN_DIR/rollout.safetensors. The model-derived index and
normaliser tensors (prog_in, prog_out, kept_in, forcing_index, mul/add) are copied from TEMPLATE_ROLLOUT, a rollout
file of the same checkpoint (e.g. $VARDA_SCRATCH/rollout/ref/cpu-fp32-pyg/rollout.safetensors); the variables, the
forcing variables and the checkpoint uuid must agree. Only reshapes and copies: every value is the reference's.
Then `varda-rollout --input OUT --compare OUT` judges a real-data rollout per step (docs/rollout-engine.md section 4).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import safetensors_io as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_rollout import write  # noqa: E402


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    run, template = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) == 4 else run / "rollout.safetensors"
    sm, s = st.read_safetensors(run / "stages.safetensors")
    info = json.loads(sm["stages"])
    tm, t = st.read_safetensors(template)
    manifest = json.loads((run / "manifest.json").read_text())

    in_vars, out_vars = list(info["in_vars"]), list(info["out_vars"])
    fvars = list(info["forcing_vars"])
    for key, mine in (("in_vars", in_vars), ("out_vars", out_vars), ("forcing_vars", fvars)):
        assert json.loads(tm[key]) == mine, f"{key} differ from the template rollout's"

    steps = sorted(int(k.split(".")[1]) for k in s if k.startswith("y.") and not k.endswith(".post"))
    n = len(steps)
    assert steps == list(range(n)), steps
    assert sorted(int(k.split(".")[1]) for k in s if k.startswith("window.")) == list(range(1, n))
    assert sorted(int(k.split(".")[1]) for k in s if k.startswith("forcings.")) == list(range(1, n))
    g, v_in, v_out = s["x0"].shape[2], len(in_vars), len(out_vars)

    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    x0 = np.asarray(s["x0"]).reshape(2, g, v_in)
    window = np.empty((n, 2, g, v_in), np.float32)
    window[0] = x0  # step 0 sees x0
    for i in range(1, n):
        window[i] = np.asarray(s[f"window.{i}"]).reshape(2, g, v_in)
    tensors = {
        "x0": T(x0),
        "y": T(np.stack([np.asarray(s[f"y.{i}"]) for i in range(n)])),  # before post-processors
        "window": T(window),
        "forcings": T(np.stack([np.asarray(s[f"forcings.{i}"]).T for i in range(1, n)])),  # [N-1, G, F]
    }
    for k in ("forcing_index", "prog_in", "prog_out", "kept_in", "mul_in", "add_in", "mul_out", "add_out"):
        tensors[k] = T(np.asarray(t[k]))
    iso = lambda d: d.replace(" ", "T")  # noqa: E731
    dates = [iso(info["dates_out"][str(i)]) for i in range(n)]  # valid dates of outputs 0..N-1
    for i in range(1, n):
        assert info["forcing_dates"][str(i)] == [dates[i - 1]], (i, info["forcing_dates"])
    meta = {
        "varda_rollout": "1",
        # varda-rollout picks the tolerance table from the tag prefix (cuda-fp16-* -> fast-fp16-vs-ref-fp16)
        "tag": f"cuda-{'fp16' if manifest.get('autocast') == 'torch.float16' else 'fp32'}-triton-realdata-{run.parent.name}",
        "steps": str(n),
        "start_date": iso(info["dates"][-1]),  # newest input slot
        "timestep": tm.get("timestep", ""),
        "dates": json.dumps(dates[: n - 1]),  # forcing s is for the valid date of output s
        "forcing_vars": json.dumps(fvars),
        "in_vars": json.dumps(in_vars),
        "out_vars": json.dumps(out_vars),
        "kept_vars": tm["kept_vars"],
        "run": json.dumps({"realdata_run": str(run), "autocast": manifest.get("autocast"), "precision_arg": manifest.get("precision_arg"),
                           "source": "stages.safetensors (export_stages.py)"}),
    }
    if "checkpoint_uuid" in tm:
        meta["checkpoint_uuid"] = tm["checkpoint_uuid"]
    nbytes = write(out, tensors, meta)
    print(f"wrote {out} ({nbytes / 1e9:.2f} GB): {n} steps, start {meta['start_date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
