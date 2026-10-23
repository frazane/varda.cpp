"""Roll the ORIGINAL model (the Varda forecaster) out for N steps with anemoi-inference's own loop; save the steps.

Variable counts come from the model, the time step from the checkpoint (Varda: 6 h; lead_time = steps x timestep,
dates follow the runner); inference mode only; a stub-model mode records the forcings; forcings replay is
cross-checked against the runner's own ComputedForcings; per-run bitwise check without keeping every run in memory.

Runs in $VARDA_SCRATCH/venv-ref-varda only (it imports anemoi-inference, anemoi-models). Never import this from src/.
anemoi-inference is pinned to 9505dd0926ec3c5dea9308813cc7859bfdad755a there (recipe: header of run_reference.py;
lock: requirements-ref-varda.lock.txt; differences to eaeb36fa: docs/rollout.md).

Inference mode: anemoi-inference's own loop, not a mirror of it: a DefaultRunner(RunConfiguration(checkpoint=...,
input="dummy", output="none", device, precision)) whose `model` (a cached_property) is replaced by the model loaded as
in run_reference.py (map_location="cpu", attention backend override, .eval()), then runner.run(input_states={"data":
state}, lead_time=N*timestep, return_numpy=False). The state holds the golden input (all input fields, (2, G) each, so
no initial forcing is computed) and the date of its newest slot (--date). Everything else is the runner's code:
Runner.forecast (inference_mode, autocast with the runner's precision), copy_prognostic_fields_to_input_tensor and
add_dynamic_forcings_to_input_tensor (tensors.py), the "check" that every non-constant input is refreshed. Two
record-only hooks: a RecordingRunner.predict_step override that copies the window each step sees, and a wrapper
around the dynamic forcing providers' load_forcings_array (anemoi's real ComputedForcings: cos/sin julian day,
cos/sin local time, insolation from the dates and the grid) that records the arrays it returns. --forcings-from FILE
replays recorded forcings instead (same variables and dates asserted; the provider's own result is computed too and
asserted bitwise equal), so every device sees the identical forcings. --forcings-only runs the same loop with a
stub model (zeros out) to produce forcings.pt in seconds without running the network.

Usage (compute nodes; see CLAUDE.md; PY=$VARDA_SCRATCH/venv-ref-varda/bin/python; R=$VARDA_SCRATCH/rollout):

    # forcings of the 24-step rollout (stub model, any node, ~1 min); the other runs replay them
    $PY python/reference/run_rollout.py --forcings-only --steps 24 --out $R/forcings
    sbatch ($VARDA_SLURM_CPU --mem=300G, a partition allowing > 1 h) $PY python/reference/run_rollout.py --device cpu --backend pyg --precision 32 \
        --steps 24 --forcings-from $R/forcings/forcings.pt --out $R/ref/cpu-fp32-pyg
    sbatch ($VARDA_SLURM_GPU, 1 h) $PY python/reference/run_rollout.py --device cuda \
        --backend triton --precision 32|checkpoint --steps 24 --runs 2 --forcings-from $R/forcings/forcings.pt \
        --out $R/ref/cuda-fp32-triton | cuda-fp16-triton
    # perturbed fp16 ensemble (docs/rollout.md): ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=8 | _PROCESSOR=4 with
    # --allow-chunk-env; --perturb-ulp 1|2; --backend pyg on cuda; all with --precision checkpoint
    $PY python/reference/run_rollout.py --calibrate $R/ref/cpu-fp32-pyg $R/ref/cuda-fp32-triton \
        $R/ref/cuda-fp16-triton --fp16-ensemble $ENS [--check tolerances/rollout/varda.json]
    # writes tolerances/rollout/<model>.json (commit it); --check: exit 1 unless the tables equal the committed file
    $PY python/reference/export_rollout.py $R/ref/<tag> $R/ref/<tag>/rollout.safetensors    # for the C++ side

Output (--out): rollout.pt = {x0 (1,2,G,V_in), y (N,1,1,1,G,V_out) physical, window (N,1,2,G,V_in) seen by each step,
forcings (N-1,1,G,F), forcing_vars, forcing_index, dates, prog_in, prog_out, kept_vars, in_vars, out_vars, mul_in,
add_in, mul_out, add_out}, forcings.pt, manifest.json.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_reference as rr  # noqa: E402
from compare_golden import checkpoint_identity, diff_metric  # noqa: E402

DS = rr.DATASET
ROOT = f"{rr.VARDA_SCRATCH}/rollout"
REF_GOLDEN = os.environ.get("REF_GOLDEN") or sys.exit("REF_GOLDEN is not set: source tools/env.sh")  # read-only
INPUT = f"{REF_GOLDEN}/cpu_float32_pyg/input.pt"  # the reference golden one-step input (read-only)
DATE = "2024-06-01T00:00:00"  # date of the newest input slot; the forcing dates follow the runner (+ checkpoint timestep)
MODEL = "varda"  # name of the tolerances file: tolerances/rollout/<MODEL>.json
KEYS = ("y", "y_norm", "window", "window_norm")
REPO = Path(__file__).resolve().parents[1]
ROLLOUT_TOLERANCES_DIR = REPO / "tolerances" / "rollout"  # tolerances/rollout/<model>.json, found by checkpoint uuid


def versions() -> dict:
    v = rr.versions()
    for p in ("anemoi-inference", "earthkit-data"):
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v[p] = None
    return v


def load_model(ckpt: str, device: torch.device, backend: str):
    """As run_reference.py: map_location="cpu", attention backend override, fp32, .eval(), then to the device."""
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    changed = rr.set_attention_backend(model, backend)
    model = model.to(dtype=torch.float32).eval().to(device)
    assert not any(m.training for m in model.modules())
    return model, changed


def normaliser(model) -> dict:
    norm = model.pre_processors[DS].processors["normalizer"]
    mul, add = norm._norm_mul.detach().cpu(), norm._norm_add.detach().cpu()
    i, o = norm._input_idx.long().cpu(), norm._output_idx.long().cpu()
    return {"mul_in": mul[i].clone(), "add_in": add[i].clone(), "mul_out": mul[o].clone(), "add_out": add[o].clone()}


# ------------------------------------------------------------------------------------------ inference semantics


def make_runner(model, ckpt: str, device: torch.device, precision: str | None):
    """A real anemoi-inference DefaultRunner around `model`, plus a record-only predict_step override."""
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners.default import DefaultRunner

    class RecordingRunner(DefaultRunner):
        windows: list | None = None
        stub_n_out: int | None = None  # --forcings-only: no network, zeros of the right shape

        def predict_step(self, model, input_tensors_torch, **kwargs):
            if self.windows is not None:
                self.windows.append(input_tensors_torch[DS].detach().to("cpu", copy=True))
            if self.stub_n_out is not None:
                x = input_tensors_torch[DS]
                return {DS: torch.zeros(x.shape[0], 1, 1, x.shape[2], self.stub_n_out, dtype=x.dtype, device=x.device)}
            return super().predict_step(model, input_tensors_torch, **kwargs)

    cfg = RunConfiguration(checkpoint=ckpt, device=str(device), precision=precision, input="dummy", output="none",
                           lead_time=1)
    runner = RecordingRunner(cfg)
    runner.__dict__["model"] = model  # Runner.model is a cached_property: this is the model it uses
    return runner


class ForcingTape:
    """Records (or replays) what the runner's dynamic forcing providers return, per call (= per step)."""

    def __init__(self, runner, replay: dict | None = None):
        h = runner.tensor_handlers[DS]
        self.providers = h.dynamic_forcings_providers
        self.vars = [v for p in self.providers for v in p.variables]
        self.index = [int(i) for p in self.providers for i in p.mask]
        self.calls: list[tuple[str, np.ndarray]] = []  # (date iso, (F, G) float32) per step
        self.n_crosschecked = 0
        self.replay = replay
        if replay is not None:
            assert replay["forcing_vars"] == self.vars and replay["forcing_index"] == self.index, \
                (replay["forcing_vars"], self.vars)
            self.by_date = {d: replay["values"][k, 0].numpy().T.copy() for k, d in enumerate(replay["dates"])}
        for p in self.providers:
            orig = p.load_forcings_array

            def load(dates, state, _p=p, _orig=orig):
                assert len(dates) == 1, dates
                d = dates[0].isoformat()
                if self.replay is not None:
                    full = self.by_date[d]  # (F_all, G)
                    lo = sum(len(q.variables) for q in self.providers[:self.providers.index(_p)])
                    a = full[lo:lo + len(_p.variables)][:, None, :]  # (vars, dates=1, G), float32
                    own = _orig(dates, state)  # what the runner itself computes for this date: must be bitwise the same
                    assert np.array_equal(np.asarray(own), a), f"replayed forcings differ from ComputedForcings at {d}"
                    self.n_crosschecked += 1
                else:
                    a = _orig(dates, state)
                self.calls.append((d, _p, np.array(a, copy=True)))
                return a

            p.load_forcings_array = load

    def stacked(self) -> tuple[list[str], torch.Tensor]:
        """-> dates, (S, 1, G, F) in forcing_vars order (providers concatenated)."""
        dates, out = [], []
        n = len(self.providers)
        assert len(self.calls) % n == 0
        for k in range(0, len(self.calls), n):
            group = self.calls[k:k + n]
            assert len({d for d, _, _ in group}) == 1
            dates.append(group[0][0])
            arr = np.concatenate([a[:, 0, :] for _, _, a in group], axis=0)  # (F, G)
            out.append(torch.from_numpy(arr.T.copy())[None])  # (1, G, F)
        return dates, torch.stack(out) if out else torch.empty(0)


def run_inference(runner, x0: torch.Tensor, n_steps: int, date: datetime.datetime, in_names, out_names,
                  record_windows: bool = True, keep: str = "cpu", timer: list | None = None):
    """runner.run(...) for n_steps; returns y (N,1,1,1,G,72) and the windows (N,1,2,G,84) (CPU).

    keep: "cpu" (each step's output fields stacked and copied to CPU), "device" (stacked, left on the device:
    no host sync per step), "numpy" (the runner's own return_numpy=True: per-field .cpu().numpy(), production).
    """
    return_numpy = keep == "numpy"
    h = runner.tensor_handlers[DS]
    v2i = h.metadata.variable_to_input_tensor_index
    assert [v2i[n] for n in in_names] == list(range(len(in_names))), "input order differs from the runner's"
    state = {"date": date, "fields": {n: x0[0, :, :, i].numpy().copy() for i, n in enumerate(in_names)}}
    runner.windows = [] if record_windows else None
    ys = []
    if runner.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    lead_time = runner.checkpoint.timestep * n_steps  # NOT the int n_steps: an int is hours (anemoi's to_timedelta)
    for st in runner.run(input_states={DS: state}, lead_time=lead_time, return_numpy=return_numpy):
        f = st[DS]["fields"]
        if return_numpy:
            ys.append(np.stack([f[n] for n in out_names], -1))
        else:
            y = torch.stack([f[n] for n in out_names], -1)[None, None, None]
            ys.append(y.to("cpu", copy=True) if keep == "cpu" else y)
    if runner.device.type == "cuda":
        torch.cuda.synchronize()
    if timer is not None:
        timer.append(time.perf_counter() - t0)
    assert len(ys) == n_steps, len(ys)
    if return_numpy:
        return ys, None
    windows = torch.stack(runner.windows) if record_windows else None
    runner.windows = None
    return torch.stack(ys), windows


# ------------------------------------------------------------------------------------------ calibration


def step_metrics(ref: dict, other: dict, n: int | None = None) -> dict:
    """Per step, contract metric (docs/format.md §2) of y, window and their normalised versions (float64)."""
    n = n or min(ref["y"].shape[0], other["y"].shape[0])
    mo, ao = ref["mul_out"].double(), ref["add_out"].double()
    mi, ai = ref["mul_in"].double(), ref["add_in"].double()
    res = {k: [] for k in KEYS}
    for s in range(n):
        ry, oy = ref["y"][s].double(), other["y"][s].double()
        rw, ow = ref["window"][s].double(), other["window"][s].double()
        res["y"].append(diff_metric(ry, oy))
        res["y_norm"].append(diff_metric(ry * mo + ao, oy * mo + ao))
        res["window"].append(diff_metric(rw, ow))
        res["window_norm"].append(diff_metric(rw * mi + ai, ow * mi + ai))
    return res


def round_up(x: float, digits: int = 2) -> float:
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - digits + 1
    return float(f"{math.ceil(x / 10 ** e) * 10 ** e:.{digits}g}")


def compare(ref_dir: str, dirs: list[str], out: str | None, per_var: int | None = None) -> dict:
    """Per-step metrics (y, y_norm, window, window_norm) of each rollout in dirs against ref_dir; prints y_norm/y."""
    load = lambda d: torch.load(Path(d) / "rollout.pt", weights_only=True, mmap=True)  # noqa: E731
    ref = load(ref_dir)
    res = {}
    for d in dirs:
        o = load(d)
        res[d] = step_metrics(ref, o, min(ref["y"].shape[0], o["y"].shape[0]))
    names = [Path(d).name for d in dirs]
    print(f"vs {ref_dir}: rel_absmax per step, y_norm | y")
    print(f"{'step':>4s} " + " ".join(f"{n[-30:]:>30s}" for n in names))
    for s in range(max(len(v["y"]) for v in res.values())):
        print(f"{s:4d} " + " ".join((f"{res[d]['y_norm'][s]['rel_absmax']:14.3e} {res[d]['y'][s]['rel_absmax']:14.3e}  "
                                     if s < len(res[d]["y"]) else " " * 32) for d in dirs))
    if per_var is not None:  # which output variables carry the error, and is it a bias (mean signed error)?
        names_out = ref["out_vars"]
        s = per_var
        ry = ref["y"][s].double().reshape(-1, len(names_out))
        mo, ao = ref["mul_out"].double(), ref["add_out"].double()
        for d in dirs:
            oy = load(d)["y"][s].double().reshape(-1, len(names_out))
            dn = (oy - ry) * mo  # normalised error
            top = torch.argsort(dn.abs().amax(0), descending=True)[:6].tolist()
            print(f"step {s}, {Path(d).name}: top variables by max |error| (normalised units): " + ", ".join(
                f"{names_out[j]} max {dn[:, j].abs().max():.2e} bias {dn[:, j].mean():+.2e}" for j in top))
            dp = (oy - ry).abs().amax(0)
            topp = torch.argsort(dp, descending=True)[:4].tolist()
            print("    top variables by max |error| (physical units): "
                  + ", ".join(f"{names_out[j]} {dp[j]:.3g}" for j in topp))
            zl = [j for j, n in enumerate(names_out) if n.startswith("z_") or n in ("msl", "sp")]
            print(f"    mean signed error (physical) of geopotential/pressure: " + ", ".join(
                f"{names_out[j]} {(oy[:, j] - ry[:, j]).mean():+.3f}" for j in zl))
    if out:
        Path(out).write_text(json.dumps({"ref": ref_dir, "metrics": res}, indent=1) + "\n")
    return res


CHECKED = ("steps", "keys", "fast-fp32-vs-cpu", "fast-fp16-vs-cpu", "fast-fp16-vs-ref-fp16", "fp16_allowance",
           "fp16_mean_tolerance")  # what the tests read: --check compares these exactly


def calibrate(dirs: list[str], out: Path | None, fp16_ensemble: list[str] = (), check: str | None = None) -> dict:
    """dirs: the cpu-fp32-pyg, cuda-fp32-triton and cuda-fp16-triton rollout directories (in this order).

    Writes ``out`` (default: tolerances/rollout/<model>.json in this repository, <model> = MODEL),
    recording the checkpoint identity of the CPU reference rollout (its manifest's checkpoint) so the tests find
    the file by uuid. ``check``: exit 1 unless the tables the tests read (CHECKED) equal that file's exactly.
    """
    old = json.loads(Path(check).read_text()) if check else None  # read before anything is written
    ckpt = json.loads((Path(dirs[0]) / "manifest.json").read_text())["checkpoint"]
    ident = checkpoint_identity(ckpt)
    model = MODEL
    if out is None and check:  # --check must not overwrite the committed file it checks against
        out = Path(dirs[0]).parent / f"tolerances-check-{model}.json"
    out = Path(out) if out else ROLLOUT_TOLERANCES_DIR / f"{model}.json"
    paths = dict(zip(("cpu-fp32-pyg", "cuda-fp32-triton", "cuda-fp16-triton"), (Path(d) / "rollout.pt" for d in dirs)))
    load = lambda tag: torch.load(paths[tag], weights_only=True, mmap=True)  # noqa: E731
    cpu, g32, g16 = load("cpu-fp32-pyg"), load("cuda-fp32-triton"), load("cuda-fp16-triton")
    for o in (g32, g16):
        assert torch.equal(o["x0"], cpu["x0"]) and o["forcing_vars"] == cpu["forcing_vars"]
        assert torch.equal(o["forcings"], cpu["forcings"][:o["forcings"].shape[0]])
    n = min(t["y"].shape[0] for t in (cpu, g32, g16))
    ref = {"cuda-fp32-vs-cpu": step_metrics(cpu, g32, n), "cuda-fp16-vs-cpu": step_metrics(cpu, g16, n),
           "cuda-fp16-vs-cuda-fp32": step_metrics(g32, g16, n)}
    rel = lambda name, k: [e["rel_absmax"] for e in ref[name][k]]  # noqa: E731
    floor_exact, floor32 = 1e-6, 1e-5
    tol = {"fast-fp32-vs-cpu": {}, "fast-fp16-vs-cpu": {}, "fast-fp16-vs-ref-fp16": {}}
    for k in KEYS:
        tol["fast-fp32-vs-cpu"][k] = [round_up(max(floor32 if k.startswith("y") or s > 0 else floor_exact, 3 * r))
                                      for s, r in enumerate(rel("cuda-fp32-vs-cpu", k))]
        tol["fast-fp16-vs-cpu"][k] = [round_up(max(floor_exact, 1.75 * r)) for r in rel("cuda-fp16-vs-cpu", k)]
        tol["fast-fp16-vs-ref-fp16"][k] = [round_up(max(floor_exact, 2 * r)) for r in rel("cuda-fp16-vs-cuda-fp32", k)]
    v1 = {name: {k: list(v) for k, v in tol[name].items()} for name in ("fast-fp16-vs-cpu", "fast-fp16-vs-ref-fp16")}
    ensemble, dropped, extra = {}, [], {}
    if fp16_ensemble:  # v2: envelope over equally valid (neutrally perturbed) fp16 runs of the reference
        members = {}
        for d in fp16_ensemble:
            e = torch.load(Path(d) / "rollout.pt", weights_only=True, mmap=True)
            assert e["forcing_vars"] == cpu["forcing_vars"] and torch.equal(e["forcings"], cpu["forcings"][:e["forcings"].shape[0]])
            dx = (e["x0"].double() - cpu["x0"].double()).abs() / cpu["x0"].double().abs().clamp_min(1e-30)
            assert dx.max() <= 2.0 ** -23, (d, dx.max())  # at most one ulp from the golden input
            if torch.equal(e["y"], g16["y"][:e["y"].shape[0]]):
                dropped.append(str(d))  # bitwise the reference fp16 run: not a perturbation
                continue
            members[str(d)] = e
            n = min(n, e["y"].shape[0])
        for d, e in members.items():
            ensemble[d] = {"vs-cpu": step_metrics(cpu, e, n), "vs-cuda-fp16": step_metrics(g16, e, n)}
        f16 = 1.75
        # runs entering each envelope: vs cpu = the reference fp16 run + members; vs the reference fp16 = members
        runs = {"fast-fp16-vs-cpu": {"cuda-fp16-triton": ref["cuda-fp16-vs-cpu"], **{d: m["vs-cpu"] for d, m in ensemble.items()}},
                "fast-fp16-vs-ref-fp16": {d: m["vs-cuda-fp16"] for d, m in ensemble.items()}}
        allowance, mean_tol, loo = {}, {}, {}
        for name, rr_ in runs.items():
            r = {d: {k: [e["rel_absmax"] for e in m[k][:n]] for k in KEYS} for d, m in rr_.items()}
            for k in KEYS:
                env = [max(r[d][k][s] for d in r) for s in range(n)]
                t = [round_up(max(floor_exact, f16 * x)) for x in env]
                if name == "fast-fp16-vs-ref-fp16":
                    t = [max(a, b) for a, b in zip(v1[name][k], t)]
                t[0] = v1[name][k][0]  # step 0 = the one-step comparison: keep the strict one-step tolerance
                tol[name][k] = t
                mean_tol.setdefault(name, {})[k] = round_up(f16 * max(sum(r[d][k]) / n for d in r))
            # leave-one-out: worst ratio of each valid run to the 1.75x envelope of the OTHER runs (steps >= 1)
            loo[name] = {}
            for d in r:
                others = [o for o in r if o != d]
                w = max((r[d][k][s] / (f16 * max(r[o][k][s] for o in others)), k, s) for k in KEYS for s in range(1, n)
                        if max(r[o][k][s] for o in others) > 0)
                loo[name][d] = {"worst_ratio": w[0], "key": w[1], "step": w[2]}
            allowance[name] = round_up(max(1.0, max(v["worst_ratio"] for v in loo[name].values())))
        extra = {"fp16_allowance": allowance, "fp16_mean_tolerance": mean_tol, "fp16_leave_one_out": loo,
                 "fp16_criterion": ("fp16 checks (v2): step 0: rel <= the per-step tolerance (= the v1 one-step value); "
                                    "steps >= 1: rel <= fp16_allowance x the per-step tolerance; and per key, the mean "
                                    "over steps of rel <= fp16_mean_tolerance. Entries above the per-step tolerance "
                                    "alone are reported (valid runs have them too, see fp16_leave_one_out)."),
                 "fp16_ensemble_dropped_identical": dropped}
        for name in tol:
            for k in KEYS:
                tol[name][k] = tol[name][k][:n]
    doc = {
        "method": (
            "Per rollout step s (0-based; step s is the s-th model call), contract metric rel_absmax = max|a-r|/max|r| "
            "(float64) of: y (the step's physical output (1,1,1,G,72)), y_norm (y*mul_out+add_out in float64: every "
            "variable O(1)), window (the physical (1,2,G,84) window step s sees) and window_norm. The references are "
            "anemoi-inference rollouts of the original model (python/reference/run_rollout.py, identical input and forcings): "
            "cpu-fp32-pyg (ground truth), cuda-fp32-triton (runner precision '32'), cuda-fp16-triton (runner precision "
            "from the checkpoint: fp16 autocast, production). Error grows with the step, so tolerances are per step, "
            "with the one-step method of tolerances/<model>.json: fast-fp32-vs-cpu = max(1e-5, 3 x ref cuda-fp32 vs cpu) "
            "(window at step 0 = the input: 1e-6); fast-fp16-vs-cpu = 1.75 x ref cuda-fp16 vs cpu; "
            "fast-fp16-vs-ref-fp16 = 2 x ref cuda-fp16 vs ref cuda-fp32 (two independent roundings add); floors 1e-6; "
            "rounded up to 2 significant digits."
            + (" v2 (fp16 keys; v1 kept under 'v1'): one fp16 reference trajectory is not a noise floor for a chaotic "
               "rollout (its error vs CPU fluctuates by ~2x from step to step and between equally valid runs). The "
               "fp16 tolerances are calibrated on an ensemble of mathematically neutral perturbations of the "
               "reference fp16 rollout (listed under 'fp16_ensemble': ANEMOI_INFERENCE_NUM_CHUNKS_* summation "
               "grouping, the pyg attention backend, 1-ulp random-sign perturbations of the input): "
               "fast-fp16-vs-cpu = 1.75 x max over {reference fp16} + ensemble of rel vs cpu (per step, per key); "
               "fast-fp16-vs-ref-fp16 = max(v1 value, 1.75 x max over the ensemble of rel vs the reference fp16 "
               "rollout), step 0 kept at the v1 one-step value. A per-step max over a handful of chaotic "
               "trajectories is heavy-tailed: valid reference runs exceed the 1.75x envelope of the OTHER runs by up "
               "to fp16_leave_one_out's worst ratio, so steps >= 1 allow that factor (fp16_allowance), and the mean "
               "over steps (a stable statistic) must stay within 1.75 x the members' largest mean "
               "(fp16_mean_tolerance). See fp16_criterion and docs/rollout.md §4."
               if fp16_ensemble else "")),
        "model": model,
        "checkpoint": {k: ident.get(k) for k in ("uuid", "run_id", "size_bytes", "sha256", "path")},
        "steps": n, "keys": list(KEYS), **tol,
        **({"v1": v1, "fp16_ensemble": list(ensemble), **extra} if fp16_ensemble else {}),
        "reference": {name: {k: [{kk: e[kk] for kk in ("max_abs", "ref_absmax", "rel_absmax", "bitwise_equal")}
                                 for e in v] for k, v in d.items()} for name, d in ref.items()},
        "ensemble": {d: {c: {k: [e["rel_absmax"] for e in v] for k, v in m.items()} for c, m in em.items()}
                     for d, em in ensemble.items()},
        "sources": {t: str(p) for t, p in paths.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(out)
    print(f"{'step':>4s} " + " ".join(f"{n_:>26s}" for n_ in ref))
    for s in range(n):
        print(f"{s:4d} " + " ".join(f"{ref[n_]['y_norm'][s]['rel_absmax']:12.3e} {ref[n_]['y'][s]['rel_absmax']:12.3e} "
                                    for n_ in ref))
    print(f"wrote {out} (checkpoint uuid {ident['uuid']})")
    if check:
        new = json.loads(out.read_text())
        diff = [k for k in CHECKED if old.get(k) != new.get(k)]
        print(f"reproduces {check}: " + ("exactly (" + ", ".join(CHECKED) + ")" if not diff else f"NO, differs in {diff}"))
        if diff:
            sys.exit(1)
    return doc


# ------------------------------------------------------------------------------------------ main


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=rr.CKPT, help="inference checkpoint (default: the Hugging Face Varda forecaster)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--backend", default="pyg", choices=["pyg", "triton"])
    ap.add_argument("--precision", default="32",
                    help="runner precision: '32' (fp32; autocast(float32) is a no-op), '16-mixed', or 'checkpoint' "
                         "(the checkpoint's: 16-mixed -> fp16 autocast, production)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--date", default=DATE, help="date of the newest input slot (forcing dates follow the runner: "
                                                 "date + k x the checkpoint's timestep, 6 h for Varda)")
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--forcings-from", default=None, help="forcings.pt to replay (cross-checked against ComputedForcings)")
    ap.add_argument("--forcings-only", action="store_true",
                    help="run the runner's loop with a stub model (zeros) and write only forcings.pt + manifest.json")
    ap.add_argument("--runs", type=int, default=1,
                    help=">1: repeat the rollout in-process, report/require bitwise equality with run 0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate", nargs=3, default=None, metavar=("CPU32", "CUDA32", "CUDA16"),
                    help="the three reference rollout dirs (cpu-fp32-pyg, cuda-fp32-triton, cuda-fp16-triton): "
                         "write --tolerances")
    ap.add_argument("--tolerances", default=None,
                    help="with --calibrate: output file (default: tolerances/rollout/<model>.json in this repository; "
                         "with --check and no --tolerances: a temporary file next to the rollouts)")
    ap.add_argument("--check", default=None,
                    help="with --calibrate: exit 1 unless the tables equal this existing tolerances file's exactly")
    ap.add_argument("--fp16-ensemble", nargs="*", default=[],
                    help="with --calibrate: perturbed reference fp16 rollouts (equally valid runs); fp16 tolerances "
                         "become factor x the per-step envelope over them (docs/rollout.md)")
    ap.add_argument("--compare", nargs="+", default=None, metavar="DIR",
                    help="REF DIR...: per-step metrics of each rollout DIR against the rollout REF (table + --json)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-var", type=int, default=None, metavar="STEP", help="with --compare: per-variable errors at STEP")
    ap.add_argument("--allow-chunk-env", action="store_true",
                    help="permit ANEMOI_INFERENCE_NUM_CHUNKS* (a neutral perturbation: different summation grouping)")
    ap.add_argument("--perturb-ulp", type=int, default=None, metavar="SEED",
                    help="move every input value by one fp32 ulp up or down (random signs from SEED): a neutral "
                         "perturbation of the initial state")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.calibrate, a.tolerances, a.fp16_ensemble, a.check)
        return
    if a.compare:
        compare(a.compare[0], a.compare[1:], a.json, a.per_var)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    chunk_env = {k: os.environ.get(k) for k in rr.CHUNK_ENV}
    assert a.allow_chunk_env or all(v is None for v in chunk_env.values()), f"unset these: {chunk_env}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(a.device)
    x0 = torch.load(a.input, weights_only=True)
    if a.perturb_ulp is not None:
        g = torch.Generator().manual_seed(a.perturb_ulp)
        up = torch.rand(x0.shape, generator=g) < 0.5
        x0 = torch.where(up, torch.nextafter(x0, torch.tensor(float("inf"))),
                         torch.nextafter(x0, torch.tensor(float("-inf"))))
    date = datetime.datetime.fromisoformat(a.date)
    replay = torch.load(a.forcings_from, weights_only=True) if a.forcings_from else None
    t_start = time.perf_counter()

    if a.forcings_only:
        model, changed = None, []
        in_names = out_names = None
    else:
        model, changed = load_model(a.ckpt, device, a.backend)
        in_names, out_names = rr.variable_names(model)
    precision = None if a.precision == "checkpoint" else a.precision
    runner = make_runner(model if model is not None else torch.nn.Identity(), a.ckpt, device, precision)
    md = runner.tensor_handlers[DS].metadata
    if model is None:  # names and indices from the runner's metadata (stub run)
        v2i = md.variable_to_input_tensor_index
        in_names = sorted(v2i, key=v2i.get)
        o2v = md.output_tensor_index_to_variable
        out_names = [o2v[j] for j in range(len(o2v))]
        runner.stub_n_out = len(out_names)
    assert x0.shape[-1] == len(in_names), (x0.shape, len(in_names))
    timestep = runner.checkpoint.timestep
    print(f"checkpoint timestep {timestep}, multi_step_input {runner.checkpoint.multi_step_input}, "
          f"rollout_shift {runner.checkpoint.rollout_shift}, {len(in_names)} inputs, {len(out_names)} outputs", flush=True)
    if model is not None:
        norm = normaliser(model)
        inner = model.model
        prog_in = [int(i) for i in inner._internal_input_idx[DS]]
        prog_out = [int(i) for i in inner._internal_output_idx[DS]]
        assert [int(i) for i in md.prognostic_input_mask] == prog_in, "runner/model prognostic input indices differ"
        assert [int(i) for i in md.prognostic_output_mask] == prog_out, "runner/model prognostic output indices differ"
    else:
        norm = {}
        prog_in = [int(i) for i in md.prognostic_input_mask]
        prog_out = [int(i) for i in md.prognostic_output_mask]
    kept = sorted(n for n, i in md.variable_to_input_tensor_index.items() if md.typed_variables[n].is_constant_in_time)

    # Run 0 is kept; later runs are compared with it as they finish and dropped (memory: a 24-step Varda run is ~36 GB).
    tape = ForcingTape(runner, replay)
    times, det = [], {}
    y = windows = dates = forcings = None
    for r in range(a.runs):
        tape.calls = []
        yr, wr = run_inference(runner, x0, a.steps, date, in_names, out_names, record_windows=not a.forcings_only,
                               timer=times)
        dr, fr = tape.stacked()
        print(f"run {r}: {a.steps} steps in {times[-1]:.2f} s", flush=True)
        if r == 0:
            y, windows, dates, forcings = yr, wr, dr, fr
        else:
            det[f"run{r}_vs_run0_bitwise"] = bool(torch.equal(yr, y) and (wr is None or torch.equal(wr, windows)))
            if not det[f"run{r}_vs_run0_bitwise"]:
                det[f"run{r}_vs_run0_y_max_abs"] = float((yr.double() - y.double()).abs().max())
            del yr, wr
    if a.backend == "triton" or device.type == "cpu":  # pyg on CUDA uses scatter atomics: not deterministic
        assert all(v for k, v in det.items() if k.endswith("bitwise")), det
    if replay is not None:
        assert dates == replay["dates"][:len(dates)]
        assert torch.equal(forcings, replay["values"][:len(dates)]), "replayed forcings differ"
        print(f"forcings: {tape.n_crosschecked} provider calls, replay bitwise equal to ComputedForcings", flush=True)
    forc = {"dates": dates, "forcing_vars": tape.vars, "forcing_index": tape.index, "values": forcings,
            "source": a.forcings_from or "anemoi-inference ComputedForcings (recorded)"}
    if replay is None:
        save_atomic(forc, out / "forcings.pt")
    info = {"runner": f"{type(runner).__mro__[1].__module__}.{type(runner).__mro__[1].__name__}",
            "runner_autocast_dtype": str(runner.autocast), "runner_precision_config": precision,
            "autocast_effective": ("disabled (CPU autocast supports only fp16/bf16; torch warns)"
                                   if device.type == "cpu" and runner.autocast not in (torch.float16, torch.bfloat16)
                                   else f"torch.autocast({device.type!r}, {runner.autocast})"),
            "checkpoint_timestep": str(timestep), "kept_constant_in_time": kept, "determinism": det,
            "forcings_crosschecked_calls": tape.n_crosschecked, "rollout_s": times}
    if a.forcings_only:
        info["forcings_only"] = True
        manifest = {"script": "python/reference/run_rollout.py", "command": " ".join(["python", *sys.argv]),
                    "checkpoint": a.ckpt, "steps": a.steps, "start_date": a.date, "forcing_vars": tape.vars,
                    "forcing_index": tape.index, "dates": dates, "versions": versions(),
                    "wall_s": time.perf_counter() - t_start, **info}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {out} (forcings only: {len(dates)} dates, {dates[0]} .. {dates[-1]})")
        return

    roll = {"x0": x0, "y": y, "window": windows}
    roll.update({"forcings": forc["values"][:max(a.steps - 1, 0)].clone(), "forcing_vars": forc["forcing_vars"],
                 "forcing_index": forc["forcing_index"], "dates": forc["dates"][:max(a.steps - 1, 0)],
                 "prog_in": prog_in, "prog_out": prog_out, "kept_vars": kept, "in_vars": list(in_names),
                 "out_vars": list(out_names), **norm})
    save_atomic(roll, out / "rollout.pt")
    manifest = {
        "script": "python/reference/run_rollout.py", "command": " ".join(["python", *sys.argv]), "mode": "inference",
        "checkpoint": a.ckpt, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "backend": a.backend,
        "backend_override_blocks": len(changed),
        "precision": a.precision, "steps": a.steps, "start_date": a.date, "input": a.input,
        "perturb_ulp_seed": a.perturb_ulp,
        "forcings_from": a.forcings_from, "forcing_vars": forc["forcing_vars"], "forcing_index": forc["forcing_index"],
        "torch_num_threads": torch.get_num_threads(), "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "chunk_env": chunk_env, "versions": versions(),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in roll.items() if isinstance(v, torch.Tensor)},
        "y_finite": bool(torch.isfinite(roll["y"]).all()),
        "y_absmax_per_step": [float(t.abs().max()) for t in roll["y"]],
        "peak_rss_gb": rr.peak_rss_gb(),
        **({"cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3} if device.type == "cuda" else {}),
        "wall_s": time.perf_counter() - t_start, **info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("device", "precision", "steps", "y_finite", "peak_rss_gb", "wall_s")}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
