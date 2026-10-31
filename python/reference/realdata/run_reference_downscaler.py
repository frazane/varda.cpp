"""Run the ORIGINAL Varda temporal downscaler (varda-td) on the forecaster reference outputs, with anemoi-inference's
own `temporal_downscaler` runner and the Hugging Face config, and dump what the model receives and produces.

Inputs: the GRIB files a forecaster reference run left behind (docs/realdata-reference.md: `<run>/outputs/20*.grib`
for the LAM and `<run>/outputs/global*.grib`, steps 0 .. 24 h every 6 h). They are symlinked into a work directory
laid out as the HF config expects (`outputs/`, `templates/`, `varda-temporal-downscaler-det-sgm/`). Nothing is
fetched: the downscaler needs no network. The runner is built programmatically,
`create_runner(RunConfiguration.load(<HF inference_opendata.yaml>, overrides))` then `runner.execute()`, with one
deliberate difference from the HF config: `runner: temporal_downscaler` instead of
`runner: {parallel: {base_runner: temporal_downscaler, cluster: {manual: 1}}}` (one device, as for the forecaster
reference). Environment: the notebook's (ECCODES_DEFINITION_PATH from eccodes_cosmo_resources,
ECCODES_VERSION_CHECK_OFF=1, ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
python -O).

Outputs (<out>, written under a hidden staging name and renamed when complete):
  model_io.safetensors   x.<k> [2, G, 77] F32 (model input of window k, model-input channel order, physical units,
                         slot 0 = t, slot 1 = t+6h), y.<k> [6, G, 67] F32 (raw model output, before any
                         post-processor, hours 1..6), forcdyn.<k> [2, G, 5] F32 (the dynamic forcing channels of x.<k>);
                         metadata: variable orders, dates, uuid, run config, versions, source hashes
  tp_stages.safetensors  tp [G] F32 at every post-processing stage of every state the runner handles:
                         call<n>.pre / .acc / .post (top-level post-processors: Accumulate, rescale) and
                         written.<n> (the LAM state just before GRIB encoding, after extract_from_state and
                         copy-prognostic-from-forecaster)
  outputs/               the forecaster inputs (symlinks) and the hourly GRIB files varda-single_*.grib
  dumps/                 source/cutout/constant states of window 0, the first written states in full
  manifest.json, config.yaml

Usage (GPU node, via job_reference_downscaler.sh):
    $PY -O -I python/reference/realdata/run_reference_downscaler.py --date 2026-10-22T15 \
        --forecaster-run $VARDA_SCRATCH/realdata/20261022T15/ref-fp16 --device cuda --precision checkpoint \
        --out $VARDA_SCRATCH/downscale/ref/20261022T15/fp16
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# the notebook's environment is set by this import, before anything imports eccodes
import run_reference_realdata as rr  # noqa: E402  (HF paths, Recorder, versions, cpu_info, sha256, atomic_json)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from safetensors_io import write_safetensors  # noqa: E402

VARDA_SCRATCH = rr.VARDA_SCRATCH
HF = rr.HF
HF_TD = HF / "varda-temporal-downscaler-det-sgm"
HF_CONFIG = HF_TD / "configs" / "inference" / "inference_opendata.yaml"
TD_UUID = "00e40f03-2f7e-4c1c-8e6d-91ed9062249f"  # HF varda-temporal-downscaler-det-sgm/model.ckpt
DYN = ("cos_julian_day", "cos_local_time", "insolation", "sin_julian_day", "sin_local_time")


def install_hooks(runner, rec, state: dict, args) -> None:
    from anemoi.inference.inputs.cutout import Cutout
    from anemoi.inference.outputs.grib import BaseGribOutput

    tt = rr.tt

    # 1. per-source and cutout states of window 0 (prognostics: the GRIB-decoded forecaster states; constants once)
    for purpose, inputs in (("prognostics", runner.prognostics_inputs),
                            ("constant_forcings", runner.constant_forcings_inputs),
                            ("dynamic_forcings", runner.dynamic_forcings_inputs)):
        for ds, inp in inputs.items():
            if not isinstance(inp, Cutout):
                state["input_kinds"][purpose] = repr(inp)
                continue
            counter = {"n": 0}
            for src, sub in inp.sources.items():
                orig = sub.create_input_state

                def create_input_state(*a, _o=orig, _p=purpose, _s=src, _sub=sub, _c=counter, **k):
                    st = _o(*a, **k)
                    if _c["n"] == 0 or args.dump_all_windows:
                        w = _c["n"] if args.dump_all_windows else 0
                        rec.save_fields(f"source.{_p}.{_s}.w{w}", st, meta={"input": repr(_sub), "kwargs": {
                            kk: str(vv) for kk, vv in k.items()}})
                    return st

                sub.create_input_state = create_input_state
            orig_c = inp.create_input_state

            def create_cutout_state(*a, _o=orig_c, _p=purpose, _inp=inp, _c=counter, **k):
                st = _o(*a, **k)
                if _c["n"] == 0 or args.dump_all_windows:
                    w = _c["n"] if args.dump_all_windows else 0
                    masks = {f"mask/{s}": np.asarray(m) for s, m in st.get("_mask", {}).items()}
                    rec.save_fields(f"cutout.{_p}.w{w}", st, extra=masks, meta={
                        "dates": [str(d) for d in k.get("dates", [])], "kwargs": {kk: str(vv) for kk, vv in k.items()
                                                                                    if kk != "dates"},
                        "sources": list(_inp.sources)})
                _c["n"] += 1
                return st

            inp.create_input_state = create_cutout_state

    # 2. computed forcings
    for ds, h in runner.tensor_handlers.items():
        for kind, providers in (("constant", h.constant_forcings_providers), ("dynamic", h.dynamic_forcings_providers)):
            for p in providers:
                orig_l = p.load_forcings_array

                def load_forcings_array(dates, st, _o=orig_l, _p=p, _kind=kind):
                    a = _o(dates, st)
                    i = rec.next(f"forcings.{_kind}")
                    rec.save(f"forcings.{_kind}.{i}", {"values": np.asarray(a)},
                             {"provider": repr(_p), "variables": list(_p.variables), "mask": [int(m) for m in _p.mask],
                              "dates": [d.isoformat() for d in dates], "layout": "[variables, dates, values]"})
                    return a

                p.load_forcings_array = load_forcings_array

    # 3. the model call of each window
    cls = type(runner)
    orig_predict = cls.predict_step
    h0 = next(iter(runner.tensor_handlers.values()))
    in_names = [None] * len(h0.metadata.variable_to_input_tensor_index)
    for v, i in h0.metadata.variable_to_input_tensor_index.items():
        in_names[i] = v
    state["in_vars"] = in_names
    state["out_vars"] = None
    dyn_idx = [in_names.index(v) for v in DYN]
    state["dyn_idx"] = dyn_idx

    def predict_step(self, model, input_tensors_torch, **kwargs):
        k = len(state["predict"])
        (ds, x), = input_tensors_torch.items()
        xc = x.detach().float().cpu()[0].contiguous()  # [2, G, 77]
        state["x"][k] = xc
        state["forcdyn"][k] = xc[:, :, dyn_idx].contiguous()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = orig_predict(self, model, input_tensors_torch, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        yt = y[ds]
        h = self.tensor_handlers[ds]
        state["out_vars"] = [h.metadata.output_tensor_index_to_variable[j] for j in range(yt.shape[-1])]
        state["y"][k] = yt.detach().float().cpu()[0, :, 0].contiguous()  # [6, G, 67]
        state["predict"].append({"window": k, "seconds": seconds, "x_dtype": str(x.dtype), "y_dtype": str(yt.dtype),
                                 "y_shape": list(yt.shape), "kwargs": {kk: str(v) for kk, v in kwargs.items()},
                                 "nan_y": int(torch.isnan(yt).sum()), "nan_x": int(torch.isnan(x).sum()),
                                 "autocast": str(torch.get_autocast_dtype(self.device.type))
                                 if torch.is_autocast_enabled(self.device.type) else None})
        print(f"[predict] window {k}: {seconds:.3f} s, y {tuple(yt.shape)} {yt.dtype}", flush=True)
        return y

    cls.predict_step = predict_step

    # 4. top-level post-processors: tp before/after each one, for every call
    for ds, procs in runner.post_processors.items():
        for i, p in enumerate(procs):
            orig_pp = p.process

            def process(st, _o=orig_pp, _i=i, _p=p):
                if _i == 0:
                    state["calls"].append({"n": len(state["calls"]), "step": str(st.get("step")),
                                           "date": str(st.get("date")), "fields": sorted(st["fields"]),
                                           "has_tp": "tp" in st["fields"], "stages": {}})
                    if "tp" in st["fields"]:
                        state["tp"][f"call{len(state['calls']) - 1}.pre"] = tt(np.array(st["fields"]["tp"], copy=True))
                n = len(state["calls"]) - 1
                before = {k: np.array(v, copy=True) for k, v in st["fields"].items()}
                out = _o(st)
                changed = [k for k, v in out["fields"].items()
                           if k not in before or not np.array_equal(np.asarray(v), before[k], equal_nan=True)]
                added = [k for k in out["fields"] if k not in before]
                state["calls"][n]["stages"][type(_p).__name__] = {"changed": changed, "added": added, "repr": repr(_p)}
                if "tp" in out["fields"]:
                    state["tp"][f"call{n}.{'acc' if _i == 0 else 'post'}"] = tt(np.array(out["fields"]["tp"], copy=True))
                return out

            p.process = process

    # 5. what the GRIB writer encodes (after the output's own post-processors)
    orig_write = BaseGribOutput.write_step

    def write_step(self, st):
        n = len(state["written"])
        info = {"n": n, "step": str(st.get("step")), "date": str(st.get("date")), "previous_step": str(st.get("previous_step")),
                "start_steps": {k: str(v) for k, v in st.get("start_steps", {}).items()},
                "fields": sorted(st["fields"]), "path": str(getattr(self, "out", ""))}
        if "tp" in st["fields"]:
            state["tp"][f"written.{n}"] = tt(np.array(st["fields"]["tp"], copy=True))
        state["written"].append(info)
        if n < args.dump_written_full:
            rec.save_fields(f"written.{n}", st, meta=info)
        return orig_write(self, st)

    BaseGribOutput.write_step = write_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="reference date t of the forecaster run, e.g. 2026-10-22T15")
    ap.add_argument("--forecaster-run", required=True, help="forecaster reference run dir (its outputs/ hold the inputs)")
    ap.add_argument("--precision", default="checkpoint", help="'checkpoint' (runner default: fp16 autocast) or 32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lead-time", default="1d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-all-windows", action="store_true", help="source/cutout dumps for every window")
    ap.add_argument("--dump-written-full", type=int, default=7,
                    help="dump the first N written states with all fields (default: the 7 states of window 0)")
    ap.add_argument("--attention-backend", default=None, choices=["pyg", "triton"],
                    help="override the pickled graph-transformer backend (CPU needs pyg: triton has no CPU driver)")
    ap.add_argument("--no-tf32", action="store_true", help="matmul and cudnn TF32 off")
    ap.add_argument("--no-dumps", action="store_true", help="no hooks, no dumps (timing baseline)")
    args = ap.parse_args()

    date = dt.datetime.fromisoformat(args.date)
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists")
    stage = out.with_name(f".{out.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    fc_out = Path(args.forecaster_run).resolve() / "outputs"
    (stage / "outputs").mkdir()
    inputs = sorted(list(fc_out.glob("20*.grib")) + list(fc_out.glob("global*.grib")))
    if len(inputs) != 10:
        sys.exit(f"expected 10 forecaster files in {fc_out}, found {len(inputs)}")
    for p in inputs:
        os.symlink(p, stage / "outputs" / p.name)
    for name, target in (("varda-temporal-downscaler-det-sgm", HF_TD), ("templates", HF / "templates")):
        os.symlink(target, stage / name)
    os.chdir(stage)

    cfg = yaml.safe_load(HF_CONFIG.read_text())
    cfg["runner"] = "temporal_downscaler"
    overrides: dict = {"date": date.isoformat(), "lead_time": args.lead_time, "device": args.device}
    if args.precision != "checkpoint":
        overrides["precision"] = args.precision
    (stage / "config.yaml").write_text(yaml.safe_dump({"base": str(HF_CONFIG), "resolved": cfg,
                                                       "overrides": overrides}, sort_keys=False))
    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners import create_runner

    rec = rr.Recorder(stage)
    state = {"x": {}, "y": {}, "forcdyn": {}, "predict": [], "calls": [], "written": [], "tp": {}, "input_kinds": {}}
    t0 = time.perf_counter()
    config = RunConfiguration.load(cfg, [overrides])
    runner = create_runner(config)
    from anemoi.models.layers.block import GraphTransformerBaseBlock

    blocks = [m for m in runner.model.modules() if isinstance(m, GraphTransformerBaseBlock)]
    if args.attention_backend:
        from anemoi.models.layers.block import GraphTransformerConv

        for m in blocks:
            m.graph_attention_backend = args.attention_backend
            if args.attention_backend == "pyg":  # what the constructor does when triton is not available
                m.conv = GraphTransformerConv(out_channels=m.out_channels_conv)
    backends = sorted({m.graph_attention_backend for m in blocks})
    print(f"[backend] graph attention backends: {backends} ({len(blocks)} blocks)", flush=True)
    if not args.no_dumps:
        install_hooks(runner, rec, state, args)
    t1 = time.perf_counter()
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    failed = None
    try:
        runner.execute()
    except BaseException as e:  # noqa: BLE001  keep what we have
        failed = repr(e)
        import traceback
        traceback.print_exc()
    t2 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() if runner.device.type == "cuda" else None

    ck = runner.checkpoint
    uuid = TD_UUID  # not exposed by the checkpoint object; the sha256 in the manifest identifies the file
    meta = {
        "varda_downscaler_io": "1", "date": date.isoformat(), "uuid": str(uuid),
        "checkpoint": str(HF_TD / "model.ckpt"),
        "in_vars": json.dumps(state.get("in_vars")), "out_vars": json.dumps(state.get("out_vars")),
        "dyn_idx": json.dumps(state.get("dyn_idx")), "dyn_vars": json.dumps(list(DYN)),
        "windows": json.dumps([{"k": k, "date_t": (date + dt.timedelta(hours=6 * k)).isoformat(),
                                "date_t6": (date + dt.timedelta(hours=6 * k + 6)).isoformat(),
                                "output_dates": [(date + dt.timedelta(hours=6 * k + h)).isoformat() for h in range(1, 7)]}
                               for k in sorted(state["x"])]),
        "layout": json.dumps({"x": "[slot(t,t+6h), G, 77]", "y": "[hour 1..6, G, 67]", "forcdyn": "[slot, G, 5]"}),
        "precision_arg": args.precision, "autocast": str(runner.autocast), "device": str(runner.device),
        "tf32": json.dumps({"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}),
        "versions": json.dumps(rr.versions()),
        "source_sha256": json.dumps({p.name: rr.sha256(p) for p in inputs}),
    }
    tensors = {}
    for k in sorted(state["x"]):
        tensors[f"x.{k}"], tensors[f"forcdyn.{k}"] = state["x"][k], state["forcdyn"][k]
        if k in state["y"]:
            tensors[f"y.{k}"] = state["y"][k]
    if tensors:
        write_safetensors(stage / "model_io.safetensors", tensors, meta)
    if state["tp"]:
        write_safetensors(stage / "tp_stages.safetensors", state["tp"],
                          {"calls": json.dumps(state["calls"]), "written": json.dumps(state["written"])})

    hourly = sorted((stage / "outputs").glob("varda-single_*.grib"))
    manifest = {
        "kind": "varda temporal downscaler reference run", "date": date.isoformat(), "argv": sys.argv,
        "failed": failed, "forecaster_run": str(Path(args.forecaster_run).resolve()),
        "runner_class": type(runner).__name__, "runner_precision": runner.precision,
        "checkpoint_precision": ck.precision, "autocast": str(runner.autocast), "device": str(runner.device),
        "device_name": torch.cuda.get_device_name(0) if runner.device.type == "cuda" else rr.platform.processor(),
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "graph_attention_backends": backends, "torch_threads": torch.get_num_threads(), "python_flags_optimize": sys.flags.optimize,
        "env": {k: os.environ.get(k) for k in ("ECCODES_DEFINITION_PATH", "ECCODES_VERSION_CHECK_OFF",
                                                 "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER", "PYTORCH_CUDA_ALLOC_CONF",
                                                 "SLURM_JOB_ID")},
        "host": socket.gethostname(), "cpu": rr.cpu_info(), "versions": rr.versions(),
        "seconds": {"setup": t1 - t0, "execute": t2 - t1, "predict_total": sum(p["seconds"] for p in state["predict"])},
        "predict": state["predict"], "peak_cuda_memory_allocated_bytes": peak,
        "input_kinds": state["input_kinds"],
        "calls": state["calls"], "written": state["written"],
        "inputs": {p.name: {"bytes": os.stat(p).st_size, "sha256": rr.sha256(p), "source": str(p)} for p in inputs},
        "checkpoint_sha256": rr.sha256(HF_TD / "model.ckpt"),
        "hourly_grib_outputs": {p.name: {"bytes": p.stat().st_size, "sha256": rr.sha256(p)} for p in hourly},
        "dumps": rec.index,
        "model_io": "model_io.safetensors" if tensors else None,
    }
    rr.atomic_json(stage / "manifest.json", manifest)
    os.chdir(stage.parent)
    os.rename(stage, out)
    print(f"done{' (FAILED: ' + failed + ')' if failed else ''}: {len(hourly)} hourly files, "
          f"{len(state['predict'])} windows, execute {t2 - t1:.1f} s -> {out}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
