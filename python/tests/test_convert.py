"""Checks of a converted model.safetensors (docs/format.md section 1).

    VARDA_CONVERTED=/path/model.safetensors [VARDA_WEIGHTS_PT=/path/weights.pt] \
        .venv/bin/python -I python/tests/test_convert.py

VARDA_CONVERTED may list several files separated by ':'. VARDA_WEIGHTS_PT (same count, same order, or
unset) enables the bitwise comparison with the independent weight export weights.pt, and config.json next to it
is compared too; an empty entry (e.g. `a/weights.pt::c/weights.pt`) skips that file (the Varda temporal downscaler
has no independent export). VARDA_CKPTS (same count/order) enables the check of grid.* against the zip members.
Without VARDA_CONVERTED the tests are skipped. The restricted-unpickler and converter-helper tests need no files.

Interpolator files (architecture.model.kind "enc_proc_dec_interpolator") are checked for the `times` section, the
multi-step decoder head (dec.out_proj = n_output_steps * n_output_vars rows) and, for the Varda temporal downscaler
(uuid TD_UUID), the values read from its checkpoint.
"""

import io
import json
import os
import pickle
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import checkpoint as ck  # noqa: E402
import safetensors_io as sio  # noqa: E402

FILES = [p for p in os.environ.get("VARDA_CONVERTED", "").split(":") if p]
WEIGHTS = os.environ.get("VARDA_WEIGHTS_PT", "").split(":") if os.environ.get("VARDA_WEIGHTS_PT") else []
TD_UUID = "00e40f03-2f7e-4c1c-8e6d-91ed9062249f"  # Varda-single-1.0 temporal downscaler (anemoi.json uuid)
INTERPOLATOR = "enc_proc_dec_interpolator"


def expected_names(cfg):
    """Tensor names of the contract table (docs/format.md 1.1) from the config."""
    names = {"norm.mul", "norm.add", "nodes.data.attr", "nodes.hidden.attr"}
    for g in ("enc", "proc", "dec"):
        names |= {f"graph.{g}.edge_index", f"graph.{g}.edge_attr"}
    mods = ["enc.emb_src", "enc.emb_dst", "dec.emb_dst", "dec.out_norm", "dec.out_proj"]
    for prefix, b in cfg["blocks"].items():
        sfx = ["q", "k", "v", "self", "edge", "proj", "ln_dst", "ln_mlp", "mlp1", "mlp2"]
        if b["kind"] == "graph_transformer_mapper_block":
            sfx.append("ln_src")
        mods += [f"{prefix}.{s}" for s in sfx]
    for m in mods:
        names |= {f"{m}.weight", f"{m}.bias"}
    return names - set(cfg["absent"])


GRID_NAMES = {"grid.latitudes", "grid.longitudes"}  # F64 [G]: coordinates anemoi-inference uses (docs/forcings.md)


def grid_names(header):
    """GRID_NAMES plus the cutout masks `grid.<source>.cutout_mask` named by the `cutout` metadata (docs/format.md 1.1)."""
    cut = json.loads(header["__metadata__"].get("cutout", "[]"))
    return GRID_NAMES | {f"grid.{n}.cutout_mask" for n in cut}


def config_of(header):
    return json.loads(header["__metadata__"]["config"])


@unittest.skipUnless(FILES, "VARDA_CONVERTED not set")
class ConvertedFile(unittest.TestCase):
    def each(self):
        for i, p in enumerate(FILES):
            with self.subTest(file=p):
                yield i, p

    def test_container(self):
        for _, p in self.each():
            header, start = sio.read_header(p)  # contiguous, aligned, sizes match shapes
            self.assertEqual(start % 8, 0)
            meta = header["__metadata__"]
            self.assertEqual(sorted(set(meta) - {"cutout"}), ["config", "varda_format"])
            if "cutout" in meta:  # source order of the data grid; one mask per source
                self.assertEqual(sorted(k for k in header if k.endswith(".cutout_mask")),
                                 sorted(f"grid.{n}.cutout_mask" for n in json.loads(meta["cutout"])))
            self.assertEqual(header["__metadata__"]["varda_format"], "1")
            self.assertTrue(all(isinstance(v, str) for v in header["__metadata__"].values()))
            names = [k for k in header if k != "__metadata__"]
            order = sorted(names, key=lambda n: header[n]["data_offsets"][0])
            self.assertEqual(order, sorted(names, key=lambda n: (header[n]["dtype"] not in ("I64", "F64"), n)))

    def test_config_fields(self):
        for _, p in self.each():
            header, _ = sio.read_header(p)
            c = json.loads(header["__metadata__"]["config"])
            self.assertEqual(c["format_version"], 3)
            for k in ("checkpoint", "uuid", "run_id", "sha256", "size_bytes"):
                self.assertIn(k, c["source"])
            self.assertEqual(len(c["source"]["sha256"]), 64)
            a = c["architecture"]
            self.assertIn(a["model"]["kind"], ("enc_proc_dec", INTERPOLATOR))
            self.assertEqual(a["encoder"]["kind"], "graph_transformer_forward_mapper")
            self.assertEqual(a["processor"]["kind"], "graph_transformer_processor")
            self.assertEqual(a["decoder"]["kind"], "graph_transformer_backward_mapper")
            d = c["dims"]
            for k in ("num_channels", "num_heads", "head_dim", "mlp_hidden", "num_processor_layers", "n_input_steps",
                      "n_output_steps", "n_data_vars", "n_input_vars", "n_output_vars", "n_data_nodes",
                      "n_hidden_nodes", "node_attr_dim", "edge_attr_dim"):
                self.assertIsInstance(d[k], int, k)
            self.assertEqual(d["num_heads"] * d["head_dim"], d["num_channels"])
            self.assertIsInstance(d["layer_norm_eps"], float)
            self.assertIn(d["gelu_approximate"], ("none", "tanh"))
            self.assertIsInstance(d["latent_skip"], bool)
            if a["model"]["kind"] == INTERPOLATOR:
                self.assertGreaterEqual(d["n_output_steps"], 1)
                self.assertIn("times", c)
            else:  # the forecaster: one output step, no times section (its config is unchanged)
                self.assertEqual(d["n_output_steps"], 1)
                self.assertNotIn("times", c)
            v = c["variables"]
            self.assertEqual(len(v["data"]), d["n_data_vars"])
            self.assertEqual(len(v["input"]), d["n_input_vars"])
            self.assertEqual(len(v["output"]), d["n_output_vars"])
            ix = c["indices"]
            self.assertEqual(len(ix["norm_input"]), d["n_input_vars"])
            self.assertEqual(len(ix["norm_output"]), d["n_output_vars"])
            self.assertEqual(len(ix["residual_input"]), len(ix["residual_output"]))
            for k, n in (("norm_input", d["n_data_vars"]), ("norm_output", d["n_data_vars"]),
                         ("residual_input", d["n_input_vars"]), ("residual_output", d["n_output_vars"])):
                self.assertTrue(all(0 <= i < n for i in ix[k]), k)
            self.assertEqual(c["residual"]["type"], "skip")
            self.assertIn(c["residual"]["step"], range(-d["n_input_steps"], d["n_input_steps"]))
            for b in c["bounding"]:
                self.assertIn(b["type"], ("relu", "hardtanh", "fraction"))
                self.assertTrue(all(0 <= i < d["n_output_vars"] for i in b["index"]))
                if b["type"] != "relu":
                    self.assertIsInstance(b["min"], float)
                    self.assertIsInstance(b["max"], float)
                if b["type"] == "fraction":
                    self.assertIn(len(b["total_index"]), (1, len(b["index"])))
            for g in ("enc", "dec"):
                self.assertGreaterEqual(c["mappers"][g]["num_chunks"], 1)
            order = ["enc.block"] + [f"proc.{i}" for i in range(d["num_processor_layers"])] + ["dec.block"]
            self.assertEqual(sorted(c["blocks"]), sorted(order))
            for n in order:
                want = "graph_transformer_mapper_block" if n in ("enc.block", "dec.block") \
                    else "graph_transformer_processor_block"
                self.assertEqual(c["blocks"][n]["kind"], want)
            self.assertIsInstance(c["absent"], list)
            self.assertTrue(all(isinstance(u["path"], str) and u["reason"] for u in c["unused"]))

    def test_forcings_section(self):
        for _, p in self.each():
            header, _ = sio.read_header(p)
            c = json.loads(header["__metadata__"]["config"])
            f = c["forcings"]
            self.assertEqual(sorted(f), ["constant", "constant_input_index", "dynamic", "dynamic_input_index",
                                         "source", "timestep_hours"])
            inp = c["variables"]["input"]
            self.assertEqual([inp[i] for i in f["dynamic_input_index"]], f["dynamic"])
            self.assertEqual([inp[i] for i in f["constant_input_index"]], f["constant"])
            self.assertEqual(f["dynamic_input_index"], sorted(f["dynamic_input_index"]))
            self.assertEqual(f["constant_input_index"], sorted(f["constant_input_index"]))
            self.assertEqual(f["dynamic"], ["cos_julian_day", "cos_local_time", "insolation", "sin_julian_day",
                                            "sin_local_time"])
            prog = {inp[i] for i in c["indices"]["residual_input"]}
            self.assertEqual(prog | set(f["dynamic"]) | set(f["constant"]), set(inp))  # a partition
            self.assertEqual(len(prog) + len(f["dynamic"]) + len(f["constant"]), len(inp))
            self.assertIsInstance(f["timestep_hours"], int)
            self.assertGreater(f["timestep_hours"], 0)

    def test_names_shapes(self):
        for _, p in self.each():
            header, _ = sio.read_header(p)
            c = json.loads(header["__metadata__"]["config"])
            d = c["dims"]
            names = {k for k in header if k != "__metadata__"}
            self.assertEqual(names, expected_names(c) | grid_names(header))
            C = d["num_channels"]
            for g in ("enc", "proc", "dec"):
                ei, ea = header[f"graph.{g}.edge_index"], header[f"graph.{g}.edge_attr"]
                self.assertEqual(ei["dtype"], "I64")
                self.assertEqual(ei["shape"][0], 2)
                self.assertEqual(ea["dtype"], "F32")
                self.assertEqual(ea["shape"], [ei["shape"][1], d["edge_attr_dim"]])
            self.assertEqual(header["nodes.data.attr"]["shape"], [d["n_data_nodes"], d["node_attr_dim"]])
            self.assertEqual(header["nodes.hidden.attr"]["shape"], [d["n_hidden_nodes"], d["node_attr_dim"]])
            self.assertEqual(header["norm.mul"]["shape"], [d["n_data_vars"]])
            for n in sorted(GRID_NAMES):
                self.assertEqual(header[n]["dtype"], "F64", n)
                self.assertEqual(header[n]["shape"], [d["n_data_nodes"]], n)
            self.assertEqual(header["dec.out_proj.weight"]["shape"], [d["n_output_steps"] * d["n_output_vars"], C])
            self.assertEqual(header["enc.emb_src.weight"]["shape"],
                             [C, d["n_input_steps"] * d["n_input_vars"] + d["node_attr_dim"]])
            self.assertEqual(header["proc.0.q.weight"]["shape"], [C, C])
            self.assertEqual(header["proc.0.mlp1.weight"]["shape"], [d["mlp_hidden"], C])
            for n in sorted(grid_names(header) - GRID_NAMES):
                self.assertEqual(header[n]["dtype"], "U8", n)
            for n, h in header.items():
                if n != "__metadata__" and n not in grid_names(header):
                    self.assertEqual(h["dtype"], "I64" if n.endswith("edge_index") else "F32", n)

    def test_grid_coordinates_agree_with_node_attr(self):
        for _, p in self.each():
            _, t = sio.read_safetensors(p)
            lat, lon = np.asarray(t["grid.latitudes"]), np.asarray(t["grid.longitudes"])
            self.assertTrue(np.isfinite(lat).all() and np.isfinite(lon).all())
            self.assertLessEqual(np.abs(lat).max(), 90.0)
            ref = np.stack([np.sin(np.deg2rad(lat)), np.sin(np.deg2rad(lon)),
                            np.cos(np.deg2rad(lat)), np.cos(np.deg2rad(lon))], 1)
            self.assertLess(np.abs(ref - np.asarray(t["nodes.data.attr"], np.float64)).max(), 1e-6)

    def test_interpolator(self):
        """Interpolator files: times, decoder head, and the Varda downscaler's values (read from its checkpoint)."""
        n = 0
        for _, p in self.each():
            header, _ = sio.read_header(p)
            c = config_of(header)
            if c["architecture"]["model"]["kind"] != INTERPOLATOR:
                continue
            n += 1
            d, t = c["dims"], c["times"]
            self.assertEqual(sorted(t), ["frequency_hours", "input", "target"])
            self.assertEqual(len(t["input"]), d["n_input_steps"])
            self.assertEqual(len(t["target"]), d["n_output_steps"])
            for k in ("input", "target"):
                self.assertTrue(all(isinstance(v, int) for v in t[k]))
                self.assertEqual(t[k], sorted(set(t[k])), k)
            self.assertIsInstance(t["frequency_hours"], int)
            self.assertGreater(t["frequency_hours"], 0)
            self.assertEqual(c["forcings"]["timestep_hours"], t["frequency_hours"])
            self.assertEqual(header["dec.out_proj.weight"]["shape"], [d["n_output_steps"] * d["n_output_vars"],
                                                                      d["num_channels"]])
            self.assertEqual(header["dec.out_proj.bias"]["shape"], [d["n_output_steps"] * d["n_output_vars"]])
            if c["source"]["uuid"] == TD_UUID:
                self.assertEqual(c["architecture"]["model"]["class"],
                                 "anemoi.models.models.interpolator.AnemoiModelEncProcDecMultiOutInterpolator")
                self.assertEqual(t, {"input": [0, 6], "target": [1, 2, 3, 4, 5, 6], "frequency_hours": 1})
                self.assertEqual((d["n_input_steps"], d["n_output_steps"], d["n_input_vars"], d["n_output_vars"],
                                  d["n_data_vars"], d["num_processor_layers"], d["num_channels"], d["latent_skip"]),
                                 (2, 6, 77, 67, 78, 16, 1024, False))
                self.assertEqual(header["dec.out_proj.weight"]["shape"], [402, 1024])
                self.assertEqual(header["enc.emb_src.weight"]["shape"], [1024, 158])
                self.assertEqual(c["residual"], {"type": "skip", "step": 0})
                self.assertEqual([(b["type"], [c["variables"]["output"][i] for i in b["index"]]) for b in c["bounding"]],
                                 [("relu", ["tp"])])
                self.assertEqual(c["forcings"]["constant"], ["cos_latitude", "cos_longitude", "lsm", "sin_latitude",
                                                             "sin_longitude", "z"])
                self.assertEqual(c["source"]["sha256"],
                                 "452fda53b073579f60deaeee666bc5434ed20c91ad4506a37fbbd1d343a21d0e")
        if not n:
            self.skipTest("no interpolator file in VARDA_CONVERTED")

    @unittest.skipUnless(any(WEIGHTS), "VARDA_WEIGHTS_PT not set")
    def test_bitwise_vs_weights_pt(self):
        for i, p in self.each():
            wp = WEIGHTS[i] if i < len(WEIGHTS) else ""
            if not wp:  # no independent weight export for this file
                continue
            meta, t = sio.read_safetensors(p)
            grid = grid_names(sio.read_header(p)[0])
            w = torch.load(wp, mmap=True, weights_only=True)
            self.assertEqual(set(t) - grid, set(w))  # all old tensors, plus exactly the grid.* ones
            self.assertEqual(set(t) & grid, grid)
            for k, r in w.items():
                a = torch.from_numpy(np.array(t[k]))
                self.assertEqual(a.dtype, r.dtype, k)
                self.assertEqual(tuple(a.shape), tuple(r.shape), k)
                self.assertTrue(a.reshape(-1).view(torch.uint8).equal(r.contiguous().reshape(-1).view(torch.uint8)), k)
            with open(os.path.join(os.path.dirname(wp), "config.json")) as f:
                ref = json.load(f)
            cfg = json.loads(meta["config"])
            cfg.pop("forcings")  # added by the converter; checked in test_forcings_section
            for s in (cfg["source"], ref["source"]):
                s.pop("checkpoint"), s.pop("mtime", None)
            self.assertEqual(cfg, ref)

    @unittest.skipUnless(os.environ.get("VARDA_CKPTS"), "VARDA_CKPTS not set")
    def test_grid_coordinates_are_the_zip_members_bitwise(self):
        """grid.* equals the checkpoint's data/latitudes|longitudes members (read independently, via numpy)."""
        import zipfile
        for i, p in self.each():
            ck_path = os.environ["VARDA_CKPTS"].split(":")[i]
            _, t = sio.read_safetensors(p)
            with zipfile.ZipFile(ck_path) as z:
                for key in ("latitudes", "longitudes"):
                    m = [n for n in z.namelist() if n.endswith(f"anemoi-metadata/data/{key}.numpy")]
                    self.assertEqual(len(m), 1)
                    self.assertEqual(np.frombuffer(z.read(m[0]), "<f8").tobytes(), np.asarray(t[f"grid.{key}"]).tobytes())


class InterpolatorTimes(unittest.TestCase):
    """convert.interpolator_times: the model's times must agree with every metadata copy (no files needed)."""

    @staticmethod
    def case(t_in=(0, 6), t_out=(1, 2, 3, 4, 5, 6), explicit=None, rel=None, freq="1h", timestep="1h"):
        import types
        model = types.SimpleNamespace(input_times=list(t_in), output_times=list(t_out))
        explicit = explicit or {"input": list(t_in), "target": list(t_out)}
        rel = rel or (list(t_in), list(t_out))
        meta = {"config": {"training": {"explicit_times": explicit}, "data": {"frequency": freq}},
                "metadata_inference": {"data": {"timesteps": {"input_relative_date_indices": rel[0],
                                                              "output_relative_date_indices": rel[1],
                                                              "timestep": timestep}}}}
        return model, meta

    def test_ok(self):
        import convert
        model, meta = self.case()
        self.assertEqual(convert.interpolator_times(model, meta, 2, 6),
                         {"input": [0, 6], "target": [1, 2, 3, 4, 5, 6], "frequency_hours": 1})

    def test_mismatches_rejected(self):
        import convert
        bad = [dict(explicit={"input": [0, 6], "target": [1, 2, 3, 4, 5]}),
               dict(rel=([0, 6], [1, 2, 3, 4, 5, 7])),
               dict(freq="6h"), dict(freq="30m", timestep="30m"),
               dict(t_out=(1, 3, 2, 4, 5, 6))]
        for kw in bad:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                model, meta = self.case(**kw)
                with self.assertRaises(convert.ConvertError):
                    convert.interpolator_times(model, meta, 2, 6)
        model, meta = self.case()
        with self.assertRaises(convert.ConvertError):
            convert.interpolator_times(model, meta, 2, 5)  # n_step_output disagrees with the times

    def test_kinds(self):
        import convert
        self.assertEqual(convert.MODEL_KINDS["anemoi.models.models.interpolator.AnemoiModelEncProcDecMultiOutInterpolator"],
                         INTERPOLATOR)
        self.assertEqual(convert.MULTI_OUTPUT_KINDS, {INTERPOLATOR})


class RestrictedUnpickler(unittest.TestCase):
    def test_foreign_global_is_inert(self):
        marker = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"varda-pwned-{os.getpid()}")

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        data = pickle.dumps({"x": Evil(), "n": [1, 2.5, "s"]})
        obj = ck.Unpickler(io.BytesIO(data)).load()
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(obj["x"]._qualname, "posix.system")
        self.assertIsInstance(obj["x"], ck.Stub)
        self.assertEqual(obj["n"], [1, 2.5, "s"])

    def test_getattr_only_on_stubs(self):
        with self.assertRaises(pickle.UnpicklingError):
            ck._safe_getattr(os, "system")

    def test_unknown_rebuild_rejected(self):
        class R:
            def __reduce__(self):
                return (torch._utils._rebuild_qtensor, ())

        with self.assertRaises(pickle.UnpicklingError):
            ck.Unpickler(io.BytesIO(pickle.dumps(R()))).load()


class Container(unittest.TestCase):
    def test_roundtrip_and_order(self):
        import tempfile
        t = {"b": torch.arange(5, dtype=torch.int64), "a": torch.randn(3, 2), "c": torch.arange(4, dtype=torch.int64),
             "z": torch.zeros(0)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.safetensors")
            n = sio.write_safetensors(p, t, {"k": "v"})
            self.assertEqual(os.path.getsize(p), n)
            h, start = sio.read_header(p)
            self.assertEqual(sorted(t, key=lambda k: h[k]["data_offsets"][0] if h[k]["data_offsets"][1] else 1 << 60)[:3],
                             ["b", "c", "a"])
            m, r = sio.read_safetensors(p)
            self.assertEqual(m, {"k": "v"})
            for k in t:
                self.assertTrue(np.array_equal(r[k], t[k].numpy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
