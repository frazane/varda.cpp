"""Checks of a converted model.safetensors (docs/format.md section 1).

    VARDA_CONVERTED=/path/model.safetensors [VARDA_WEIGHTS_PT=/path/weights.pt] \
        .venv/bin/python -I python/tests/test_convert.py

VARDA_CONVERTED may list several files separated by ':'. VARDA_WEIGHTS_PT (same count, same order, or
unset) enables the bitwise comparison with the Python engine's weights.pt, and config.json next to it
is compared too. Without VARDA_CONVERTED the tests are skipped. The restricted-unpickler tests need no files.
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
WEIGHTS = [p for p in os.environ.get("VARDA_WEIGHTS_PT", "").split(":") if p]


def expected_names(cfg):
    """Tensor names of the contract table (docs/format.md 1.1; $IE_ROOT/docs/contracts.md 1) from the config."""
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
            self.assertEqual(sorted(header["__metadata__"]), ["config", "varda_format"])
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
            self.assertEqual(a["model"]["kind"], "enc_proc_dec")
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

    def test_names_shapes(self):
        for _, p in self.each():
            header, _ = sio.read_header(p)
            c = json.loads(header["__metadata__"]["config"])
            d = c["dims"]
            names = {k for k in header if k != "__metadata__"}
            self.assertEqual(names, expected_names(c))
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
            self.assertEqual(header["dec.out_proj.weight"]["shape"], [d["n_output_vars"], C])
            self.assertEqual(header["proc.0.q.weight"]["shape"], [C, C])
            self.assertEqual(header["proc.0.mlp1.weight"]["shape"], [d["mlp_hidden"], C])
            for n, h in header.items():
                if n != "__metadata__":
                    self.assertEqual(h["dtype"], "I64" if n.endswith("edge_index") else "F32", n)

    @unittest.skipUnless(WEIGHTS, "VARDA_WEIGHTS_PT not set")
    def test_bitwise_vs_weights_pt(self):
        for i, p in self.each():
            wp = WEIGHTS[i]
            meta, t = sio.read_safetensors(p)
            w = torch.load(wp, mmap=True, weights_only=True)
            self.assertEqual(set(t), set(w))
            for k, r in w.items():
                a = torch.from_numpy(np.array(t[k]))
                self.assertEqual(a.dtype, r.dtype, k)
                self.assertEqual(tuple(a.shape), tuple(r.shape), k)
                self.assertTrue(a.reshape(-1).view(torch.uint8).equal(r.contiguous().reshape(-1).view(torch.uint8)), k)
            with open(os.path.join(os.path.dirname(wp), "config.json")) as f:
                ref = json.load(f)
            cfg = json.loads(meta["config"])
            for s in (cfg["source"], ref["source"]):
                s.pop("checkpoint"), s.pop("mtime", None)
            self.assertEqual(cfg, ref)


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
