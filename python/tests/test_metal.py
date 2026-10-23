"""Metal integration checks using the existing synthetic fixtures (no torch).

Run after tests/run_synthetic.sh metal:
  .venv/bin/python -I python/tests/test_metal.py --bin build-metal --synthetic local/synthetic --work local/metal
"""
import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("varda_synthetic", Path(__file__).with_name("synthetic.py"))
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)
BIN = SYNTHETIC = WORK = None


class MetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BIN is None:
            raise unittest.SkipTest("run this GPU integration suite with --bin, --synthetic and --work")
        cls.model = SYNTHETIC / "a.model.safetensors"
        cls.inputs = SYNTHETIC / "a.golden.safetensors"
        cls.tmp = tempfile.TemporaryDirectory(prefix="metal-integration-", dir=WORK)
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_app(self, name, *args, ok=True, env=None):
        result = subprocess.run([str(BIN / name), *map(str, args)], check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=120)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_repeated_inference(self):
        result = self.run_app("varda-run", "--model", self.model, "--input", self.inputs,
                              "--device", "metal", "--precision", "fp32", "--steps", 3)
        self.assertIn("repeats bitwise equal", result.stdout)

    def test_fp16_rejected(self):
        result = self.run_app("varda-run", "--model", self.model, "--input", self.inputs,
                              "--device", "metal", "--precision", "fp16", ok=False)
        self.assertIn("Metal backend runs fp32 only", result.stdout)

    def test_invalid_chunk_rejected(self):
        for value in ("0", "not-a-number"):
            with self.subTest(value=value):
                env = dict(os.environ, VARDA_METAL_CHUNK_ROWS=value)
                result = self.run_app("varda-run", "--model", self.model, "--input", self.inputs,
                                      "--device", "metal", "--precision", "fp32", env=env, ok=False)
                self.assertIn("VARDA_METAL_CHUNK_ROWS must be", result.stdout)

    def test_host_rollout(self):
        # Add rollout metadata to a separate test model: original weights and
        # fixtures remain untouched. Use a real computed forcing to exercise
        # feedback, constant channels, dates and CPU/GPU handover together.
        tensors, metadata = synthetic.read_st(self.model)
        cfg = json.loads(metadata["config"])
        cfg["source"]["uuid"] += "-metal-rollout"
        prognostic = set(cfg["indices"]["residual_input"])
        kept = [i for i in range(cfg["dims"]["n_input_vars"]) if i not in prognostic]
        dynamic, *kept = kept
        data_index = cfg["indices"]["norm_input"][dynamic]
        cfg["variables"]["input"][dynamic] = "cos_julian_day"
        cfg["variables"]["data"][data_index] = "cos_julian_day"
        for out, data in enumerate(cfg["indices"]["norm_output"]):
            if data == data_index:
                cfg["variables"]["output"][out] = "cos_julian_day"
        cfg["forcings"] = {"dynamic": ["cos_julian_day"], "dynamic_input_index": [dynamic],
                           "constant": [cfg["variables"]["input"][i] for i in kept],
                           "constant_input_index": kept, "timestep_hours": 6}
        nodes = cfg["dims"]["n_data_nodes"]
        tensors["grid.latitudes"] = np.linspace(-45, 45, nodes, dtype=np.float64)
        tensors["grid.longitudes"] = np.linspace(-180, 180, nodes, dtype=np.float64)
        model = self.root / "rollout.model.safetensors"
        synthetic.write_st(str(model), tensors, {"varda_format": "1", "config": json.dumps(cfg)})
        results = {}
        for device in ("cpu", "metal"):
            output = self.root / f"rollout.{device}.safetensors"
            result = self.run_app("varda-rollout", "--model", model, "--input", self.inputs,
                                  "--device", device, "--precision", "fp32", "--threads", 2,
                                  "--date", "2024-06-01T00:00:00", "--steps", 4, "--out", output)
            self.assertIn("engine host", result.stdout)
            results[device] = synthetic.read_st(output)[0]["y"].astype(np.float64)
        self.assertTrue(np.isfinite(results["metal"]).all())
        for step, (actual, reference) in enumerate(zip(results["metal"], results["cpu"])):
            error = np.max(np.abs(actual - reference)) / np.max(np.abs(reference))
            self.assertLessEqual(error, synthetic.FP32_TOL, f"step {step}: rel_absmax={error}")

    def test_empty_graphs(self):
        tensors, metadata = synthetic.read_st(self.model)
        cfg = json.loads(metadata["config"])
        cfg["source"]["uuid"] += "-empty-graphs"
        for graph in ("enc", "proc", "dec"):
            tensors[f"graph.{graph}.edge_index"] = np.empty((2, 0), dtype=np.int64)
            tensors[f"graph.{graph}.edge_attr"] = np.empty((0, cfg["dims"]["edge_attr_dim"]), dtype=np.float32)
        model = self.root / "empty-graphs.model.safetensors"
        synthetic.write_st(str(model), tensors, {"varda_format": "1", "config": json.dumps(cfg)})
        results = {}
        for device in ("cpu", "metal"):
            output = self.root / f"empty-graphs.{device}.safetensors"
            self.run_app("varda-run", "--model", model, "--input", self.inputs,
                         "--device", device, "--precision", "fp32", "--threads", 2, "--out", output)
            results[device] = synthetic.read_st(output)[0]["out.y"].astype(np.float64)
        self.assertTrue(np.isfinite(results["metal"]).all())
        error = np.max(np.abs(results["metal"] - results["cpu"])) / np.max(np.abs(results["cpu"]))
        self.assertLessEqual(error, synthetic.FP32_TOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    options, remaining = parser.parse_known_args()
    BIN, SYNTHETIC, WORK = options.bin.resolve(), options.synthetic.resolve(), options.work.resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([str(BIN / "varda-metal-info")], check=False)
    if probe.returncode:
        raise SystemExit(probe.returncode)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
