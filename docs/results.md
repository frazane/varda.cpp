# Results

Every number names its configuration. Parity metric: `rel_absmax` (docs/format.md §2);
"ratio" is rel_absmax / tolerance from `tolerances/<model>.json`. Varda = varda
(bit-identical checkpoint). Full per-key tables: docs/cpu.md, docs/cuda.md, and the JSON
reports in the job directories named below.

## 1. Converter

`python/convert.py` on the HF checkpoint and on dev: every `weights.pt` tensor of
the Python engine is bitwise equal (names, dtypes, shapes, bytes); config equal except
`source.checkpoint`/`mtime` and the added `forcings` section; plus `grid.latitudes/longitudes`
(F64, bitwise the checkpoint's supporting arrays). Independently re-checked by the integrator
(data-section sha256 and tensor tables). Details: docs/tools.md, docs/forcings.md.

## 2. One-step parity (integrator re-runs from a clean build)

| model | device / precision | vs | table | worst ratio (key) | hook inert | job |
|---|---|---|---|---|---|---|
| dev | CPU fp32, 64 threads | cpu-fp32-pyg | cuda-fp32 | 0.101 (proc.1.out) | bitwise | 283438 |
| Varda | CPU fp32, 64 threads | cpu-fp32-pyg | cuda-fp32 | 0.181 (dec.block_out) | bitwise | 283438 |
| dev | CUDA fp32 | cpu-fp32-pyg | cuda-fp32 | 0.314 (dec.block_out) | bitwise | 283534 |
| dev | CUDA fp16 | cpu-fp32-pyg | cuda-fp16 | 0.656 (dec.out) | bitwise | 283534 |
| dev | CUDA fp16 | cuda-fp16-triton | cuda-fp16-vs-ref-fp16 | 0.733 (proc.8.out) | bitwise | 283534 |
| Varda | CUDA fp32 | cpu-fp32-pyg | cuda-fp32 | 0.329 (dec.block_out) | bitwise | 283534 |
| Varda | CUDA fp16 | cpu-fp32-pyg | cuda-fp16 | 0.670 (proc.2.out) | bitwise | 283534 |
| Varda | CUDA fp16 | cuda-fp16-triton | cuda-fp16-vs-ref-fp16 | 0.620 (proc.7.out) | bitwise | 283534 |

All 30 contract keys present and NaN-free in every run; input-assembly keys (`in.*`,
`enc.x_*`) are bitwise everywhere, and in fp32 so are the embeddings. CPU outputs are bitwise
independent of the thread count; CUDA graph replay, eager execution and hooked runs are bitwise
equal. Jobs: `$VARDA_SCRATCH/jobs/int-cpu-verify/`, `$VARDA_SCRATCH/jobs/int-gpu-verify/`.

## 3. Speed: one forward pass, same job

NVIDIA A100-SXM4-80GB, driver 595.91.07, node <node>; hook-free, warm (median of 20 after 5
warm-up calls, CUDA events), input on the device, TF32 off; `bench/run_bench.sh`, jobs 283535
(Varda) and 283536 (dev). varda.cpp replays its CUDA graph; the Python fast path
uses torch.compile + CUDA graph; the original is anemoi with the triton attention backend
under fp16 autocast (or plain fp32).

| model | precision | original anemoi | Python fast path | **varda.cpp** | vs original | vs Python |
|---|---|---|---|---|---|---|
| Varda | fp16 | 1338.0 ms | 880.4 ms | **724.3 ms** | 1.85x | 1.22x |
| Varda | fp32 | 8067.5 ms | 7572.9 ms | **7438.5 ms** | 1.08x | 1.02x |
| dev | fp16 | 79.0 ms | 38.0 ms | **31.7 ms** | 2.49x | 1.20x |
| dev | fp32 | 379.1 ms | 333.6 ms | **329.8 ms** | 1.15x | 1.01x |

First call (load excluded): varda.cpp 0.9 s / 7.7 s (Varda fp16 / fp32), against 4.9 s / 12.7 s
for the original and 16-36 s for the Python fast path (torch.compile, warm cache). Peak device
memory (cudaMemGetInfo, incl. CUDA context and ~3.5 GB of the bench's own buffers on Varda):
varda.cpp 24.7 / 42.2 GB, original 24.2 / 38.0 GB, Python 19.7 / 34.8 GB.

Where the fp16 time goes (nsys, docs/cuda.md): GEMMs 541 ms at 249 TFLOP/s (80% of the
312 TFLOP/s peak, 135 TFLOP per forward), attention 70 ms, GELU 48 ms, add+LayerNorm 46 ms.
fp32 is SGEMM-bound at 97.6% of the FMA peak: only TF32 could move it, which the 1e-5 fp32
tolerances exclude.

CPU (cpu-short, AMD EPYC 7713, 64 threads on 64 cores, fp32): Varda forward 60 s warm, peak RSS
24.7 GB; dev 5.3 s. For comparison (from the Python project): anemoi CPU 196-213 s,
Python plain path 242 s, both at 64 threads.
