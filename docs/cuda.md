# CUDA backend

`varda::make_cuda_backend` (`src/varda/cuda/`): the fast path of varda.cpp on NVIDIA GPUs
(A100, sm_80), fp32 and fp16 (autocast-faithful), with CUDA-graph replay and a device-level
entry point for the autoregressive rollout. Same maths as the CPU oracle and the Python
engine's fast path (`$IE_ROOT/engine/fast.py`), checked against the golden activations with
the tolerances of `tolerances/<model>.json` (`docs/format.md` §2-4).

Files:

| file | content |
|---|---|
| `src/varda/cuda/backend_cuda.cu` | `Engine<T>` (T = `float` or `__half`): loading, buffers, forward pass, activation hook, CUDA graph |
| `src/varda/cuda/cuda_backend.h` | public device-level interface `varda::cuda::CudaBackend` (§6) |
| `src/varda/cuda/kernels.h/.cu` | all custom kernels (§3) |
| `src/varda/cuda/gemm.h/.cu` | cuBLASLt wrapper for the Linear layers (§3.1) |
| `src/varda/cuda/common.h` | error checks, `DeviceBuffer` |
| `tests/test_cuda_kernels.cu`, `tests/test_cuda_gemm.cu` | ctest unit tests (§7) |
| `apps/varda-bench.cpp` | benchmark (§8) |

## 1. Design

One `Engine<T>` per loaded model. Everything is sized and allocated at load time from the
`Config` (no model-specific literal anywhere; Varda and dev run the same code); a
forward pass only enqueues kernels and cuBLASLt calls on a stream. The sequence (spec-model
§1-8, spec-interface §2-4):

```
normalise          xn = (x * mul_in) + add_in                       fp32, two rounded ops
assemble           x_data[g] = [xn t0 | xn t1 | node attr | 0 pad]  T, K padded to a multiple of 8
encoder, source    emb_src = Linear(x_data)          [G, C]         (hook: enc.emb_src)
                   LN_src in place; [k | v] = Linear(.)  [G, 2C]
encoder, dest.     emb_dst = Linear(h_attr)          [H, C]
                   [q | self] = Linear(LN_dst(emb_dst))  [H, 2C]
                   s = T(T(attention) + self)        fused attention kernel
                   out1 = T(proj(s) + emb_dst), LN_mlp           fused add+LN kernel
                   h = gelu(mlp1(.)); z_enc = T(mlp2(h) + out1), LN_dst of proc.0
processor (x16)    [q | k | v | self] = Linear(xn)   [H, 4C]  one GEMM
                   attention, proj, add+LN, mlp1, gelu, mlp2, add+LN (next block's LN_dst;
                   after the last block: latent = T(z + z_enc) and the decoder's LN_src)
decoder, source    [k | v] = Linear(LN_src(latent))  [H, 2C]
decoder, dest.     per chunk of data nodes: emb_dst, LN, [q | self], attention, proj, add+LN,
                   mlp, add+LN (dec.out_norm), out_proj -> dec_out [G, V_out (padded to 8)]
output             y = float(dec_out) + xn[step] on the prognostic outputs, boundings in config
                   order (normalised space)
denormalise        (y - add_out) / mul_out
```

Graphs come from `build_csr_by_dst` (`src/varda/graph.h`): destination-major, stable within a
destination. Each graph is uploaded as `row_ptr` and one 16-byte record per edge
`{a0, a1, a2, bitcast(src)}` (attributes zero-padded to 3; `edge_attr_dim` 1..3 supported), so an
edge costs one broadcast load.

Ideas carried over from `engine/fast.py`: the edge projection folded into the attention (no
`(E, C)` edge tensor, no K = 3 GEMMs), destination-sorted layouts, fused `[q|k|v|self]` /
`[q|self]` / `[k|v]` GEMMs, explicit casts reproducing autocast, destination chunking of the
decoder, one CUDA graph over static buffers. What is different: no padded degree buckets (a
warp walks a destination's CSR row, any degree, 0 included), the residual adds are fused with the
LayerNorm that follows them, the attention writes the projection's input `T(T(a) + x_r)`
directly, and nothing is compiled at run time (first call = one eager forward).

## 2. Precision plan

fp32 (`Precision::FP32`): every tensor fp32; GEMMs are cuBLASLt `CUBLAS_COMPUTE_32F` (true
fp32 FMA, no TF32); the reference is `cpu-fp32-pyg` and the table `cuda-fp32`.

fp16 (`Precision::FP16`) reproduces `torch.autocast("cuda", float16)` around the reference's
`predict_step` with the triton attention (spec-model §9, `fast.py`'s cast placement):

| op | reference under autocast | here |
|---|---|---|
| normaliser, input assembly | fp32 | fp32; `x_data` stored fp16 (the Linear's input cast) |
| every Linear | fp16 in, fp16 weights and bias, fp32 accumulation, fp16 out | cuBLASLt fp16 GEMM, `COMPUTE_32F`, bias epilogue in fp32, one rounding |
| LayerNorm | fp32 (input upcast), fp32 out, cast to fp16 by the next Linear | fp32 statistics and affine, output rounded to fp16 (same value as the cast) |
| edge Linear | fp16 attr and weights, fp16 output `e` | attr, `W_e`, `b_e` rounded to fp16 at load; folded (the fp16 rounding of `e` itself is not reproduced, as in `fast.py`) |
| attention | triton: q, k, v, e upcast, fp32 online softmax, output cast to fp16 | the same in one kernel; output rounded to fp16, then `+ x_r` in fp32 and rounded again |
| residual adds, GELU | fp16 tensors: fp32 opmath, rounded to fp16 | the same, each add rounded separately (fused kernels keep the roundings) |
| mapper output / residual stream | fp16 | fp16 |
| `dec.out` -> residual, bounding, denormalise | `.to(fp32)`, fp32 | fp32 |

GELU is exact erf (`dims.gelu_approximate` = "none"; "tanh" supported) in a separate kernel,
never cuBLASLt's GELU epilogue (tanh approximation). The activation hook upcasts fp16 tensors on
the device in 128 MB pieces; in fp16 mode `enc.x_src` is re-assembled in fp32 for the hook (the
reference's `x_data` is fp32), `enc.x_dst` is the file's own buffer.

## 3. Kernels

| kernel | what | notes |
|---|---|---|
| `normalise`, `denormalise` | spec-interface §2.3 | `__fmul_rn`/`__fadd_rn`, `__fsub_rn`/`__fdiv_rn`: bitwise the reference's two ops |
| `assemble_x_data` | `[xn t0 | xn t1 | attr | 0]` | time-major features, zero padding to the GEMM's K |
| `layernorm` | warp per row, two-pass fp32 statistics in registers | in place allowed; any C that is a multiple of 256 up to 2048 (256, 512, ..., 2048; checked at load), as for every kernel |
| `add_layernorm` | `out1 = T(y + r1)`, optional `out2 = T(out1 + r2)`, then LN | residual + next LN in one pass; the latent skip uses `r2` |
| `gelu_inplace` | exact erf (or tanh) in fp32, 8 elements per thread | bandwidth-bound |
| `attention` | §3.2 | fp32 online softmax, folded edge projection, fused `+ x_r` |
| `output_assemble` | fp32 cast, prognostic residual, boundings (relu, hardtanh, fraction) | optional `out.pre_bound` copy (hooked runs only) |
| `convert_2d` | dtype conversion with leading dimensions | weights at load, hook upcasts |

### 3.1 GEMMs (cuBLASLt)

`y[M, N] = x[M, K] W[N, K]^T + b` as the column-major `W^T x^T` (TN), bias epilogue. One plan
(descriptors + algorithm) per shape and alignment class, created on the first (eager) call of the
shape, so every later call, captured or not, runs the same kernel. Default: top heuristic
candidate (deterministic across processes). `VARDA_GEMM_AUTOTUNE=1` times up to 16 candidates on
the first call (timing-based, may differ between processes); on Varda it found nothing faster
(709 vs 708 ms). Padding: K of the embeddings to a multiple of 8 (zero columns, exact), the
output Linear's N from 67 to 72 (zero rows): the unpadded N = 67 selected an `align1` kernel at
34 TFLOP/s (6.9 ms per forward), padded it runs 0.64 ms per chunk.

### 3.2 Attention

A destination's C channels are split over `C / (256 vpl)` warps (default `vpl` = 1: a warp owns
256 channels, a lane one 8-channel vector = one 16-byte load in fp16, two in fp32). With
head_dim 64, 8 lanes share a head: the per-head dot product is a 3-step butterfly. Per edge: one
broadcast 16-byte edge record, one k and one v vector per lane (two edges in flight), fp32
online softmax per head, and the folded edge projection

```
s_ij = (<q_i, k_j> + <W_e^T q_i, a_ij> + <q_i, b_e>) / sqrt(d)
o_i  = sum_j p_ij v_j + W_e (sum_j p_ij a_ij) + b_e                (sum_j p_ij = 1)
```

with `W_e^T q_i` and `<q_i, b_e>` computed once per destination (`W_e` stored attribute-major
`[4, C]`, rows `W_e[:, 0..2]`, `b_e`, so the loads are coalesced). Zero in-degree gives 0 (as the
reference), so the output is `x_r`. All arithmetic uses explicit `__fmaf_rn`/`__fmul_rn`/
`__fadd_rn`, so the bits do not depend on the compiler's contraction of each template instance
(the unit test checks `vpl` 1, 2, 4 bitwise equal).

Two things mattered (dev fp16, per forward):
- one warp per destination holding all 1024 channels took 216 registers (8 warps/SM): 6.6 ms; four
  warps of 256 channels, 73 registers: 5.1 ms;
- `W_e` stored channel-major `[C, 3]`: the per-lane scalar loads had a 96-byte stride and made
  the kernel L1-bound (77% L1 throughput, 87% hit rate); attribute-major `[4, C]` with vector
  loads: 3.3 ms.

## 4. Memory plan

All buffers are allocated at load (`act()`), none during a forward pass:

| buffer | size |
|---|---|
| `x`, `xn` (fp32) | T G V_in each |
| `yn`, `y` (fp32) | G V_out each |
| `x_data` (T) | G ldF |
| `dec_out` (T) | G ldVo |
| hidden-node buffers (T) | 8 x [H, C] + [H, 4C] + [H, mlp]: `z_enc`, two ping-pong `z`, LN out, attention out, GEMM out, `out1`; `[q|k|v|self]` (also the decoder's `[k|v]`); MLP hidden |
| shared region R1 (T) | max(encoder source side 3 G C, decoder chunk (7C + mlp) per row x rows) |

The encoder's source side (`emb_src` with its LayerNorm in place, then `[k|v]` of all data
nodes) and the decoder's destination chunks never live at the same time, so they share R1. The
decoder chunk size is what fits in R1 (at least 2^18 rows, a multiple of 128 rows so that every
row offset stays 256-byte aligned; `VARDA_DEC_CHUNK_ROWS` overrides): Varda gets 4 chunks of
422 272 rows, dev one chunk. A destination's edges are never split (the attention takes
a destination range `[lo, lo + n)` of the global CSR), so chunking changes no result except
through GEMM row counts. Varda: fp16 20.1 GB of buffers + 0.7 GB weights and graphs; fp32 37.1 + 1.2 GB.

CUDA graph: `Model::predict_step` runs its first call eagerly (this creates every cuBLASLt plan),
captures `predict_step_device` on the internal buffers on the second call and replays it from
then on; host copies stay outside the graph. A graph over static buffers needs no extra memory.
Hooked calls always run eagerly. If anything fails during the capture (an exception from the
backend or cuBLASLt, an invalidated capture, instantiation), the capture is always ended and the
partial graph destroyed, a warning is printed, and the call and all later calls run eagerly
(`VARDA_DEBUG_CAPTURE_THROW=1` forces such a failure, for testing).

Loading is strict: every Linear weight and bias is taken with its exact expected shape
(`[N, K]` from the config: C, mlp_hidden, F_src, node_attr_dim, V_out), including the parts
stacked into `[q|k|v|self]`, `[q|self]`, `[k|v]`; a mismatch fails at load.

### 4.1 Tuning knobs (environment, read at load; defaults are the measured best)

| variable | default | effect |
|---|---|---|
| `VARDA_DEC_CHUNK_ROWS` | what fits in R1, >= 2^18 | decoder destination rows per chunk |
| `VARDA_ATT_VPL` | 1 | attention: 8-channel vectors per lane (1, 2, 4); same bits |
| `VARDA_KPAD` | 8 | K padding of the embedding GEMMs |
| `VARDA_GEMM_AUTOTUNE` | 0 | time cuBLASLt candidates per shape (non-deterministic choice) |
| `VARDA_GEMM_LOG` | 0 | print each GEMM plan |

## 5. Profile

Varda (varda), A100-SXM4-80GB, nsys kernel sums per forward (eager launches, job 283468,
`run-283468/prof-varda-{fp16,fp32}*`):

| kernels | fp16 ms | fp32 ms |
|---|---|---|
| GEMMs (cuBLASLt) | 541.5 (249 TFLOP/s, 80% of the 312 TFLOP/s tensor peak) | 7107.1 (19.0 TFLOP/s, 97.6% of the 19.5 TFLOP/s FMA peak) |
| attention (21 launches) | 69.6 | 110.6 |
| GELU | 48.3 | 96.5 |
| add + LayerNorm (42) | 45.9 | 105.4 |
| LayerNorm (6) | 8.5 | 18.7 |
| assemble, normalise, output, denormalise | 5.1 | 5.2 |
| **total** | **719.1** | **7444.0** |

GEMM work is 135 TFLOP per forward (processor 61%, decoder destination side 29%, encoder 9%);
at the fp16 tensor peak that alone is 434 ms, at cuBLAS's 249 TFLOP/s 542 ms. The other 178 ms
are memory- or latency-bound: per processor block attention 3.0 ms (ncu: 34% of DRAM bandwidth,
72% L2 hit rate, 36% occupancy, latency-bound; the hidden-node order already gives 6.8 uses per
gathered k/v row within a 10k-destination window), GELU 1.9 ms and add+LN 2 x 0.9 ms at ~1.9
TB/s (bandwidth-bound). The encoder attention (2.85M edges from 1.69M data nodes, no reuse) runs at
75% of DRAM bandwidth. Against the Python fast path (GEMMs 532 ms, attention 207, pointwise 88,
LayerNorm 42 ms): same GEMM time, attention 3x faster, pointwise/LN fused into fewer passes.

fp32 is SGEMM-bound (95% of the time in `ampere_sgemm_128x64_tn` at 97.6% of peak): matching the
Python path is the ceiling without TF32, which the 1e-5 tables exclude.

## 6. Device-level entry point (rollout)

`src/varda/cuda/cuda_backend.h`:

```cpp
namespace varda::cuda {
class CudaBackend : public Backend {
public:
    virtual int64_t input_numel() const = 0;   // (1, T_in, [1,] G, V_in)
    virtual int64_t output_numel() const = 0;  // (1, 1, 1, G, V_out)
    virtual void normalise(const float* x, float* xn, cudaStream_t s) = 0;
    virtual void forward_normalised(const float* xn, float* yn, cudaStream_t s) = 0;
    virtual void denormalise(const float* yn, float* y, cudaStream_t s) = 0;
    virtual void predict_step_device(const float* x, float* y, cudaStream_t s) = 0;
    virtual void prepare() = 0;               // one eager forward: creates every GEMM plan
    ...
};
CudaBackend* as_cuda(Model& m);               // nullptr if not CUDA
}
```

Device pointers, fp32, contiguous; each call only enqueues on `s` (no host sync, no allocation),
so after `prepare()` a caller may capture any sequence of these calls in its own CUDA graph
(`varda-bench` does). `predict_step_device(x, y) == denormalise(forward_normalised(normalise(x)))`
bitwise (checked by `varda-bench`), and `Model::predict_step` equals both. The calls share the
backend's scratch buffers: one forward in flight per model.

## 7. Tests

`ctest` (GPU node): `test_cuda_kernels` checks the attention against a host double reference that
materialises `e = W_e a + b_e` and adds it to k and v with a two-pass softmax (fp32 1e-5, fp16
2e-3 rel_absmax), on a graph with every degree class 0, 1, 2, 3, 5, 8, 17, 33, 64, odd node
counts, shuffled stored edge order, five channel/head layouts (C 256-1024 incl. 768, head_dim 8-128,
1-3 edge attributes), destination sub-ranges (bitwise equal to the full launch), zero in-degree
(output = `x_r` exactly) and `vpl` invariance (bitwise); LayerNorm and add+LN for every accepted C (sums bitwise, LN
vs double), GELU erf/tanh, normaliser and denormaliser (bitwise), input assembly (bitwise), output
assembly with residual and all three bounding types (bitwise). `test_cuda_gemm` checks the
Linear wrapper in both precisions with and without bias, N = 67 in rows of 72 (padding left
untouched), K = 8 / 168 / 4096, odd M, repeatability, with and without autotuning.

## 8. Benchmark

`varda-bench --model M --input GOLDEN [--precision fp16|fp32] [--no-cuda-graph] [--warmup N]
[--iters N] [--json OUT] [--determinism]`: load time, first call (host API, eager), `iters` warm
hook-free forwards timed with CUDA events on device-resident input and output
(`predict_step_device`; with the graph on: replays of a graph captured by the app), host-API
time (H2D + forward + D2H), peak device memory (cudaMemGetInfo). It checks bitwise graph ==
eager, split entry points == fused, host == device, first call == warm; `--determinism` hashes
every activation of two hooked runs of one instance and one of a second instance.

### Results (job 283468, <node>, NVIDIA A100-SXM4-80GB, same job for all rows)

Warm, hook-free, median of 10 (Varda) / 20 (dev) forwards after 3 warm-up, CUDA
events, device-resident input/output (`predict_step_device`: normalise, forward, residual,
bounding, denormalise). Baselines: coordinator's job 283323 (A100-SXM4-80GB, warm, hook-free).

| model | precision | CUDA graph | varda.cpp ms | Python fast path ms | original (triton) ms | speedup vs Python / original |
|---|---|---|---|---|---|---|
| Varda | fp16 | on | **718.2** | 868 | 1321 | 1.21x / 1.84x |
| Varda | fp16 | off | 718.1 | | | |
| Varda | fp32 | on | **7439.1** | 7592 | 8089 | 1.02x / 1.09x |
| Varda | fp32 | off | 7440.3 | | | |
| dev | fp16 | on | **31.56** | 38.0 | 79.9 | 1.20x / 2.53x |
| dev | fp16 | off | 31.82 | | | |
| dev | fp32 | on | **329.1** | 332.9 | 378.5 | 1.01x / 1.15x |
| dev | fp32 | off | 329.3 | | | |

| model | precision | load s | first call s (eager, host API) | host `predict_step` ms (incl. H2D/D2H, graph on) | peak device memory GB (incl. 0.45 context, 3.49 / 0.09 bench buffers) |
|---|---|---|---|---|---|
| Varda | fp16 | 1.2 (12.7 cold file cache; 1.9-2.7 cold since the parallel prefetch, `Model::load`, job 286500) | 0.89 | 891 | 24.74 |
| Varda | fp32 | 1.1 | 7.59 | 7624 | 42.24 |
| dev | fp16 | 0.7 | 0.08 | 36.5 | 2.43 |
| dev | fp32 | 0.9 | 0.34 | 334.4 | 4.16 |

The graph makes no difference at these sizes (~400 kernels of milliseconds each). Every run
checked bitwise: graph replay == eager, `forward_normalised` split == fused, host == device,
first call == warm; with `--determinism` (graph-on rows) all 30 activations of two hooked runs of
one instance and of a second instance were bitwise identical. The host-API number includes
pageable copies of 1.07 GB in and 0.45 GB out (~170 ms on Varda).

### Parity (job 283444, same GPU type; `run-283444/parity-*.{log,json}`)

**dev** — rel_absmax (ratio to tolerance); fp32 vs cuda-fp32-triton for information (no table)

| key | fp32 vs cpu-fp32-pyg (`cuda-fp32`) | fp32 vs cuda-fp32-triton | fp16 vs cpu-fp32-pyg (`cuda-fp16`) | fp16 vs cuda-fp16-triton (`cuda-fp16-vs-ref-fp16`) |
|---|---|---|---|---|
| in.x | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| in.x_norm | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_src | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_dst | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.emb_src | 0 (bitwise) | 0 (bitwise) | 5.16e-04 (0.57) | 7.17e-04 (0.65) |
| enc.emb_dst | 0 (bitwise) | 0 (bitwise) | 6.00e-04 (0.55) | 3.90e-04 (0.30) |
| enc.out | 2.03e-06 (0.20) | 7.92e-07 | 1.11e-03 (0.48) | 1.04e-03 (0.40) |
| proc.0.out | 2.82e-06 (0.28) | 1.09e-06 | 1.93e-03 (0.52) | 1.99e-03 (0.47) |
| proc.1.out | 2.90e-06 (0.29) | 1.18e-06 | 1.89e-03 (0.52) | 2.33e-03 (0.55) |
| proc.2.out | 2.78e-06 (0.28) | 1.18e-06 | 2.06e-03 (0.53) | 2.22e-03 (0.49) |
| proc.3.out | 2.66e-06 (0.27) | 1.27e-06 | 2.09e-03 (0.55) | 2.26e-03 (0.52) |
| proc.4.out | 1.90e-06 (0.19) | 1.09e-06 | 1.64e-03 (0.57) | 1.78e-03 (0.54) |
| proc.5.out | 1.59e-06 (0.16) | 8.84e-07 | 1.34e-03 (0.56) | 1.81e-03 (0.67) |
| proc.6.out | 1.53e-06 (0.15) | 7.66e-07 | 1.28e-03 (0.53) | 1.65e-03 (0.61) |
| proc.7.out | 1.36e-06 (0.14) | 6.45e-07 | 1.55e-03 (0.55) | 1.76e-03 (0.57) |
| proc.8.out | 1.23e-06 (0.12) | 5.84e-07 | 1.44e-03 (0.58) | 2.13e-03 (0.73) |
| proc.9.out | 1.14e-06 (0.11) | 5.99e-07 | 1.54e-03 (0.55) | 1.96e-03 (0.63) |
| proc.10.out | 1.09e-06 (0.11) | 5.44e-07 | 1.70e-03 (0.61) | 1.78e-03 (0.56) |
| proc.11.out | 1.10e-06 (0.11) | 5.76e-07 | 1.86e-03 (0.55) | 2.15e-03 (0.55) |
| proc.12.out | 1.28e-06 (0.13) | 6.40e-07 | 2.23e-03 (0.57) | 2.42e-03 (0.54) |
| proc.13.out | 1.42e-06 (0.14) | 6.68e-07 | 2.47e-03 (0.56) | 2.73e-03 (0.55) |
| proc.14.out | 1.75e-06 (0.17) | 8.10e-07 | 2.72e-03 (0.56) | 3.06e-03 (0.56) |
| proc.15.out | 2.00e-06 (0.20) | 1.04e-06 | 3.00e-03 (0.59) | 3.21e-03 (0.55) |
| latent | 2.25e-06 (0.23) | 1.20e-06 | 3.39e-03 (0.63) | 3.39e-03 (0.55) |
| dec.emb_dst | 0 (bitwise) | 0 (bitwise) | 5.00e-04 (0.57) | 4.50e-04 (0.45) |
| dec.block_out | 3.14e-06 (0.31) | 1.54e-06 | 1.51e-03 (0.56) | 1.61e-03 (0.52) |
| dec.out | 1.43e-06 (0.14) | 1.48e-06 | 1.25e-03 (0.66) | 8.69e-04 (0.39) |
| out.pre_bound | 1.43e-06 (0.14) | 1.48e-06 | 1.25e-03 (0.66) | 8.69e-04 (0.39) |
| out.norm | 1.43e-06 (0.14) | 1.48e-06 | 1.25e-03 (0.66) | 8.69e-04 (0.39) |
| out.y | 7.48e-08 (0.01) | 1.50e-07 | 6.66e-06 (0.55) | 5.98e-06 (0.43) |
| **worst ratio** | 0.314 (dec.block_out) | | 0.656 (dec.out) | 0.733 (proc.8.out) |

hook-free == hooked (bitwise): fp32 True, fp16 True; status fp32 pass, fp16 pass

**varda** — rel_absmax (ratio to tolerance); fp32 vs cuda-fp32-triton for information (no table)

| key | fp32 vs cpu-fp32-pyg (`cuda-fp32`) | fp32 vs cuda-fp32-triton | fp16 vs cpu-fp32-pyg (`cuda-fp16`) | fp16 vs cuda-fp16-triton (`cuda-fp16-vs-ref-fp16`) |
|---|---|---|---|---|
| in.x | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| in.x_norm | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_src | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_dst | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.emb_src | 0 (bitwise) | 0 (bitwise) | 4.87e-04 (0.57) | 6.01e-04 (0.61) |
| enc.emb_dst | 0 (bitwise) | 0 (bitwise) | 6.92e-04 (0.53) | 5.20e-04 (0.37) |
| enc.out | 1.51e-06 (0.15) | 6.23e-07 | 9.09e-04 (0.57) | 1.08e-03 (0.57) |
| proc.0.out | 1.26e-06 (0.13) | 5.49e-07 | 1.02e-03 (0.57) | 9.00e-04 (0.43) |
| proc.1.out | 1.40e-06 (0.14) | 6.15e-07 | 1.30e-03 (0.56) | 1.28e-03 (0.49) |
| proc.2.out | 1.43e-06 (0.14) | 8.10e-07 | 1.54e-03 (0.67) | 1.17e-03 (0.43) |
| proc.3.out | 1.22e-06 (0.12) | 6.07e-07 | 1.31e-03 (0.50) | 1.34e-03 (0.46) |
| proc.4.out | 1.17e-06 (0.12) | 5.26e-07 | 1.40e-03 (0.50) | 1.85e-03 (0.60) |
| proc.5.out | 1.11e-06 (0.11) | 4.85e-07 | 1.55e-03 (0.53) | 1.70e-03 (0.52) |
| proc.6.out | 1.06e-06 (0.11) | 6.16e-07 | 1.75e-03 (0.51) | 2.24e-03 (0.59) |
| proc.7.out | 1.06e-06 (0.11) | 6.63e-07 | 1.98e-03 (0.64) | 2.17e-03 (0.62) |
| proc.8.out | 9.91e-07 (0.10) | 6.39e-07 | 2.09e-03 (0.57) | 2.09e-03 (0.50) |
| proc.9.out | 9.72e-07 (0.10) | 6.07e-07 | 2.06e-03 (0.59) | 1.99e-03 (0.50) |
| proc.10.out | 9.43e-07 (0.09) | 7.07e-07 | 2.15e-03 (0.54) | 1.93e-03 (0.42) |
| proc.11.out | 1.01e-06 (0.10) | 6.56e-07 | 2.33e-03 (0.57) | 1.95e-03 (0.42) |
| proc.12.out | 1.04e-06 (0.10) | 6.64e-07 | 2.25e-03 (0.52) | 1.89e-03 (0.39) |
| proc.13.out | 1.06e-06 (0.11) | 6.38e-07 | 2.23e-03 (0.50) | 2.18e-03 (0.43) |
| proc.14.out | 1.13e-06 (0.11) | 7.89e-07 | 2.37e-03 (0.52) | 2.02e-03 (0.39) |
| proc.15.out | 1.45e-06 (0.14) | 1.02e-06 | 2.10e-03 (0.55) | 2.30e-03 (0.54) |
| latent | 1.45e-06 (0.15) | 1.05e-06 | 2.16e-03 (0.54) | 2.31e-03 (0.51) |
| dec.emb_dst | 0 (bitwise) | 0 (bitwise) | 4.36e-04 (0.57) | 3.39e-04 (0.38) |
| dec.block_out | 3.29e-06 (0.33) | 2.31e-06 | 3.02e-03 (0.66) | 3.01e-03 (0.58) |
| dec.out | 3.13e-06 (0.29) | 2.62e-06 | 3.54e-03 (0.57) | 2.91e-03 (0.41) |
| out.pre_bound | 3.13e-06 (0.29) | 2.62e-06 | 3.54e-03 (0.57) | 2.91e-03 (0.41) |
| out.norm | 3.13e-06 (0.29) | 2.62e-06 | 3.54e-03 (0.57) | 2.91e-03 (0.41) |
| out.y | 1.52e-07 (0.02) | 1.52e-07 | 2.96e-05 (0.64) | 2.31e-05 (0.44) |
| **worst ratio** | 0.329 (dec.block_out) | | 0.670 (proc.2.out) | 0.620 (proc.7.out) |

hook-free == hooked (bitwise): fp32 True, fp16 True; status fp32 pass, fp16 pass

## 9. Experiments that did not help (measured, reverted)

| change | result (Varda fp16, warm, graph) |
|---|---|
| cuBLASLt autotuning over up to 16 heuristic candidates | 708 vs 709 ms: nothing faster than the top heuristic; costs cross-process determinism. Kept as `VARDA_GEMM_AUTOTUNE=1`, off by default |
| embedding K padded to 32 / 64 instead of 8 | 708 / 708 ms (no change) |
| decoder chunks on 2 / 3 streams (memory-bound kernels of one chunk overlapping another chunk's GEMMs) | 716 / 716 vs 713 ms: cuBLAS GEMMs fill the GPU, nothing to overlap with. Removed |
| 4 edges in flight per warp in the attention instead of 2 | 732 vs 718 ms (103 instead of 75 registers, fewer resident warps). Removed |
| one warp per destination (all 1024 channels in a warp) | 216 registers, 8 warps/SM; replaced by 4 warps x 256 channels (§3.2) |
| own fp16 tensor-core GEMM (mma.sync m16n8k16, ldmatrix, cp.async, 128x256 tiles) with bias + exact-GELU epilogue, for mlp1 | correct (bitwise equal to cuBLASLt + GELU on small K), but 150 TFLOP/s without the GELU and 72 TFLOP/s with it, vs cuBLASLt 245-251 TFLOP/s + 1.9 ms GELU at [189 809 x 1024] x [1024 x 4096]: removed. A competitive fused kernel needs a CUTLASS-grade pipeline (not a dependency we may add) |

## 10. Lessons (generic, for docs/design.md "Methodology additions")

- **Profile per kernel before optimising, and look at the limiter, not the time.** ncu showed the
  first attention kernel L1-bound (strided scalar loads of a 3-column weight), then latency-bound
  (36% occupancy, 34% of DRAM bandwidth); neither is visible in a timeline.
- **Gather kernels: registers decide occupancy.** Splitting a destination's channels over several
  warps (each warp redoes the cheap per-edge scalar work) cut registers 3x and the kernel time
  1.3x; more loads in flight per warp (deeper unrolling) was slower because it cost occupancy.
- **Store small per-channel parameter sets structure-of-arrays** (`[4, C]`, not `[C, 3]`), so a
  warp's loads are contiguous: 1.5x on the attention.
- **Bitwise reproducibility needs fixed arithmetic, not just fixed order.** Two template instances
  of the same kernel contracted multiply-adds differently and gave different fp32 bits; explicit
  `__fmaf_rn` / `__fmul_rn` / `__fadd_rn` fixed it. A unit test that runs the same maths under two
  launch layouts and compares bits finds this immediately.
- **One cuBLASLt plan per shape, created on the first eager call**, makes graph replay, eager
  launches and hooked runs bitwise identical and capture-safe (no heuristics or allocation during
  capture). Autotuning by timing would break cross-process determinism for no gain here.
- **Pad GEMM dimensions to multiples of 8 with zero rows/columns** (exact): an N = 67 output layer
  silently selected an `align1` kernel 2.6x slower than the padded N = 72 (6.9 vs 2.6 ms per Varda forward).
- **Keep the reference's roundings when fusing**: `T(T(attn) + x_r)`, `T(y + r1)` then LN, the
  latent skip as a second rounded add: the fused kernels cost the same bandwidth and the fp16
  results stay within ~0.65-0.75 of the fp16 tolerance tables.
- **Benchmark inputs must be in host memory before timing**: reading `in.x` straight from an
  mmapped golden file on Lustre made the "first call" 9 s instead of 3 s.
- **fp32 on A100 is SGEMM-bound** (19.5 TFLOP/s FMA peak): only TF32 or emulation could move it, and
  TF32 breaks the 1e-5 tables. fp16 is GEMM-bound at ~245 TFLOP/s (cuBLAS); what remains is
  bandwidth-bound elementwise work whose biggest removable item (exact-erf GELU, 48 ms) needs a
  GEMM epilogue that cuBLASLt does not offer (its GELU epilogue is the tanh form).
