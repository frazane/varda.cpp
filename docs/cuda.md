# CUDA backend

`varda::make_cuda_backend` (`src/varda/cuda/`): the fast backend of varda.cpp on NVIDIA GPUs
(A100, sm_80), fp32 and fp16 (autocast-faithful), with CUDA-graph replay and a device-level
entry point for the autoregressive rollout. It runs both models of Varda-single-1.0: the
forecaster (one output step) and the temporal downscaler `varda-td` (anemoi's multi-output
interpolator: T_out = 6 output steps per call, §11). Same maths as the CPU oracle, checked against the
golden activations with the tolerances of `tolerances/<model>.json` (`docs/format.md` §2-4).

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
`Config` (no model-specific literal anywhere; Varda and the dev model run the same code); a
forward pass only enqueues kernels and cuBLASLt calls on a stream. The sequence:

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
                   mlp, add+LN (dec.out_norm), out_proj -> dec_out [G, T_out V_out (padded to 8)]
output             per output step t: y[t] = float(dec_out[:, t V_out + v]) + xn[step] on the
                   prognostic outputs, boundings in config order (normalised space)
denormalise        (y - add_out) / mul_out over (T_out, G, V_out)
```

Graphs come from `build_csr_by_dst` (`src/varda/graph.h`): destination-major, stable within a
destination. Each graph is uploaded as `row_ptr` and one 16-byte record per edge
`{a0, a1, a2, bitcast(src)}` (attributes zero-padded to 3; `edge_attr_dim` 1..3 supported), so an
edge costs one broadcast load.

Choices: the edge projection folded into the attention (no `(E, C)` edge tensor, no K = 3
GEMMs), destination-sorted layouts, fused `[q|k|v|self]` / `[q|self]` / `[k|v]` GEMMs, explicit
casts reproducing autocast, destination chunking of the decoder, one CUDA graph over static
buffers; no padded degree buckets (a warp walks a destination's CSR row, any degree, 0
included), the residual adds are fused with the LayerNorm that follows them, the attention writes the projection's input `T(T(a) + x_r)`
directly, and nothing is compiled at run time (first call = one eager forward).

## 2. Precision plan

fp32 (`Precision::FP32`): every tensor fp32; GEMMs are cuBLASLt `CUBLAS_COMPUTE_32F` (true
fp32 FMA, no TF32); the reference is `cpu-fp32-pyg` and the table `cuda-fp32`.

fp16 (`Precision::FP16`) reproduces `torch.autocast("cuda", float16)` around the reference's
`predict_step` with the triton attention:

| op | reference under autocast | here |
|---|---|---|
| normaliser, input assembly | fp32 | fp32; `x_data` stored fp16 (the Linear's input cast) |
| every Linear | fp16 in, fp16 weights and bias, fp32 accumulation, fp16 out | cuBLASLt fp16 GEMM, `COMPUTE_32F`, bias epilogue in fp32, one rounding |
| LayerNorm | fp32 (input upcast), fp32 out, cast to fp16 by the next Linear | fp32 statistics and affine, output rounded to fp16 (same value as the cast) |
| edge Linear | fp16 attr and weights, fp16 output `e` | attr, `W_e`, `b_e` rounded to fp16 at load; folded (the fp16 rounding of `e` itself is not reproduced) |
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
| `normalise`, `denormalise` | `x*mul + add`, `(y - add)/mul` | `__fmul_rn`/`__fadd_rn`, `__fsub_rn`/`__fdiv_rn`: bitwise the reference's two ops |
| `assemble_x_data` | `[xn t0 | xn t1 | attr | 0]` | time-major features, zero padding to the GEMM's K |
| `layernorm` | warp per row, two-pass fp32 statistics in registers | in place allowed; any C that is a multiple of 256 up to 2048 (256, 512, ..., 2048; checked at load), as for every kernel |
| `add_layernorm` | `out1 = T(y + r1)`, optional `out2 = T(out1 + r2)`, then LN | residual + next LN in one pass; the latent skip uses `r2` |
| `gelu_inplace` | exact erf (or tanh) in fp32, 8 elements per thread | bandwidth-bound |
| `attention` | §3.2 | fp32 online softmax, folded edge projection, fused `+ x_r` |
| `output_assemble` | fp32 cast, prognostic residual, boundings (relu, hardtanh, fraction), per output step (grid y = step) | optional `out.pre_bound` copy (hooked runs only) |
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

Two things mattered (dev model fp16, per forward):
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
| `yn`, `y` (fp32) | T_out G V_out each |
| `x_data` (T) | G ldF |
| `dec_out` (T) | G ldVo (ldVo = T_out V_out rounded up to a multiple of 8) |
| hidden-node buffers (T) | 8 x [H, C] + [H, 4C] + [H, mlp]: `z_enc`, two ping-pong `z`, LN out, attention out, GEMM out, `out1`; `[q|k|v|self]` (also the decoder's `[k|v]`); MLP hidden |
| shared region R1 (T) | max(encoder source side 3 G C, decoder chunk (7C + mlp) per row x rows) |

The encoder's source side (`emb_src` with its LayerNorm in place, then `[k|v]` of all data
nodes) and the decoder's destination chunks never live at the same time, so they share R1. The
decoder chunk size is what fits in R1 (at least 2^18 rows, a multiple of 128 rows so that every
row offset stays 256-byte aligned; `VARDA_DEC_CHUNK_ROWS` overrides): Varda gets 4 chunks of
422 272 rows, the dev model one chunk. A destination's edges are never split (the attention takes
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

Varda, A100-SXM4-80GB, nsys kernel sums per forward (eager launches, the job of the results
in §8):

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
75% of DRAM bandwidth.

fp32 is SGEMM-bound (95% of the time in `ampere_sgemm_128x64_tn` at 97.6% of peak): the ceiling
without TF32, which the 1e-5 tables exclude.

## 6. Device-level entry point (rollout)

`src/varda/cuda/cuda_backend.h`:

```cpp
namespace varda::cuda {
class CudaBackend : public Backend {
public:
    virtual int64_t input_numel() const = 0;   // (1, T_in, [1,] G, V_in)
    virtual int64_t output_numel() const = 0;  // (1, T_out, 1, G, V_out); T_out = 1 for the forecaster
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
assembly with residual and all three bounding types, for one and for six output steps (bitwise). `test_cuda_gemm` checks the
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

### Results (NVIDIA A100-SXM4-80GB, same job for all rows)

Warm, hook-free, median of 10 (Varda) / 20 (dev model) forwards after 3 warm-up, CUDA
events, device-resident input/output (`predict_step_device`: normalise, forward, residual,
bounding, denormalise). Baselines: the original model in a separate job (A100-SXM4-80GB, warm, hook-free); the same-job comparison
is `docs/results.md` section 3.

| model | precision | CUDA graph | varda.cpp ms | original (triton) ms | speedup vs original |
|---|---|---|---|---|---|
| Varda | fp16 | on | **718.2** | 1321 | 1.84x |
| Varda | fp16 | off | 718.1 | | |
| Varda | fp32 | on | **7439.1** | 8089 | 1.09x |
| Varda | fp32 | off | 7440.3 | | |
| dev model | fp16 | on | **31.56** | 79.9 | 2.53x |
| dev model | fp16 | off | 31.82 | | |
| dev model | fp32 | on | **329.1** | 378.5 | 1.15x |
| dev model | fp32 | off | 329.3 | | |

| model | precision | load s | first call s (eager, host API) | host `predict_step` ms (incl. H2D/D2H, graph on) | peak device memory GB (incl. 0.45 context, 3.49 / 0.09 bench buffers) |
|---|---|---|---|---|---|
| Varda | fp16 | 1.2 (12.7 cold file cache; 1.9-2.7 cold since the parallel prefetch, `Model::load`, `docs/results.md` section 5) | 0.89 | 891 | 24.74 |
| Varda | fp32 | 1.1 | 7.59 | 7624 | 42.24 |
| dev model | fp16 | 0.7 | 0.08 | 36.5 | 2.43 |
| dev model | fp32 | 0.9 | 0.34 | 334.4 | 4.16 |

The graph makes no difference at these sizes (~400 kernels of milliseconds each). Every run
checked bitwise: graph replay == eager, `forward_normalised` split == fused, host == device,
first call == warm; with `--determinism` (graph-on rows) all 30 activations of two hooked runs of
one instance and of a second instance were bitwise identical. The host-API number includes
pageable copies of 1.07 GB in and 0.45 GB out (~170 ms on Varda).

### Parity (separate job, same GPU type; `varda-parity --report` JSON per model and precision)

**dev model** — rel_absmax (ratio to tolerance); fp32 vs cuda-fp32-triton for information (no table)

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

**Varda** — rel_absmax (ratio to tolerance); fp32 vs cuda-fp32-triton for information (no table)

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

## 10. Lessons (generic, for docs/design.md "Methodology")

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

## 11. Interpolator (temporal downscaler, `varda-td`)

anemoi's `AnemoiModelEncProcDecMultiOutInterpolator` (config `architecture.model.kind` =
`enc_proc_dec_interpolator`) is the forecaster's forward pass with T_out = `n_output_steps`
outputs per call (Varda downscaler: inputs at +0 h and +6 h, outputs +1 h .. +6 h). Everything up to
the decoder's output Linear is unchanged; the engine takes every size from the config (§1), so the
differences are:

| | forecaster | interpolator |
|---|---|---|
| `dec.out_proj` | [V_out, C], N padded to a multiple of 8 | [T_out V_out, C] (row t V_out + v: anemoi's "(time vars)"), N padded the same way (Varda: 402 -> 408) |
| `dec_out` (T) | G x ldVo | G x ldVo, ldVo = T_out V_out rounded up to 8 |
| `yn`, `y` (fp32) | G V_out | T_out G V_out, step t at `t G V_out` (contract `(1, T_out, 1, G, V_out)`) |
| output stage | one `output_assemble` launch | the same launch with a second grid dimension over the steps: step t reads columns `[t V_out, (t + 1) V_out)` of `dec_out`, adds the residual from the same input step for every t (`residual_step`: 0 for the Varda downscaler), applies the boundings in config order on its own values (fraction totals from the same step) |
| denormalise | G V_out values | T_out G V_out values (variable = index mod V_out, unchanged kernel) |
| hook | `dec.out` (G, V_out), `out.*` (1, 1, 1, G, V_out) | `dec.out` (G, T_out V_out), `out.*` (1, T_out, 1, G, V_out) |

fp16: the reference's `_assemble_output` is the forecaster's (`x_out.to(fp32)`, then residual, bounding,
denormalisation in fp32), so the cast placement of §2 holds for every step: `dec_out` is the fp16 Linear output,
everything after it fp32. A forecaster config with `n_output_steps` != 1 is rejected at load. The device rollout
(`rollout_cuda`) is forecaster-only: it checks `output_numel() == G V_out` and so refuses an interpolator.

Memory (Varda downscaler): `dec_out` grows by 1.1 GB (fp16) and `yn` + `y` by 4.5 GB: activations 25.6 GB in fp16
(forecaster 20.1 GB) and 43.8 GB in fp32 (37.1 GB). The hook's `out.pre_bound` scratch is T_out G V_out fp32 (2.7 GB).

### 11.1 Verification

- Forecaster unchanged: `varda-run --out` of the dev model and Varda, fp32 and fp16, graph on and off, with this
  build and with a build of the previous commit in the same job: outputs bitwise identical (8/8). `varda-parity` on
  the dev model in the same job (A100-SXM4-80GB) gives the same per-key numbers for both binaries, worst ratios
  0.314 (fp32), 0.656 / 0.733 (fp16), as in `docs/results.md` section 2; synthetic a-f: the numbers of
  `docs/testing.md` section 4, identical per key for both binaries. On a PG506-232 node (A100 96 GB) the same
  binaries give other bits in fp16 (not investigated; most likely other cuBLASLt kernel choices): dev model fp16
  worst 0.884 (dec.out) vs `cpu-fp32-pyg` and 0.697 (proc.10.out) vs `cuda-fp16-triton`, still within the tables;
  the published numbers are A100-SXM4-80GB numbers.
- `test_cuda_kernels`: output assembly with T_out = 1 and 6 (V_out = 9, padded rows), residual and all three
  bounding types, bitwise against a host reference; `test_cuda_gemm`, `test_rollout_cuda` pass.
- Synthetic interpolators (`docs/testing.md` section 4): g (T_out 6, residual from input step 0, no latent skip,
  relu) and h (T_out 3, residual from the last input step, latent skip, all three boundings), worst `rel_absmax`
  over all keys (PG506-232 node):

| model | CUDA fp32 | CUDA fp32 dec chunk 128 | CUDA fp16 |
|---|---|---|---|
| g | 3.08e-07 (0.031 of tol) | 3.08e-07 | 1.16e-03 (0.388) |
| h | 3.19e-07 (0.032) | 3.19e-07 | 1.31e-03 (0.438) |

  Hook-free == hooked bitwise, and the same numbers with `--no-cuda-graph`.

- Varda downscaler against its golden data (`varda-parity`, A100-SXM4-80GB, `tolerances/varda-td.json`): every
  key passes, 30 keys present and NaN-free; hook-free == hooked bitwise; graph on and graph off give the same
  per-key numbers. `rel_absmax / max_abs (ratio to tolerance)`; fp32 vs cuda-fp32-triton for information (no table):

| key | fp32 vs cpu-fp32-pyg (`cuda-fp32`) | fp32 vs cuda-fp32-triton | fp16 vs cpu-fp32-pyg (`cuda-fp16`) | fp16 vs cuda-fp16-triton (`cuda-fp16-vs-ref-fp16`) |
|---|---|---|---|---|
| in.x | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| in.x_norm | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_src | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.x_dst | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| enc.emb_src | 0 (bitwise) | 0 (bitwise) | 5.85e-04 / 0.00515 (0.53) | 4.44e-04 / 0.00391 (0.37) |
| enc.emb_dst | 0 (bitwise) | 0 (bitwise) | 6.89e-04 / 0.00183 (0.53) | 7.35e-04 / 0.00195 (0.53) |
| enc.out | 2.71e-06 / 0.000151 (0.27) | 1.17e-06 / 6.48e-05 | 1.40e-03 / 0.0777 (0.58) | 1.40e-03 / 0.0781 (0.52) |
| proc.0.out | 1.50e-06 / 0.000259 (0.15) | 2.05e-06 / 0.000355 | 1.85e-03 / 0.32 (0.56) | 9.01e-04 / 0.156 (0.24) |
| proc.1.out | 1.46e-06 / 0.000267 (0.15) | 1.89e-06 / 0.000347 | 1.64e-03 / 0.301 (0.47) | 8.51e-04 / 0.156 (0.21) |
| proc.2.out | 1.80e-06 / 0.000282 (0.18) | 1.75e-06 / 0.000275 | 1.65e-03 / 0.258 (0.46) | 7.98e-04 / 0.125 (0.19) |
| proc.3.out | 2.23e-06 / 0.000297 (0.22) | 1.89e-06 / 0.000252 | 2.13e-03 / 0.283 (0.51) | 9.39e-04 / 0.125 (0.20) |
| proc.4.out | 2.05e-06 / 0.000296 (0.21) | 2.52e-06 / 0.000364 | 1.93e-03 / 0.278 (0.51) | 1.30e-03 / 0.188 (0.30) |
| proc.5.out | 1.91e-06 / 0.000284 (0.19) | 2.43e-06 / 0.000362 | 2.05e-03 / 0.306 (0.60) | 1.31e-03 / 0.195 (0.34) |
| proc.6.out | 1.92e-06 / 0.000274 (0.19) | 2.47e-06 / 0.000352 | 1.91e-03 / 0.273 (0.58) | 1.31e-03 / 0.188 (0.35) |
| proc.7.out | 2.01e-06 / 0.000259 (0.20) | 2.66e-06 / 0.000342 | 1.98e-03 / 0.255 (0.64) | 1.94e-03 / 0.25 (0.54) |
| proc.8.out | 1.95e-06 / 0.000253 (0.20) | 2.57e-06 / 0.000333 | 1.90e-03 / 0.246 (0.56) | 2.41e-03 / 0.312 (0.62) |
| proc.9.out | 1.94e-06 / 0.000262 (0.19) | 2.45e-06 / 0.000332 | 1.92e-03 / 0.26 (0.53) | 2.77e-03 / 0.375 (0.68) |
| proc.10.out | 2.67e-06 / 0.000366 (0.27) | 2.45e-06 / 0.000336 | 2.09e-03 / 0.287 (0.50) | 3.64e-03 / 0.5 (0.76) |
| proc.11.out | 4.09e-06 / 0.000557 (0.41) | 2.49e-06 / 0.000339 | 2.70e-03 / 0.367 (0.34) | 3.67e-03 / 0.5 (0.40) |
| proc.12.out | 1.18e-05 / 0.00177 (0.50) | 3.96e-06 / 0.000595 | 8.84e-03 / 1.33 (0.38) | 7.49e-03 / 1.12 (0.29) |
| proc.13.out | 3.17e-05 / 0.00861 (0.50) | 1.05e-05 / 0.00284 | 2.40e-02 / 6.51 (0.38) | 1.68e-02 / 4.62 (0.23) |
| proc.14.out | 3.00e-05 / 0.0239 (0.52) | 1.07e-05 / 0.00854 | 2.17e-02 / 17.3 (0.37) | 2.66e-02 / 21.5 (0.40) |
| proc.15.out | 1.57e-05 / 0.0277 (0.51) | 8.49e-06 / 0.015 | 2.31e-02 / 40.8 (0.61) | 3.04e-02 / 54 (0.71) |
| latent | 1.57e-05 / 0.0277 (0.51) | 8.49e-06 / 0.015 | 2.31e-02 / 40.8 (0.61) | 3.04e-02 / 54 (0.71) |
| dec.emb_dst | 0 (bitwise) | 0 (bitwise) | 6.16e-04 / 0.00567 (0.56) | 4.24e-04 / 0.00391 (0.33) |
| dec.block_out | 1.41e-05 / 0.00838 (0.34) | 7.67e-06 / 0.00454 | 1.04e-02 / 6.17 (0.47) | 1.19e-02 / 7.06 (0.48) |
| dec.out | 8.39e-05 / 0.0136 (0.52) | 2.98e-05 / 0.00482 | 3.04e-02 / 4.92 (0.48) | 2.78e-02 / 4.5 (0.39) |
| out.pre_bound | 8.39e-05 / 0.0136 (0.52) | 2.98e-05 / 0.00482 | 3.04e-02 / 4.92 (0.48) | 2.78e-02 / 4.5 (0.39) |
| out.norm | 8.39e-05 / 0.0136 (0.52) | 2.98e-05 / 0.00482 | 3.04e-02 / 4.92 (0.48) | 2.78e-02 / 4.5 (0.39) |
| out.y | 4.56e-07 / 0.0938 (0.05) | 3.04e-07 / 0.0625 | 3.28e-04 / 67.4 (0.60) | 3.81e-04 / 78.5 (0.61) |
| **worst ratio** | 0.519 (proc.14.out) | | 0.637 (proc.7.out) | 0.758 (proc.10.out) |

  The error grows in processor blocks 11-14 and in the output Linear in both precisions; the reference shows the
  same growth between its own runs (cuda-fp32-triton vs cpu-fp32-pyg: 2.1e-5 at proc.13.out, 5.4e-5 at dec.out;
  fp16 vs fp32: 3.6e-2 at proc.13.out), which the tolerance file is calibrated on: a property of the model, not of the
  engine.

### 11.2 Speed

Warm, hook-free, device-resident input and output (`varda-bench` with its output buffers sized T_out G V_out: median of 10 forwards after 5 warm-up, fp32 5
after 2; CUDA graph replay unless stated), forecaster and downscaler in the same job; input of the downscaler: random
values whose normalised form is N(0, 1) (timing only). A100-SXM4-80GB (the GPU of §8):

| model | precision | graph | median ms | first call s (eager) | host `predict_step` ms (graph on) | activations GB |
|---|---|---|---|---|---|---|
| Varda downscaler | fp16 | on | **732.2** | 1.25 | 1062 | 25.6 |
| Varda downscaler | fp16 | off | 733.1 | 1.04 | 1005 (graph off) | 25.6 |
| Varda forecaster | fp16 | on | 720.1 | 0.86 | 824 | 20.1 |
| Varda downscaler | fp32 | on | **7524.4** | 7.84 | 7816 | 43.8 |
| Varda forecaster | fp32 | on | 7451.0 | 7.58 | 7579 | 37.1 |

On an `NVIDIA PG506-232` node (A100 SXM4, 96 GB; another node type, which runs these kernels ~10%
faster), same job for all rows: downscaler fp16 664.6 ms (graph off 666.0), forecaster fp16 653.8 ms; fp32 6530.8 vs
6473.9 ms.

The six output steps cost 11-12 ms over the forecaster's forward in fp16 (nsys on the PG506-232, eager, per forward:
output Linear +4.0 ms for N 408 instead of 72, output assembly 5.0 ms instead of 0.75, denormalisation 3.5 instead of
0.65; everything else unchanged), 73 ms (1%) in fp32. The host API copies 2.7 GB of output back (pageable), hence the
larger host number. Every run checked bitwise graph == eager, split == fused, host == device, first call == warm;
`--determinism` (fp16, PG506-232): the 30 activations of two hooked runs and of a second instance identical.
