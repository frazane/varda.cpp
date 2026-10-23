# CPU fp32 backend (the oracle)

`varda::make_cpu_backend` (`src/varda/cpu/`): a readable fp32 implementation of
`predict_step` on OpenMP + OpenBLAS. It mirrors the op sequence of anemoi's GraphTransformer
with the pyg backend on CPU, and is held to the `cuda-fp32` tolerance table.

| file | contents |
|---|---|
| `src/varda/cpu/kernels.h/.cpp` | Linear (tiled `cblas_sgemm` with fused epilogue), LayerNorm, GELU (erf / tanh), graph attention |
| `src/varda/cpu/backend_cpu.cpp` | weight binding (strict `take()`), CSR graphs, memory plan, forward pass, activation hook, profile |
| `tests/test_cpu_kernels.cpp` | kernels vs naive double-precision references; zero in-degree; chunk and thread-count bitwise invariance |
| `apps/varda-run.cpp` | load, read `in.x` from a safetensors file, run `predict_step` N times, timings, peak RSS, optional `out.y` |

Knobs (environment, for experiments only; neither changes results): `VARDA_CPU_CHUNK_ROWS`
(destination rows per chunk, default 65 536), `VARDA_CPU_PROFILE=1` (wall time per phase
and kernel class to stderr after each call).

## 1. Forward pass

```
xn     = x * mul[norm_input], rounded, + add[norm_input]          (T, G, V_in)  two fp32 ops (no FMA: -ffp-contract=off)
x_src  = [xn[t0] | xn[t1] | data attr]                             (G, T*V_in + A)
z      = encoder: src  kv = [k|v](LN_src(emb_src(x_src)))         all G data rows
                  dst  mapper block on emb_dst(hidden attr)        (H, C)
z      = proc_{n-1}(... proc_0(z)) (+ z if latent_skip)            (H, C)
dec    = decoder: src  kv = [k|v](LN_src(z));  dst: emb_dst(x_src), block, LN + Linear   (G, V_out)
y      = dec, + xn[residual_step] on the prognostic outputs; bounding in config order
Y      = (y - add[norm_output]) / mul[norm_output]                 division, as the reference
```

Each block (encoder, every processor block, decoder) is a *source side* (`LN_src`, then `k`
and `v` into one `[k | v]` row of 2C per source node, so one gather fetches both) and a
*destination side*, run over chunks of complete destination rows:

```
ln  = LN_dst(xd);  [q | x_r] = [q | self](ln)
att = attention(q, kv, lin_edge(edge_attr) on the fly)       per destination, CSR order
att = att + x_r;   h = proj(att) + xd                         residual on the un-normalised xd
out = mlp2(GELU(mlp1(LN_mlp(h)))) + h
```

In a processor block `LN_src` is `ln_dst` (one LayerNorm; it is evaluated on the source
side and again on the destination side, with identical results). Everything is read from
`Config`: sizes, index lists, LayerNorm eps, GELU variant, latent skip, residual step,
bounding list, and which LayerNorm/Linear biases are `absent`.

### Attention kernel

One OpenMP task per destination `i` (dynamic schedule; in-degree 3..64 on Varda). For its
incoming edges in CSR order (= stored order within a destination; `build_csr_by_dst` is a
stable sort):

1. `e_ij = attr_ij . W_e^T + b_e` for all C channels (the 3-wide `lin_edge`, recomputed per
   block per edge into a per-thread `deg x C` scratch; never stored for the graph);
2. `s_ij,h = sum_c q_ic (k_jc + e_ijc) / sqrt(head_dim)` (division by the fp32 value of
   `sqrt(C)`, as `alpha / out_channels**0.5`);
3. per head: max, `exp(s - max)`, sum in edge order, `+ 1e-16`, divide (torch_geometric
   softmax);
4. `out_i = sum_j (v_j + e_ij) * a_ij`, from zero, in edge order (scatter-add order).

A destination without incoming edges gets zeros. The head split is heads-major (`h*C + c`).
The channel sum in step 2 uses 8 independent partial sums so that it vectorises without
`-ffast-math`; torch's own reduction order differs anyway.

### Numerics choices

- Linear: the output tile is pre-filled with the bias and `sgemm` accumulates into it
  (`beta = 1`): the dot product is rounded before the bias is added, as in `addmm`.
- LayerNorm: mean and (biased) variance in double, then torch's fp32 formula
  `rstd = 1/sqrt(var + eps)`, `y = (x*rstd + (-rstd*mean)) * w + b`.
- GELU: `x * 0.5 * (1 + erf(x * M_SQRT1_2))` with glibc `erff` (tanh variant per torch).
- Residual adds, `a + x_r`, the latent skip and the prognostic residual are plain fp32 adds
  in the reference's order. Bounding: relu, hardtanh (clamp), fraction (clamp, then multiply
  by the already bounded total, the right-hand side read first).

## 2. Threading and determinism

The OpenBLAS used here (0.3.26, a site-provided build) is a **single-threaded build**: calling
`cblas_sgemm` from serial code runs on one core (91 GFLOP/s whatever
`openblas_set_num_threads` says). All parallelism is therefore OpenMP: `cpu::linear` cuts
the output into fixed 128-row x 512-column tiles, scheduled dynamically, each one
single-threaded `cblas_sgemm` call followed by its element-wise epilogue (bias, residual
add, GELU) while the tile is hot. Measured on 32 cores (`-c 64`), M = 65 536: 128 x 512 tiles
1.7-1.9 TFLOP/s against 1.2-1.5 for one row block per thread; 64-row tiles or full-width
rows were no better.

OpenBLAS's result for an element depends on the shape of the call that computes it
(splitting the same product into row blocks of 16, 160 or 640 changed 2-25% of the outputs
in the last bit). With a tile grid that depends only on the matrix shape, every element is
computed by the same call whatever the thread count. Together with the per-destination
attention and per-row LayerNorm, **results are bitwise independent of the thread count**:
`out.y` is identical at 7 and 64 threads (the dev model) and at 64 and 128 threads
(Varda), and repeated calls are bitwise equal. The destination chunk size does change GEMM
row counts but not tile shapes as long as chunks are multiples of 128 rows, so it changes
nothing either (unit test: a destination sub-range of the attention is bitwise equal).

## 3. Memory plan

Weights stay in the read-only mmap of the converted file (1.3 GB for Varda); only the CSR
graphs (destination-sorted copies of `edge_index`/`edge_attr`) and the transposed 3-wide
`lin_edge` weights are copied. Activations live in a small arena of named buffers that is
kept across calls (buffers with disjoint lifetimes share a name). Varda, C = 1024, fp32:

| buffer | rows x width | GB |
|---|---|---|
| `kv` (encoder: all G data rows; processor/decoder reuse it on H rows) | G x 2C | 13.8 |
| destination workspace (LN, [q, x_r], attention, MLP hidden) | 65 536 x (C + 2C + C + 4C) | 2.1 |
| `x_norm`, `x_src` | T·G x V_in, G x 162 | 1.1 + 1.1 |
| `z`, `latent`, two processor ping-pong buffers, encoder `emb_dst` | 5 x H x C | 3.9 |
| encoder `emb_src` chunk, decoder `emb_dst` / block-out chunks | 65 536 x C each | 0.8 |
| decoder output | G x V_out | 0.45 |

No per-edge tensor exists (the largest would be E_dec x C = 21 GB); `e_ij` lives in a
`deg x C` per-thread scratch (at most 64 x 4 KB). The destination side never holds more
than one chunk of rows, so the decoder's (G, C) intermediates (6.9 GB each) are never
materialised. With an activation hook the contract tensors that do not otherwise exist as
a whole (`enc.emb_src`, `dec.emb_dst`, `dec.block_out`, 6.9 GB each) are assembled in
extra buffers, freed after reporting; the computation is the same code path, so the
hook-free output is bitwise equal.

Measured peak RSS (`varda-run`, VmHWM, includes the resident pages of the mapped model
and input files): **Varda 24.4-24.7 GB**, the dev model 3.1 GB. `varda-parity` on Varda
peaks at 70 GB, most of it the pages of the two 40 GB golden files it compares against.

## 4. Numbers

CPU node, AMD EPYC 7713 (2 x 64 cores, SMT 2), gcc 12.3 `-O3 -mavx2 -mfma
-ffp-contract=off`, OpenBLAS 0.3.26, `OMP_PROC_BIND=close OMP_PLACES=threads` unless noted.
`-c 64` gives 32 physical cores with both hyperthreads; `-c 128` gives 64 cores.
All times hook-free, warm unless "first", wall clock of `predict_step`.

### Parity (64 threads, `-c 64`)

Every key is within the `cuda-fp32` table against `cpu-fp32-pyg`; the hooked and hook-free
outputs are bitwise equal. One job per model (runs A and C of the speed table).

| model | worst ratio to tolerance | worst rel_absmax | vs cuda-fp32-triton (report only), worst key |
|---|---|---|---|
| dev model | 0.101 (`proc.1.out`, 1.01e-6) | 1.01e-6 | `dec.block_out` 3.36e-6 |
| Varda | 0.181 (`dec.block_out`, 1.81e-6) | 1.81e-6 | `dec.out` 3.46e-6 |

`in.*`, `enc.x_*` and `enc.emb_dst` (a 4-wide GEMM) are bitwise. The error grows from
~2-6e-7 at the embeddings to ~7e-7 through the processor, and peaks in the decoder; it is
about half the GPU reference's own spread against the CPU reference (`cuda-fp32-triton`
vs `cpu-fp32-pyg`: 3.6e-6 on both models). Full per-key tables: `varda-parity --report`
(`report.json`), as in `docs/testing.md`.

### Speed

| model | threads | allocation | first call | warm | run |
|---|---|---|---|---|---|
| Varda | 64 | `-c 64` (32 cores + SMT), shared node | 106.2 s | 87.8-117.3 s | A |
| Varda | 128 | `-c 128` (64 cores + SMT) | 63.0 s | **61.7 s** | B |
| Varda | 64, `OMP_PLACES=cores OMP_PROC_BIND=spread` | `-c 128` (one thread per core) | 60.7 s | **60.2 s** | B |
| dev model | 64 | `-c 64` | 5.3 s | 5.25 s | C |
| dev model | 7 | `-c 64` | 35.2 s | | C |

Runs A, B and C are separate jobs (same binary); B's two rows are the same job.

For comparison (CPU fp32, 64 threads, measured separately): original anemoi 196-213 s. Load: 3.5-4.2 s for Varda with a cold page cache
(building the three CSR graphs and faulting in the weights), 0.2-1.2 s for the dev model.
The `-c 64` timings varied by 30% between calls on nodes shared with I/O-heavy jobs
(golden export); the `-c 128` run was on a quiet node and its calls agree within 0.1 s.

Where the time goes (Varda, 128 threads, warm, `VARDA_CPU_PROFILE=1`): processor 37.5 s,
decoder 18.6 s, encoder 5.5 s. By kernel: GEMMs 57 s (mlp1 + GELU 20.0, mlp2 15.7, k|v 8.2,
q|self 8.2, proj 4.7, embeddings 0.7), attention 2.1 s, LayerNorm 1.4 s, element-wise 0.4 s.
About 140 TFLOP per forward run at ~2.4 TFLOP/s; mlp1 and mlp2 have equal FLOPs, so the
difference (~4-5 s) is the scalar `erff` of the exact GELU. Levers not taken, to keep the
oracle simple: a vectorised erf, fusing the q/self and k/v weight pairs into one GEMM,
NUMA-aware first touch.

Each job kept its script, log, `report.json` and the exact binaries it ran. Earlier runs of
intermediate builds: one row block per thread with per-call allocation (Varda parity pass, worst
ratio 0.163, 103 s hook-free at 64 threads on `-c 64`), then two runs with fixed tiles and
per-call allocation.

## 5. Lessons (generic)

- **Check what the BLAS really is.** A "linked OpenBLAS" may be a serial build:
  `openblas_set_num_threads` is accepted and ignored. Benchmark one large GEMM at 1 and N
  threads before designing around it.
- **A BLAS result depends on the call shape.** Splitting a GEMM differently (by thread
  count, chunk size) changes last bits. Fixed tiles make the oracle bitwise reproducible
  across thread counts, which turns "is this difference a bug?" into a yes/no question.
- **Fault in mmap'd weights at load.** On Lustre the first forward pass paid ~14 s of page
  faults (18.6 s vs 4.4 s warm on the dev model); a parallel touch of every weight page at
  load (after `MADV_WILLNEED`) moved this into a ~1 s load.
- **Keep large activation buffers across calls.** Allocating ~23 GB per call re-faults and
  re-zeroes it every time (system time of the Varda parity run fell from 762 s to 362 s
  with the arena); an arena of named buffers costs little extra peak memory when lifetimes
  are planned (21.4 GB per-call allocation vs 24.6 GB arena, both incl. ~2.3 GB of file pages).
- **SMT does not help a GEMM-bound fp32 pass.** 64 threads on 64 cores (60.2 s) beat 128
  threads on the same cores (61.7 s); on `-c 64` the 64 "threads" are 32 cores.
- **Contract tensors that the computation never materialises** (`enc.emb_src` on all rows,
  the decoder's (G, C) tensors) are assembled only when a hook is present, from the same
  chunks, so hook and no-hook runs share every arithmetic instruction.
