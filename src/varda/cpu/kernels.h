// CPU fp32 kernels of the oracle backend: Linear (cblas_sgemm), LayerNorm, GELU and the
// GraphTransformer attention. Each mirrors the op sequence of the Python engine's plain path
// ($IE_ROOT/engine/blocks.py, engine/ops.py), which is bitwise equal to anemoi's pyg backend
// on CPU. See docs/cpu.md.
//
// Threading: every kernel parallelises with OpenMP over rows, tiles or destination nodes,
// using the calling thread's OpenMP thread count. GEMMs are split into fixed tiles, each a
// single-threaded cblas_sgemm call (set_blas_single_threaded(); OpenBLAS's own threading is
// never used). Every output element is computed by exactly one thread, in an order that does
// not depend on the schedule: results are bitwise independent of the thread count.
#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "varda/graph.h"

namespace varda {
namespace cpu {

// nn.Linear: weight [n_out, n_in] row-major (as stored), optional bias [n_out].
struct Linear {
    const float* w = nullptr;
    const float* b = nullptr;  // null: bias=False
    int64_t n_out = 0, n_in = 0;
};

// nn.LayerNorm over the last dim: weight [dim], optional bias [dim].
struct LayerNorm {
    const float* w = nullptr;
    const float* b = nullptr;  // null: bias=False
    int64_t dim = 0;
    float eps = 0.0f;
};

// lin_edge applied on the fly: e[c] = sum_a attr[a] * wt[a * n_out + c] + b[c].
struct EdgeLinear {
    const float* wt = nullptr;  // [n_in, n_out]: the transposed Linear weight (owned by the caller)
    const float* b = nullptr;   // [n_out] or null
    int64_t n_in = 0, n_out = 0;
};

// Every cblas_sgemm call runs on the calling thread (one call per tile, see below).
void set_blas_single_threaded();

// GEMM tile of the parallel Linear: rows x output columns. The tile grid is fixed (it does not
// depend on the thread count), and OpenBLAS's result for an element depends on the shape of
// the call that computes it, so fixed tiles make every result independent of the number of
// threads. 128 x 512 was the fastest tile on an EPYC 7713 node (docs/cpu.md).
constexpr int64_t kTileRows = 128, kTileCols = 512;

// y[r, c0:c1] = x[r, :] . W[c0:c1, :]^T (+ b[c0:c1]) for `rows` rows, single-threaded. With a
// bias, y is first filled with b and sgemm accumulates into it (beta = 1), so the sum over the
// inputs is rounded before the bias is added, as in torch's addmm.
void linear_tile(const Linear& L, const float* x, int64_t ldx, int64_t rows, int64_t c0, int64_t c1, float* y,
                 int64_t ldy);

// Linear on M rows: y[r, 0:n_out] = x[r, 0:n_in] . W^T + b, tiles in parallel (dynamic schedule).
// epi(r0, r1, c0, c1) runs on each finished tile on the same thread (fused epilogue: residual
// add, GELU, ...); it must be element-wise.
template <class Epi>
void linear(const Linear& L, const float* x, int64_t ldx, int64_t M, float* y, int64_t ldy, Epi&& epi) {
    const int64_t nr = (M + kTileRows - 1) / kTileRows, nc = (L.n_out + kTileCols - 1) / kTileCols;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t t = 0; t < nr * nc; ++t) {
        const int64_t r0 = (t / nc) * kTileRows, r1 = std::min(M, r0 + kTileRows);
        const int64_t c0 = (t % nc) * kTileCols, c1 = std::min(L.n_out, c0 + kTileCols);
        linear_tile(L, x + r0 * ldx, ldx, r1 - r0, c0, c1, y + r0 * ldy, ldy);
        epi(r0, r1, c0, c1);
    }
}
inline void linear(const Linear& L, const float* x, int64_t ldx, int64_t M, float* y, int64_t ldy) {
    linear(L, x, ldx, M, y, ldy, [](int64_t, int64_t, int64_t, int64_t) {});
}

// LayerNorm on M rows (parallel). Moments in double; then, in fp32 as torch's CPU kernel:
// rstd = 1 / sqrt(var + eps), y = (x * rstd + (-rstd * mean)) * w + b.
void layer_norm(const LayerNorm& N, const float* x, int64_t ldx, int64_t M, float* y, int64_t ldy);
void layer_norm_row(const LayerNorm& N, const float* x, float* y);

// GELU, in place on rows x cols (leading dim ld), single-threaded (call from an epilogue).
// Exact: x * 0.5 * (1 + erf(x * M_SQRT1_2)); tanh: 0.5 * x * (1 + tanh(b * (x + 0.044715 x^3))),
// b = sqrt(2 / pi) (torch's formulas, fp32).
void gelu_block(float* x, int64_t rows, int64_t cols, int64_t ld, bool tanh_approx);
float gelu(float x, bool tanh_approx);

// GraphTransformer attention for destinations [d0, d0 + nd) of g (CSR by destination), in
// parallel over destinations. Per destination i, head h, incoming edge j -> i (CSR order):
//   e_ij   = lin_edge(attr_ij)                       (C_tot = heads * head_dim channels)
//   s_ij   = sum_c q[i,hc] * (k[j,hc] + e_ij[hc]) / sqrt(head_dim)
//   a_ij   = exp(s_ij - max_j s_ij) / (sum_j exp(...) + 1e-16)        (torch_geometric softmax)
//   out[i] = sum_j (v[j,hc] + e_ij[hc]) * a_ij, summed in CSR (= stored edge) order
// A destination without incoming edges gets zeros. q rows are local (row 0 = destination d0),
// k/v rows are global source ids. `out` has nd rows of C_tot.
void graph_attention(const CsrGraph& g, int64_t d0, int64_t nd, const float* q, int64_t ldq, const float* k,
                     const float* v, int64_t ldkv, const EdgeLinear& edge, int heads, int head_dim, float* out,
                     int64_t ldo);

}  // namespace cpu
}  // namespace varda
