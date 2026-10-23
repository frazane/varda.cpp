// Internal CUDA kernels of the varda backend (docs/cuda.md §3). Every launcher enqueues on `s`
// and never synchronises or allocates. Element type T of activations: float (fp32 mode) or
// __half (fp16 mode). Row-major matrices take a leading dimension (elements).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "varda/cuda/common.h"

namespace varda::cuda {

// ---------------------------------------------------------------- normaliser

// xn[i] = (x[i] * mul[v]) + add[v], two separately rounded fp32 ops; v = i % V.
void normalise(const float* x, const float* mul, const float* add, float* xn, int64_t n, int V, cudaStream_t s);
// y[i] = (yn[i] - add[v]) / mul[v], separately rounded (IEEE division); v = i % V.
void denormalise(const float* yn, const float* add, const float* mul, float* y, int64_t n, int V, cudaStream_t s);

// ---------------------------------------------------------------- input assembly

// out[g, t*V + v] = xn[t, g, v]; out[g, T*V + a] = attr[g, a]; out[g, F..ld) = 0, F = T*V + A.
template <class T>
void assemble_x_data(const float* xn, const float* attr, int T_in, int64_t G, int V, int A, T* out, int ld,
                     cudaStream_t s);

// ---------------------------------------------------------------- conversions

// dst[r, c] = (Tout) src[r, c] for c < cols; rows x cols with leading dimensions.
template <class Tin, class Tout>
void convert_2d(const Tin* src, int64_t ld_src, Tout* dst, int64_t ld_dst, int64_t rows, int64_t cols,
                cudaStream_t s);

// ---------------------------------------------------------------- LayerNorm (fp32 statistics)

// y = LN(x) * w + b over C channels per row; y may alias x. C must be a multiple of 256.
template <class T>
void layernorm(const T* x, int64_t ldx, const float* w, const float* b, T* y, int64_t ldy, int64_t rows, int C,
               float eps, cudaStream_t s);

// out1 = T(y + r1); if r2: out2 = T(out1 + r2); ln = LN(last of out1/out2) (if ln_w).
// All [rows, C] with leading dimension ld (same for every operand). Each add is computed in fp32
// and rounded to T, as separate reference ops would be.
template <class T>
void add_layernorm(const T* y, const T* r1, T* out1, const T* r2, T* out2, const float* ln_w, const float* ln_b,
                   T* ln, int64_t ld, int64_t rows, int C, float eps, cudaStream_t s);

// ---------------------------------------------------------------- GELU

// In place over n contiguous elements (n % 8 == 0): x = gelu(x), exact erf or tanh form.
template <class T>
void gelu_inplace(T* x, int64_t n, bool tanh_approx, cudaStream_t s);

// ---------------------------------------------------------------- graph attention

// Destination-sorted graph on the device. edges[slot] = {a0, a1, a2, bitcast(src)} for CSR slot
// `slot` (graph.h order), attributes zero-padded to 3.
struct DeviceGraph {
    int64_t n_src = 0, n_dst = 0, n_edges = 0;
    int attr_dim = 0;
    int max_degree = 0;
    const int32_t* row_ptr = nullptr;  // [n_dst + 1]
    const float4* edges = nullptr;     // [n_edges]
};

struct AttentionArgs {
    DeviceGraph g;
    int64_t dst_lo = 0, n_rows = 0;  // destinations [dst_lo, dst_lo + n_rows); q/xr/out rows are local
    const void* q = nullptr; int64_t ldq = 0;
    const void* k = nullptr; int64_t ldk = 0;  // rows = source ids
    const void* v = nullptr; int64_t ldv = 0;
    const void* xr = nullptr; int64_t ldxr = 0;
    void* out = nullptr; int64_t ldo = 0;
    const float* w_e = nullptr;  // [4, C]: edge Linear weight columns W_e[:, 0..2] (zero padded), then b_e
    int C = 0, heads = 0, head_dim = 0;
};

// out[i] = T(T(attn_i) + xr[i]) where attn_i = sum_j softmax_j(<q_i, k_j + e_ij>/sqrt(d)) (v_j + e_ij)
// per head, e_ij = W_e a_ij + b_e folded in (O(E*H*3)), fp32 online softmax; zero in-degree -> attn = 0.
// `vpl`: 8-channel vectors per lane (1, 2, 4); a destination takes C / (256 vpl) warps. The result
// does not depend on it (same per-head arithmetic).
template <class T>
void attention(const AttentionArgs& a, cudaStream_t s, int vpl = 1);

// ---------------------------------------------------------------- output

struct BoundingDev {
    int type = 0;           // 0 relu, 1 hardtanh, 2 fraction
    int n = 0;              // index count
    int idx_off = 0;        // into the shared index array
    int tot_off = 0, n_tot = 0;
    float lo = 0, hi = 0;
};

struct OutputArgs {
    const void* dec = nullptr; int64_t ld_dec = 0;  // [G, V_out] decoder output (T)
    const float* xn_step = nullptr;  // xn[residual_step] = [G, V_in]
    int V_in = 0, V_out = 0;
    int64_t G = 0;
    const int* res_src = nullptr;  // [V_out]: model-input index added to output v, or -1
    const BoundingDev* bounds = nullptr; int n_bounds = 0;
    const int* bidx = nullptr;     // index lists of the boundings
    float* pre_bound = nullptr;    // optional [G, V_out]
    float* y = nullptr;            // [G, V_out] normalised, after bounding
};
template <class T>
void output_assemble(const OutputArgs& a, cudaStream_t s);

}  // namespace varda::cuda
