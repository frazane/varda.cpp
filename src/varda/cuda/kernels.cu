// CUDA kernels of the varda backend (see kernels.h and docs/cuda.md §3).
#include <cuda_fp16.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "varda/cuda/kernels.h"

namespace varda::cuda {

namespace {

constexpr unsigned kFull = 0xffffffffu;

// ------------------------------------------------------------------ 8-channel vectors
// A "vector" is 8 consecutive channels: one 16-byte load in fp16, two in fp32.

template <class T> struct Raw;
template <> struct Raw<__half> { uint4 u; };
template <> struct Raw<float> { float4 a, b; };

template <class T> __device__ __forceinline__ Raw<T> ldg8(const T* p);
template <> __device__ __forceinline__ Raw<__half> ldg8(const __half* p) {
    Raw<__half> r;
    r.u = __ldg(reinterpret_cast<const uint4*>(p));
    return r;
}
template <> __device__ __forceinline__ Raw<float> ldg8(const float* p) {
    Raw<float> r;
    r.a = __ldg(reinterpret_cast<const float4*>(p));
    r.b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    return r;
}
template <class T> __device__ __forceinline__ Raw<T> ld8(const T* p);
template <> __device__ __forceinline__ Raw<__half> ld8(const __half* p) {
    Raw<__half> r;
    r.u = *reinterpret_cast<const uint4*>(p);
    return r;
}
template <> __device__ __forceinline__ Raw<float> ld8(const float* p) {
    Raw<float> r;
    r.a = reinterpret_cast<const float4*>(p)[0];
    r.b = reinterpret_cast<const float4*>(p)[1];
    return r;
}

__device__ __forceinline__ void unpack(const Raw<__half>& r, float (&f)[8]) {
    const __half2* h = reinterpret_cast<const __half2*>(&r.u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 t = __half22float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void unpack(const Raw<float>& r, float (&f)[8]) {
    f[0] = r.a.x; f[1] = r.a.y; f[2] = r.a.z; f[3] = r.a.w;
    f[4] = r.b.x; f[5] = r.b.y; f[6] = r.b.z; f[7] = r.b.w;
}

template <class T> __device__ __forceinline__ void load8(const T* p, float (&f)[8]) { unpack(ld8(p), f); }

__device__ __forceinline__ void store8(__half* p, const float (&f)[8]) {
    uint4 u;
    __half2* h = reinterpret_cast<__half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void store8(float* p, const float (&f)[8]) {
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

// Round to the storage type and back (identity in fp32).
template <class T> __device__ __forceinline__ float rnd(float x);
template <> __device__ __forceinline__ float rnd<float>(float x) { return x; }
template <> __device__ __forceinline__ float rnd<__half>(float x) { return __half2float(__float2half_rn(x)); }

template <class T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <class T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half_rn(x); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(kFull, x, o);
    return x;
}

inline unsigned blocks_for(int64_t n, int per_block) { return static_cast<unsigned>((n + per_block - 1) / per_block); }

// ------------------------------------------------------------------ normaliser

// Index type I: 32-bit arithmetic when the tensor allows it (the modulo dominates otherwise).
template <class I>
__global__ void normalise_kernel(const float* __restrict__ x, const float* __restrict__ mul,
                                 const float* __restrict__ add, float* __restrict__ xn, I n, I V) {
    for (I i = blockIdx.x * (I)blockDim.x + threadIdx.x; i < n; i += (I)gridDim.x * blockDim.x) {
        const I v = i % V;
        xn[i] = __fadd_rn(__fmul_rn(x[i], __ldg(mul + v)), __ldg(add + v));
    }
}

template <class I>
__global__ void denormalise_kernel(const float* __restrict__ yn, const float* __restrict__ add,
                                   const float* __restrict__ mul, float* __restrict__ y, I n, I V) {
    for (I i = blockIdx.x * (I)blockDim.x + threadIdx.x; i < n; i += (I)gridDim.x * blockDim.x) {
        const I v = i % V;
        y[i] = __fdiv_rn(__fsub_rn(yn[i], __ldg(add + v)), __ldg(mul + v));
    }
}

unsigned grid_1d(int64_t n) {
    const int64_t b = (n + 255) / 256;
    return static_cast<unsigned>(b < 65535 * 16 ? (b > 0 ? b : 1) : 65535 * 16);
}
constexpr int64_t kI32 = (int64_t(1) << 31) - (int64_t(1) << 29);  // headroom > the largest grid stride (2^28)

// ------------------------------------------------------------------ assembly / conversions

template <class T, class I>
__global__ void assemble_kernel(const float* __restrict__ xn, const float* __restrict__ attr, int T_in, I G, I V,
                                I A, T* __restrict__ out, I ld) {
    const I n = G * ld, TV = T_in * V;
    for (I i = blockIdx.x * (I)blockDim.x + threadIdx.x; i < n; i += (I)gridDim.x * blockDim.x) {
        const I g = i / ld;
        const I f = i - g * ld;
        float val = 0.0f;
        if (f < TV) {
            const I t = f / V, v = f - t * V;
            val = xn[(t * G + g) * V + v];
        } else if (f < TV + A) {
            val = attr[g * A + (f - TV)];
        }
        out[i] = from_f<T>(val);
    }
}

template <class Tin, class Tout>
__global__ void convert_kernel(const Tin* __restrict__ src, int64_t ld_src, Tout* __restrict__ dst, int64_t ld_dst,
                               int64_t rows, int64_t cols) {
    const int64_t n = rows * cols;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / cols, c = i % cols;
        dst[r * ld_dst + c] = from_f<Tout>(to_f<Tin>(src[r * ld_src + c]));
    }
}

// ------------------------------------------------------------------ LayerNorm
// One warp per row; lane holds vectors lane + 32 i (i < VPL), i.e. channels 8 (lane + 32 i) .. +8.

template <int VPL>
__device__ __forceinline__ void ln_rows(float (&x)[VPL][8], const float* __restrict__ w, const float* __restrict__ b,
                                        int lane, float eps) {
    constexpr int C = 256 * VPL;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < VPL; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) s += x[i][j];
    const float mean = warp_sum(s) / C;
    float q = 0.0f;
#pragma unroll
    for (int i = 0; i < VPL; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float d = x[i][j] - mean;
            q += d * d;
        }
    const float var = warp_sum(q) / C;
    const float rstd = 1.0f / sqrtf(var + eps);
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int c0 = (lane + 32 * i) * 8;
        const float4 w0 = __ldg(reinterpret_cast<const float4*>(w + c0));
        const float4 w1 = __ldg(reinterpret_cast<const float4*>(w + c0) + 1);
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(b + c0));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(b + c0) + 1);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) x[i][j] = (x[i][j] - mean) * rstd * wv[j] + bv[j];
    }
}

template <class T, int VPL>
__global__ void __launch_bounds__(256) layernorm_kernel(const T* x, int64_t ldx, const float* __restrict__ w,
                                                        const float* __restrict__ b, T* y, int64_t ldy, int64_t rows,
                                                        float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    float v[VPL][8];
#pragma unroll
    for (int i = 0; i < VPL; ++i) load8(x + r * ldx + (lane + 32 * i) * 8, v[i]);
    ln_rows<VPL>(v, w, b, lane, eps);
#pragma unroll
    for (int i = 0; i < VPL; ++i) store8(y + r * ldy + (lane + 32 * i) * 8, v[i]);
}

template <class T, int VPL>
__global__ void __launch_bounds__(256) add_layernorm_kernel(const T* y, const T* r1, T* out1, const T* r2, T* out2,
                                                            const float* __restrict__ w, const float* __restrict__ b,
                                                            T* ln, int64_t ld, int64_t rows, float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    float v[VPL][8];
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int64_t off = r * ld + (lane + 32 * i) * 8;
        float a[8], c[8];
        load8(y + off, a);
        load8(r1 + off, c);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[i][j] = rnd<T>(a[j] + c[j]);
        store8(out1 + off, v[i]);
        if (r2) {
            load8(r2 + off, c);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[i][j] = rnd<T>(v[i][j] + c[j]);
            store8(out2 + off, v[i]);
        }
    }
    if (!ln) return;
    ln_rows<VPL>(v, w, b, lane, eps);
#pragma unroll
    for (int i = 0; i < VPL; ++i) store8(ln + r * ld + (lane + 32 * i) * 8, v[i]);
}

// ------------------------------------------------------------------ GELU

__device__ __forceinline__ float gelu_f(float x, bool tanh_approx) {
    if (!tanh_approx) {
        constexpr float kAlpha = 0.70710678118654752440f;  // M_SQRT1_2
        return x * 0.5f * (1.0f + erff(x * kAlpha));
    }
    constexpr float kBeta = 0.79788456080286535588f;  // sqrt(2) * 2/sqrt(pi) * 0.5
    constexpr float kKappa = 0.044715f;
    const float inner = kBeta * (x + kKappa * (x * x * x));
    return 0.5f * x * (1.0f + tanhf(inner));
}

template <class T>
__global__ void gelu_kernel(T* x, int64_t n8, bool tanh_approx) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n8; i += (int64_t)gridDim.x * blockDim.x) {
        float f[8];
        load8(x + i * 8, f);
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = gelu_f(f[j], tanh_approx);
        store8(x + i * 8, f);
    }
}

// ------------------------------------------------------------------ attention
// A destination's C channels are split over WPD = C / (256 VPL) warps; each warp holds 256 VPL
// channels: lane holds vectors lane + 32 i (i < VPL) of its part. Vector i of a lane lies in head
// (channel / head_dim), and LPH = head_dim / 8 consecutive lanes share a head, so per-head dot
// products are a butterfly over LPH lanes. Online softmax in fp32 (as the reference triton kernel);
// the warps of a destination compute the same scores for different heads, no communication.

template <int LPH>
__device__ __forceinline__ float head_sum(float x) {
#pragma unroll
    for (int o = 1; o < LPH; o <<= 1) x += __shfl_xor_sync(kFull, x, o);
    return x;
}

template <class T, int VPL, int LPH, int U>
__global__ void __launch_bounds__(256) attention_kernel(AttentionArgs a, float scale, int wpd) {
    const int lane = threadIdx.x & 31;
    const int64_t gw = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t r = gw / wpd;
    if (r >= a.n_rows) return;
    const int part = static_cast<int>(gw - r * wpd);
    const int base = part * 256 * VPL + lane * 8;  // channel of vector 0; vector i at base + 256 i
    const int64_t d = a.dst_lo + r;
    const int beg = __ldg(a.g.row_ptr + d), end = __ldg(a.g.row_ptr + d + 1);
    const T* __restrict__ q = static_cast<const T*>(a.q) + r * a.ldq + base;
    const T* __restrict__ K = static_cast<const T*>(a.k) + base;
    const T* __restrict__ Vv = static_cast<const T*>(a.v) + base;

    float qv[VPL][8];
#pragma unroll
    for (int i = 0; i < VPL; ++i) unpack(ldg8(q + 256 * i), qv[i]);

    // Folded edge projection: qe[i][f] = sum_{c in head} q_c W_e[c, f], qb[i] = sum_{c in head} q_c b_e[c].
    // w_e is attribute-major [4, C] (rows: W_e[:, 0..2], b_e), so these loads are coalesced.
    const int C = a.C;
    float qe[VPL][3], qb[VPL];
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int c0 = base + 256 * i;
        float w[4][8];
#pragma unroll
        for (int f = 0; f < 4; ++f) unpack(ldg8(a.w_e + f * C + c0), w[f]);
        float s0 = 0, s1 = 0, s2 = 0, sb = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s0 = __fmaf_rn(qv[i][j], w[0][j], s0);
            s1 = __fmaf_rn(qv[i][j], w[1][j], s1);
            s2 = __fmaf_rn(qv[i][j], w[2][j], s2);
            sb = __fmaf_rn(qv[i][j], w[3][j], sb);
        }
        qe[i][0] = head_sum<LPH>(s0);
        qe[i][1] = head_sum<LPH>(s1);
        qe[i][2] = head_sum<LPH>(s2);
        qb[i] = head_sum<LPH>(sb);
    }

    float m[VPL], l[VPL], acc[VPL][8], pa[VPL][3];
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        pa[i][0] = pa[i][1] = pa[i][2] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    }

    // Explicit fused multiply-adds throughout: the arithmetic (hence the bits) does not depend on
    // how the compiler contracts each template instance.
    auto step = [&](const float4& ed, const Raw<T> (&kr)[VPL], const Raw<T> (&vr)[VPL]) {
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
            float kf[8];
            unpack(kr[i], kf);
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) dot = __fmaf_rn(qv[i][j], kf[j], dot);
            dot = head_sum<LPH>(dot);
            const float se = __fmaf_rn(qe[i][2], ed.z, __fmaf_rn(qe[i][1], ed.y, __fmul_rn(qe[i][0], ed.x)));
            const float s = __fmul_rn(__fadd_rn(__fadd_rn(dot, se), qb[i]), scale);
            const float mn = fmaxf(m[i], s);
            const float corr = expf(__fsub_rn(m[i], mn));
            const float p = expf(__fsub_rn(s, mn));
            m[i] = mn;
            l[i] = __fmaf_rn(l[i], corr, p);
            float vf[8];
            unpack(vr[i], vf);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(p, vf[j], __fmul_rn(acc[i][j], corr));
            pa[i][0] = __fmaf_rn(p, ed.x, __fmul_rn(pa[i][0], corr));
            pa[i][1] = __fmaf_rn(p, ed.y, __fmul_rn(pa[i][1], corr));
            pa[i][2] = __fmaf_rn(p, ed.z, __fmul_rn(pa[i][2], corr));
        }
    };

    // U edges in flight per iteration (predicated tail, so low degrees also overlap their loads).
    for (int e = beg; e < end; e += U) {
        float4 ed[U];
        Raw<T> kr[U][VPL], vr[U][VPL];
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (e + u < end) ed[u] = __ldg(a.g.edges + e + u);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (e + u < end) {
                const int64_t sj = __float_as_int(ed[u].w);
#pragma unroll
                for (int i = 0; i < VPL; ++i) {
                    kr[u][i] = ldg8(K + sj * a.ldk + 256 * i);
                    vr[u][i] = ldg8(Vv + sj * a.ldv + 256 * i);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (e + u < end) step(ed[u], kr[u], vr[u]);
    }

    const T* xr = static_cast<const T*>(a.xr) + r * a.ldxr + base;
    T* out = static_cast<T*>(a.out) + r * a.ldo + base;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int c0 = base + 256 * i;
        float o[8], x[8];
        load8(xr + 256 * i, x);
        if (end > beg) {
            const float inv = __fdiv_rn(1.0f, l[i]);
            const float p0 = __fmul_rn(pa[i][0], inv), p1 = __fmul_rn(pa[i][1], inv), p2 = __fmul_rn(pa[i][2], inv);
            float w[4][8];
#pragma unroll
            for (int f = 0; f < 4; ++f) unpack(ldg8(a.w_e + f * C + c0), w[f]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float ev = __fadd_rn(__fmaf_rn(w[2][j], p2, __fmaf_rn(w[1][j], p1, __fmul_rn(w[0][j], p0))), w[3][j]);
                o[j] = __fmaf_rn(acc[i][j], inv, ev);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = rnd<T>(rnd<T>(o[j]) + x[j]);
        store8(out + 256 * i, o);
    }
}

// ------------------------------------------------------------------ output assembly

template <class T>
__global__ void output_kernel(OutputArgs a) {
    const int lane = threadIdx.x & 31;
    const int64_t g = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= a.G) return;
    const T* dec = static_cast<const T*>(a.dec) + g * a.ld_dec;
    float* y = a.y + g * a.V_out;
    for (int v = lane; v < a.V_out; v += 32) {
        float val = to_f<T>(dec[v]);
        const int rs = a.res_src[v];
        if (rs >= 0) val = val + a.xn_step[g * a.V_in + rs];
        y[v] = val;
        if (a.pre_bound) a.pre_bound[g * a.V_out + v] = val;
    }
    __syncwarp();
    // Boundings in config order, normalised space, in place. Lane 0 applies them
    // sequentially; index lists are short.
    if (lane != 0) return;
    for (int bi = 0; bi < a.n_bounds; ++bi) {
        const BoundingDev bd = a.bounds[bi];
        const int* idx = a.bidx + bd.idx_off;
        if (bd.type == 0) {
            for (int j = 0; j < bd.n; ++j) {
                const float x = y[idx[j]];
                y[idx[j]] = x < 0.0f ? 0.0f : x;  // relu (NaN propagates)
            }
        } else {
            for (int j = 0; j < bd.n; ++j) {
                const float x = y[idx[j]];
                y[idx[j]] = x < bd.lo ? bd.lo : (x > bd.hi ? bd.hi : x);  // hardtanh (NaN propagates)
            }
            if (bd.type == 2) {
                // y[..., idx] *= y[..., tot]: the right-hand side is gathered after the clamp, before any
                // product (the backend rejects overlapping index / total_index lists at load).
                const int* tot = a.bidx + bd.tot_off;
                for (int j = 0; j < bd.n; ++j) y[idx[j]] = y[idx[j]] * y[tot[bd.n_tot == 1 ? 0 : j]];
            }
        }
    }
}

}  // namespace

// ================================================================== launchers

void normalise(const float* x, const float* mul, const float* add, float* xn, int64_t n, int V, cudaStream_t s) {
    if (n < kI32) normalise_kernel<int32_t><<<grid_1d(n), 256, 0, s>>>(x, mul, add, xn, int32_t(n), V);
    else normalise_kernel<int64_t><<<grid_1d(n), 256, 0, s>>>(x, mul, add, xn, n, V);
    VARDA_CUDA_LAUNCH();
}

void denormalise(const float* yn, const float* add, const float* mul, float* y, int64_t n, int V, cudaStream_t s) {
    if (n < kI32) denormalise_kernel<int32_t><<<grid_1d(n), 256, 0, s>>>(yn, add, mul, y, int32_t(n), V);
    else denormalise_kernel<int64_t><<<grid_1d(n), 256, 0, s>>>(yn, add, mul, y, n, V);
    VARDA_CUDA_LAUNCH();
}

template <class T>
void assemble_x_data(const float* xn, const float* attr, int T_in, int64_t G, int V, int A, T* out, int ld,
                     cudaStream_t s) {
    // 32-bit indexing needs every index, including T_in * G * V of the input, below 2^31
    if (G * ld < kI32 && int64_t(T_in) * G * V < kI32)
        assemble_kernel<T, int32_t><<<grid_1d(G * ld), 256, 0, s>>>(xn, attr, T_in, int32_t(G), V, A, out, ld);
    else
        assemble_kernel<T, int64_t><<<grid_1d(G * ld), 256, 0, s>>>(xn, attr, T_in, G, V, A, out, ld);
    VARDA_CUDA_LAUNCH();
}

template <class Tin, class Tout>
void convert_2d(const Tin* src, int64_t ld_src, Tout* dst, int64_t ld_dst, int64_t rows, int64_t cols,
                cudaStream_t s) {
    convert_kernel<Tin, Tout><<<grid_1d(rows * cols), 256, 0, s>>>(src, ld_src, dst, ld_dst, rows, cols);
    VARDA_CUDA_LAUNCH();
}

namespace {
[[noreturn]] void bad_channels(int C) {
    throw std::runtime_error("CUDA backend: num_channels " + std::to_string(C) +
                             " is not supported (kernels take a multiple of 256 up to 2048)");
}
}  // namespace

template <class T>
void layernorm(const T* x, int64_t ldx, const float* w, const float* b, T* y, int64_t ldy, int64_t rows, int C,
               float eps, cudaStream_t s) {
    if (rows <= 0) return;
    const unsigned grid = blocks_for(rows, 8);
#define VARDA_LN(VPL) layernorm_kernel<T, VPL><<<grid, 256, 0, s>>>(x, ldx, w, b, y, ldy, rows, eps)
    switch (C) {  // every multiple of 256 up to 2048 (the backend's load-time check)
        case 256: VARDA_LN(1); break;
        case 512: VARDA_LN(2); break;
        case 768: VARDA_LN(3); break;
        case 1024: VARDA_LN(4); break;
        case 1280: VARDA_LN(5); break;
        case 1536: VARDA_LN(6); break;
        case 1792: VARDA_LN(7); break;
        case 2048: VARDA_LN(8); break;
#undef VARDA_LN
        default: bad_channels(C);
    }
    VARDA_CUDA_LAUNCH();
}

template <class T>
void add_layernorm(const T* y, const T* r1, T* out1, const T* r2, T* out2, const float* ln_w, const float* ln_b,
                   T* ln, int64_t ld, int64_t rows, int C, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    const unsigned grid = blocks_for(rows, 8);
#define VARDA_ADDLN(VPL) \
    add_layernorm_kernel<T, VPL><<<grid, 256, 0, s>>>(y, r1, out1, r2, out2, ln_w, ln_b, ln, ld, rows, eps)
    switch (C) {
        case 256: VARDA_ADDLN(1); break;
        case 512: VARDA_ADDLN(2); break;
        case 768: VARDA_ADDLN(3); break;
        case 1024: VARDA_ADDLN(4); break;
        case 1280: VARDA_ADDLN(5); break;
        case 1536: VARDA_ADDLN(6); break;
        case 1792: VARDA_ADDLN(7); break;
        case 2048: VARDA_ADDLN(8); break;
        default: bad_channels(C);
    }
#undef VARDA_ADDLN
    VARDA_CUDA_LAUNCH();
}

template <class T>
void gelu_inplace(T* x, int64_t n, bool tanh_approx, cudaStream_t s) {
    if (n % 8) throw std::runtime_error("gelu_inplace: n must be a multiple of 8");
    gelu_kernel<T><<<grid_1d(n / 8), 256, 0, s>>>(x, n / 8, tanh_approx);
    VARDA_CUDA_LAUNCH();
}

template <class T>
void attention(const AttentionArgs& a, cudaStream_t s, int vpl) {
    if (a.n_rows <= 0) return;
    if (a.head_dim % 8 || a.C != a.heads * a.head_dim)
        throw std::runtime_error("attention: inconsistent heads/head_dim/C");
    if (a.C % 256) bad_channels(a.C);
    if (vpl <= 0) vpl = 1;
    while (vpl > 1 && (a.C % (256 * vpl))) vpl >>= 1;
    const int wpd = a.C / (256 * vpl);
    const float scale = 1.0f / std::sqrt(static_cast<float>(a.head_dim));
    const unsigned grid = blocks_for(a.n_rows * wpd, 8);
    const int lph = a.head_dim / 8;
// 2 edges in flight per warp: 4 (103 registers) measured 2% slower on Varda.
#define VARDA_ATT(VPL, LPH) attention_kernel<T, VPL, LPH, 2><<<grid, 256, 0, s>>>(a, scale, wpd)
#define VARDA_ATT_C(VPL)                                                         \
    switch (lph) {                                                               \
        case 1: VARDA_ATT(VPL, 1); break;                                        \
        case 2: VARDA_ATT(VPL, 2); break;                                        \
        case 4: VARDA_ATT(VPL, 4); break;                                        \
        case 8: VARDA_ATT(VPL, 8); break;                                        \
        case 16: VARDA_ATT(VPL, 16); break;                                      \
        case 32: VARDA_ATT(VPL, 32); break;                                      \
        default: throw std::runtime_error("attention: head_dim must be 8 * 2^k <= 256"); \
    }
    switch (vpl) {
        case 1: VARDA_ATT_C(1); break;
        case 2: VARDA_ATT_C(2); break;
        case 4: VARDA_ATT_C(4); break;
        default: throw std::runtime_error("attention: vectors per lane must be 1, 2 or 4");
    }
#undef VARDA_ATT_C
#undef VARDA_ATT
    VARDA_CUDA_LAUNCH();
}

template <class T>
void output_assemble(const OutputArgs& a, cudaStream_t s) {
    if (a.G <= 0) return;
    output_kernel<T><<<blocks_for(a.G, 8), 256, 0, s>>>(a);
    VARDA_CUDA_LAUNCH();
}

// ------------------------------------------------------------------ instantiations
#define VARDA_INST(T)                                                                                          \
    template void assemble_x_data<T>(const float*, const float*, int, int64_t, int, int, T*, int, cudaStream_t); \
    template void layernorm<T>(const T*, int64_t, const float*, const float*, T*, int64_t, int64_t, int, float,   \
                               cudaStream_t);                                                                  \
    template void add_layernorm<T>(const T*, const T*, T*, const T*, T*, const float*, const float*, T*, int64_t, \
                                   int64_t, int, float, cudaStream_t);                                         \
    template void gelu_inplace<T>(T*, int64_t, bool, cudaStream_t);                                            \
    template void attention<T>(const AttentionArgs&, cudaStream_t, int);                                            \
    template void output_assemble<T>(const OutputArgs&, cudaStream_t);
VARDA_INST(float)
VARDA_INST(__half)
#undef VARDA_INST

template void convert_2d<float, float>(const float*, int64_t, float*, int64_t, int64_t, int64_t, cudaStream_t);
template void convert_2d<float, __half>(const float*, int64_t, __half*, int64_t, int64_t, int64_t, cudaStream_t);
template void convert_2d<__half, float>(const __half*, int64_t, float*, int64_t, int64_t, int64_t, cudaStream_t);

}  // namespace varda::cuda
