#include "varda/cpu/kernels.h"

#include <cblas.h>

#include <cmath>
#include <vector>

namespace varda {
namespace cpu {

void set_blas_single_threaded() { openblas_set_num_threads(1); }

void linear_tile(const Linear& L, const float* x, int64_t ldx, int64_t rows, int64_t c0, int64_t c1, float* y,
                 int64_t ldy) {
    if (rows <= 0 || c1 <= c0) return;
    float beta = 0.0f;
    if (L.b) {
        for (int64_t r = 0; r < rows; ++r) std::memcpy(y + r * ldy + c0, L.b + c0, sizeof(float) * (c1 - c0));
        beta = 1.0f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows), static_cast<int>(c1 - c0),
                static_cast<int>(L.n_in), 1.0f, x, static_cast<int>(ldx), L.w + c0 * L.n_in, static_cast<int>(L.n_in),
                beta, y + c0, static_cast<int>(ldy));
}

void layer_norm_row(const LayerNorm& N, const float* x, float* y) {
    const int64_t n = N.dim;
    double sum = 0;
    for (int64_t c = 0; c < n; ++c) sum += x[c];
    const double mean = sum / n;
    double ss = 0;
    for (int64_t c = 0; c < n; ++c) {
        const double d = x[c] - mean;
        ss += d * d;
    }
    const float mean_f = static_cast<float>(mean);
    const float var_f = static_cast<float>(ss / n);  // biased, as torch
    const float rstd = 1.0f / std::sqrt(var_f + N.eps);
    const float shift = -rstd * mean_f;
    for (int64_t c = 0; c < n; ++c) {
        float t = x[c] * rstd + shift;
        t = t * N.w[c];
        y[c] = N.b ? t + N.b[c] : t;
    }
}

void layer_norm(const LayerNorm& N, const float* x, int64_t ldx, int64_t M, float* y, int64_t ldy) {
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < M; ++r) layer_norm_row(N, x + r * ldx, y + r * ldy);
}

float gelu(float x, bool tanh_approx) {
    if (!tanh_approx) return x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
    const float beta = static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);  // sqrt(2 / pi)
    const float kappa = 0.044715f;
    const float cube = x * x * x;
    const float inner = beta * (x + kappa * cube);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

void gelu_block(float* x, int64_t rows, int64_t cols, int64_t ld, bool tanh_approx) {
    for (int64_t r = 0; r < rows; ++r) {
        float* p = x + r * ld;
        for (int64_t c = 0; c < cols; ++c) p[c] = gelu(p[c], tanh_approx);
    }
}

namespace {

// sum_c q[c] * (k[c] + e[c]) with 8 independent partial sums (vectorisable without
// reassociation), combined pairwise.
inline float dot_q_ke(const float* q, const float* k, const float* e, int64_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t c = 0;
    for (; c + 8 <= n; c += 8)
        for (int l = 0; l < 8; ++l) acc[l] += q[c + l] * (k[c + l] + e[c + l]);
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; c < n; ++c) s += q[c] * (k[c] + e[c]);
    return s;
}

// e = attr . W^T + b for one edge (sum over the attributes first, then the bias, as addmm).
inline void edge_project(const EdgeLinear& L, const float* attr, float* e) {
    const int64_t n = L.n_out;
    const float a0 = attr[0];
    for (int64_t c = 0; c < n; ++c) e[c] = a0 * L.wt[c];
    for (int64_t a = 1; a < L.n_in; ++a) {
        const float aa = attr[a];
        const float* w = L.wt + a * n;
        for (int64_t c = 0; c < n; ++c) e[c] += aa * w[c];
    }
    if (L.b)
        for (int64_t c = 0; c < n; ++c) e[c] += L.b[c];
}

}  // namespace

void graph_attention(const CsrGraph& g, int64_t d0, int64_t nd, const float* q, int64_t ldq, const float* k,
                     const float* v, int64_t ldkv, const EdgeLinear& edge, int heads, int head_dim, float* out,
                     int64_t ldo) {
    const int64_t ct = static_cast<int64_t>(heads) * head_dim;
    const int64_t A = g.attr_dim;
    int32_t max_deg = 0;
#pragma omp parallel for reduction(max : max_deg) schedule(static)
    for (int64_t i = 0; i < nd; ++i) max_deg = std::max(max_deg, g.degree(d0 + i));
    // torch: alpha / out_channels**0.5, a Python float applied to an fp32 tensor in fp32.
    const float scale_div = static_cast<float>(std::sqrt(static_cast<double>(head_dim)));
    const float softmax_eps = 1e-16f;  // torch_geometric.utils.softmax

#pragma omp parallel
    {
        std::vector<float> e(static_cast<size_t>(max_deg) * ct);       // projected edge features
        std::vector<float> s(static_cast<size_t>(max_deg) * heads);    // scores, then weights
#pragma omp for schedule(dynamic, 32)
        for (int64_t i = 0; i < nd; ++i) {
            const int64_t lo = g.row_ptr[d0 + i];
            const int64_t deg = g.row_ptr[d0 + i + 1] - lo;
            float* o = out + i * ldo;
            std::fill(o, o + ct, 0.0f);
            if (deg == 0) continue;
            const float* qi = q + i * ldq;

            for (int64_t t = 0; t < deg; ++t) edge_project(edge, &g.attr[(lo + t) * A], &e[t * ct]);

            for (int64_t t = 0; t < deg; ++t) {
                const float* kj = k + static_cast<int64_t>(g.src[lo + t]) * ldkv;
                const float* et = &e[t * ct];
                for (int h = 0; h < heads; ++h) {
                    const int64_t off = static_cast<int64_t>(h) * head_dim;
                    s[t * heads + h] = dot_q_ke(qi + off, kj + off, et + off, head_dim) / scale_div;
                }
            }

            for (int h = 0; h < heads; ++h) {
                float m = s[h];
                for (int64_t t = 1; t < deg; ++t) m = std::max(m, s[t * heads + h]);
                float den = 0.0f;
                for (int64_t t = 0; t < deg; ++t) {
                    const float ex = std::exp(s[t * heads + h] - m);
                    s[t * heads + h] = ex;
                    den += ex;
                }
                den = den + softmax_eps;
                for (int64_t t = 0; t < deg; ++t) s[t * heads + h] = s[t * heads + h] / den;
            }

            for (int64_t t = 0; t < deg; ++t) {
                const float* vj = v + static_cast<int64_t>(g.src[lo + t]) * ldkv;
                const float* et = &e[t * ct];
                for (int h = 0; h < heads; ++h) {
                    const float a = s[t * heads + h];
                    const int64_t off = static_cast<int64_t>(h) * head_dim;
                    for (int64_t c = off; c < off + head_dim; ++c) o[c] += (vj[c] + et[c]) * a;
                }
            }
        }
    }
}

}  // namespace cpu
}  // namespace varda
