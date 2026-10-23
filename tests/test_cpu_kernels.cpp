// Unit tests of the CPU fp32 kernels (src/varda/cpu/kernels.h) against naive double-precision
// implementations: Linear, LayerNorm, GELU (erf and tanh) and the graph attention, including
// destinations without incoming edges, a destination sub-range, and invariance of the
// results to the thread count.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "varda/cpu/kernels.h"
#include "varda/format.h"
#include "varda/graph.h"

using namespace varda;

namespace {

int failures = 0;

void check(bool ok, const char* what, double err, double tol) {
    std::printf("%-48s err %.3e tol %.1e %s\n", what, err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// max |a - r| / max |r|, the contract metric.
double rel_absmax(const std::vector<float>& a, const std::vector<double>& r) {
    double m = 0, am = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (std::isnan(a[i])) return INFINITY;
        m = std::max(m, std::fabs(a[i] - r[i]));
        am = std::max(am, std::fabs(r[i]));
    }
    return am > 0 ? m / am : m;
}

std::vector<float> randn(size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

void test_linear(std::mt19937& rng) {
    const int64_t M = 301, K = 67, N = 45, ldy = 50;
    auto x = randn(M * K, rng), w = randn(N * K, rng), b = randn(N, rng);
    std::vector<float> y(M * ldy, -7.0f);
    cpu::Linear L{w.data(), b.data(), N, K};
    cpu::linear(L, x.data(), K, M, y.data(), ldy);
    std::vector<float> yc;
    std::vector<double> ref;
    bool pad_ok = true;
    for (int64_t r = 0; r < M; ++r)
        for (int64_t n = 0; n < ldy; ++n) {
            if (n >= N) { pad_ok &= y[r * ldy + n] == -7.0f; continue; }
            double s = b[n];
            for (int64_t k = 0; k < K; ++k) s += double(x[r * K + k]) * w[n * K + k];
            ref.push_back(s);
            yc.push_back(y[r * ldy + n]);
        }
    const double e = rel_absmax(yc, ref);
    check(e < 1e-6 && pad_ok, "linear (bias, strided output)", e, 1e-6);

    cpu::Linear L0{w.data(), nullptr, N, K};
    std::vector<float> y0(M * N);
    cpu::linear(L0, x.data(), K, M, y0.data(), N);
    ref.clear();
    for (int64_t r = 0; r < M; ++r)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) s += double(x[r * K + k]) * w[n * K + k];
            ref.push_back(s);
        }
    const double e0 = rel_absmax(y0, ref);
    check(e0 < 1e-6, "linear (no bias)", e0, 1e-6);
}

// Several row and column tiles, a fused epilogue, and bitwise thread-count invariance.
void test_linear_tiles(std::mt19937& rng) {
    const int64_t M = 300, K = 70, N = 1100;
    auto x = randn(M * K, rng), w = randn(N * K, rng), b = randn(N, rng);
    cpu::Linear L{w.data(), b.data(), N, K};
    auto run = [&](int threads) {
        std::vector<float> y(M * N);
        const int nt = omp_get_max_threads();
        omp_set_num_threads(threads);
        cpu::linear(L, x.data(), K, M, y.data(), N, [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) y[r * N + c] = 2.0f * y[r * N + c];
        });
        omp_set_num_threads(nt);
        return y;
    };
    const auto y1 = run(1), y7 = run(7);
    std::vector<double> ref(M * N);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t n = 0; n < N; ++n) {
            double s = b[n];
            for (int64_t k = 0; k < K; ++k) s += double(x[r * K + k]) * w[n * K + k];
            ref[r * N + n] = 2.0 * s;
        }
    const double e = rel_absmax(y7, ref);
    check(e < 1e-6, "linear (tiles + epilogue)", e, 1e-6);
    const bool inv = std::memcmp(y1.data(), y7.data(), sizeof(float) * M * N) == 0;
    check(inv, "linear: 1 thread == 7 threads (bitwise)", inv ? 0 : 1, 0);
}

void test_layer_norm(std::mt19937& rng) {
    const int64_t M = 97, D = 1024;
    auto x = randn(M * D, rng, 3.0f), w = randn(D, rng), b = randn(D, rng);
    for (int64_t i = 0; i < M * D; ++i) x[i] += 5.0f;  // non-zero mean
    for (bool bias : {true, false}) {
        cpu::LayerNorm N{w.data(), bias ? b.data() : nullptr, D, 1e-5f};
        std::vector<float> y(M * D);
        cpu::layer_norm(N, x.data(), D, M, y.data(), D);
        std::vector<double> ref(M * D);
        for (int64_t r = 0; r < M; ++r) {
            double mean = 0, var = 0;
            for (int64_t c = 0; c < D; ++c) mean += x[r * D + c];
            mean /= D;
            for (int64_t c = 0; c < D; ++c) var += (x[r * D + c] - mean) * (x[r * D + c] - mean);
            var /= D;
            for (int64_t c = 0; c < D; ++c)
                ref[r * D + c] = (x[r * D + c] - mean) / std::sqrt(var + 1e-5) * w[c] + (bias ? b[c] : 0.0);
        }
        const double e = rel_absmax(y, ref);
        check(e < 1e-6, bias ? "layer_norm (affine)" : "layer_norm (bias=False)", e, 1e-6);
    }
}

void test_gelu() {
    std::vector<float> x;
    for (float v = -10.0f; v <= 10.0f; v += 0.001f) x.push_back(v);
    for (bool tanh_approx : {false, true}) {
        std::vector<float> y = x;
        cpu::gelu_block(y.data(), 1, static_cast<int64_t>(y.size()), static_cast<int64_t>(y.size()), tanh_approx);
        double worst = 0;  // |y - ref| / max(|ref|, 1): 1 + erf(x) cancels for x < 0 in fp32, as in torch
        for (size_t i = 0; i < x.size(); ++i) {
            const double v = x[i];
            const double r = tanh_approx
                                 ? 0.5 * v * (1 + std::tanh(std::sqrt(2 / M_PI) * (v + 0.044715 * v * v * v)))
                                 : 0.5 * v * (1 + std::erf(v / std::sqrt(2.0)));
            worst = std::max(worst, std::fabs(y[i] - r) / std::max(std::fabs(r), 1.0));
        }
        check(worst < 1e-6, tanh_approx ? "gelu (tanh)" : "gelu (exact erf)", worst, 1e-6);
    }
}

// Naive double-precision attention over the stored edge list (no CSR).
std::vector<double> naive_attention(const std::vector<int64_t>& src, const std::vector<int64_t>& dst,
                                    const std::vector<float>& attr, int64_t A, int64_t n_dst,
                                    const std::vector<float>& q, const std::vector<float>& k,
                                    const std::vector<float>& v, const std::vector<float>& we,
                                    const std::vector<float>& be, int heads, int hd) {
    const int64_t ct = int64_t(heads) * hd, E = src.size();
    std::vector<double> e(E * ct), s(E * heads), out(n_dst * ct, 0.0);
    for (int64_t i = 0; i < E; ++i)
        for (int64_t c = 0; c < ct; ++c) {
            double acc = be[c];
            for (int64_t a = 0; a < A; ++a) acc += double(attr[i * A + a]) * we[c * A + a];
            e[i * ct + c] = acc;
        }
    for (int64_t i = 0; i < E; ++i)
        for (int h = 0; h < heads; ++h) {
            double acc = 0;
            for (int c = 0; c < hd; ++c) {
                const int64_t ch = int64_t(h) * hd + c;
                acc += double(q[dst[i] * ct + ch]) * (k[src[i] * ct + ch] + e[i * ct + ch]);
            }
            s[i * heads + h] = acc / std::sqrt(double(hd));
        }
    for (int64_t d = 0; d < n_dst; ++d)
        for (int h = 0; h < heads; ++h) {
            double m = -INFINITY, den = 0;
            for (int64_t i = 0; i < E; ++i)
                if (dst[i] == d) m = std::max(m, s[i * heads + h]);
            for (int64_t i = 0; i < E; ++i)
                if (dst[i] == d) den += std::exp(s[i * heads + h] - m);
            for (int64_t i = 0; i < E; ++i)
                if (dst[i] == d) {
                    const double a = std::exp(s[i * heads + h] - m) / den;
                    for (int c = 0; c < hd; ++c) {
                        const int64_t ch = int64_t(h) * hd + c;
                        out[d * ct + ch] += a * (v[src[i] * ct + ch] + e[i * ct + ch]);
                    }
                }
        }
    return out;
}

void test_attention(std::mt19937& rng) {
    const int heads = 4, hd = 16;  // C_tot = 64
    const int64_t ct = heads * hd, n_src = 50, n_dst = 40, A = 3;
    // Random in-degrees 0..12, edges stored in shuffled order; destinations 0, 7 and 39 have none.
    std::uniform_int_distribution<int> deg_d(0, 12), src_d(0, n_src - 1);
    std::vector<int64_t> src, dst;
    for (int64_t d = 0; d < n_dst; ++d) {
        const int deg = (d == 0 || d == 7 || d == n_dst - 1) ? 0 : deg_d(rng);
        for (int j = 0; j < deg; ++j) src.push_back(src_d(rng)), dst.push_back(d);
    }
    const int64_t E = src.size();
    std::vector<int64_t> perm(E);
    for (int64_t i = 0; i < E; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int64_t> ei(2 * E), s2(E), d2(E);
    for (int64_t i = 0; i < E; ++i) s2[i] = src[perm[i]], d2[i] = dst[perm[i]];
    for (int64_t i = 0; i < E; ++i) ei[i] = s2[i], ei[E + i] = d2[i];
    auto attr = randn(E * A, rng);
    auto q = randn(n_dst * ct, rng, 2.0f), k = randn(n_src * ct, rng, 2.0f), v = randn(n_src * ct, rng);
    auto we = randn(ct * A, rng), be = randn(ct, rng);

    TensorView tei, tea;
    tei.name = "edge_index"; tei.dtype = DType::I64; tei.shape = {2, E}; tei.data = ei.data(); tei.nbytes = ei.size() * 8;
    tea.name = "edge_attr"; tea.dtype = DType::F32; tea.shape = {E, A}; tea.data = attr.data(); tea.nbytes = attr.size() * 4;
    CsrGraph g = build_csr_by_dst(tei, tea, n_src, n_dst);

    std::vector<float> wt(A * ct);
    for (int64_t c = 0; c < ct; ++c)
        for (int64_t a = 0; a < A; ++a) wt[a * ct + c] = we[c * A + a];
    cpu::EdgeLinear el{wt.data(), be.data(), A, ct};

    // k and v interleaved as [k | v] rows (the backend's layout), q with a padded stride.
    std::vector<float> kv(n_src * 2 * ct), qp(n_dst * (ct + 8), 0.0f);
    for (int64_t j = 0; j < n_src; ++j)
        for (int64_t c = 0; c < ct; ++c) kv[j * 2 * ct + c] = k[j * ct + c], kv[j * 2 * ct + ct + c] = v[j * ct + c];
    for (int64_t i = 0; i < n_dst; ++i)
        for (int64_t c = 0; c < ct; ++c) qp[i * (ct + 8) + c] = q[i * ct + c];

    const auto ref = naive_attention(s2, d2, attr, A, n_dst, q, k, v, we, be, heads, hd);
    std::vector<float> out(n_dst * ct, -1.0f);
    cpu::graph_attention(g, 0, n_dst, qp.data(), ct + 8, kv.data(), kv.data() + ct, 2 * ct, el, heads, hd,
                         out.data(), ct);
    const double e = rel_absmax(out, ref);
    check(e < 2e-6, "attention vs naive (incl. zero in-degree)", e, 2e-6);
    bool zeros = true;
    for (int64_t d : {int64_t(0), int64_t(7), n_dst - 1})
        for (int64_t c = 0; c < ct; ++c) zeros &= out[d * ct + c] == 0.0f;
    check(zeros, "attention: zero in-degree rows are exactly 0", zeros ? 0 : 1, 0);

    // A destination sub-range [d0, d0 + nd) with local q rows gives the same bits.
    const int64_t d0 = 5, nd = 20;
    std::vector<float> sub(nd * ct);
    cpu::graph_attention(g, d0, nd, qp.data() + d0 * (ct + 8), ct + 8, kv.data(), kv.data() + ct, 2 * ct, el,
                         heads, hd, sub.data(), ct);
    const bool same = std::memcmp(sub.data(), out.data() + d0 * ct, sizeof(float) * nd * ct) == 0;
    check(same, "attention: destination chunk is bitwise equal", same ? 0 : 1, 0);

    // Thread-count invariance.
    std::vector<float> one(n_dst * ct);
    const int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    cpu::graph_attention(g, 0, n_dst, qp.data(), ct + 8, kv.data(), kv.data() + ct, 2 * ct, el, heads, hd,
                         one.data(), ct);
    omp_set_num_threads(nt);
    const bool inv = std::memcmp(one.data(), out.data(), sizeof(float) * n_dst * ct) == 0;
    check(inv, "attention: 1 thread == N threads (bitwise)", inv ? 0 : 1, 0);
}

}  // namespace

int main() {
    cpu::set_blas_single_threaded();
    std::mt19937 rng(1234);
    test_linear(rng);
    test_linear_tiles(rng);
    test_layer_norm(rng);
    test_gelu();
    test_attention(rng);
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
