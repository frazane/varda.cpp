// Unit tests of the CUDA kernels (src/varda/cuda/kernels.h) against simple host references:
// graph attention (zero in-degree destinations, degrees up to 64, odd sizes, destination
// sub-ranges, several channel/head layouts incl. C = 768, invariance to the warp split), LayerNorm
// and the fused residual + LayerNorm (every C the backend accepts), GELU (erf and tanh),
// normaliser, input assembly and output assembly (residual and every bounding type). fp32 and
// fp16 storage.
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "varda/cuda/common.h"
#include "varda/cuda/kernels.h"
#include "varda/format.h"
#include "varda/graph.h"

using namespace varda;
using namespace varda::cuda;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, double err, double tol) {
    std::printf("%-64s err %.3e tol %.1e %s\n", what.c_str(), err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

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

template <class T> float rnd(float x) {
    if (std::is_same<T, float>::value) return x;
    return __half2float(__float2half_rn(x));
}
template <class T> std::vector<float> rounded(std::vector<float> v) {
    for (auto& x : v) x = rnd<T>(x);
    return v;
}

// Device copy of host fp32 values stored as T.
template <class T> DeviceBuffer to_dev(const std::vector<float>& v) {
    std::vector<T> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = static_cast<T>(v[i]);
    DeviceBuffer b(h.size() * sizeof(T));
    VARDA_CUDA_CHECK(cudaMemcpy(b.get(), h.data(), b.bytes(), cudaMemcpyHostToDevice));
    return b;
}
template <class T> std::vector<float> from_dev(const DeviceBuffer& b, size_t n) {
    std::vector<T> h(n);
    VARDA_CUDA_CHECK(cudaDeviceSynchronize());
    VARDA_CUDA_CHECK(cudaMemcpy(h.data(), b.get(), n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(h[i]);
    return v;
}
template <class U> DeviceBuffer raw_dev(const std::vector<U>& v) {
    DeviceBuffer b(std::max<size_t>(v.size(), 1) * sizeof(U));
    if (!v.empty()) VARDA_CUDA_CHECK(cudaMemcpy(b.get(), v.data(), v.size() * sizeof(U), cudaMemcpyHostToDevice));
    return b;
}

const char* tname(bool f32) { return f32 ? "fp32" : "fp16"; }

// ------------------------------------------------------------------ attention

struct TestGraph {
    int64_t n_src, n_dst;
    std::vector<int64_t> ei;  // [2, E]
    std::vector<float> ea;    // [E, A]
    int A;
    CsrGraph csr;
};

TestGraph make_graph(std::mt19937& rng, int64_t n_src, int64_t n_dst, int A) {
    TestGraph g{n_src, n_dst, {}, {}, A, {}};
    const int degs[] = {0, 1, 2, 3, 5, 8, 17, 33, 64};
    std::uniform_int_distribution<int> pick(0, 8), src(0, static_cast<int>(n_src) - 1);
    std::vector<int64_t> s, d;
    for (int64_t i = 0; i < n_dst; ++i) {
        const int deg = i < 9 ? degs[i] : degs[pick(rng)];  // every degree class at least once
        for (int k = 0; k < deg; ++k) {
            s.push_back(src(rng));
            d.push_back(i);
        }
    }
    // shuffle the stored edge order (the CSR builder sorts stably by destination)
    std::vector<size_t> perm(s.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    const int64_t E = static_cast<int64_t>(s.size());
    g.ei.resize(2 * E);
    for (int64_t e = 0; e < E; ++e) {
        g.ei[e] = s[perm[e]];
        g.ei[E + e] = d[perm[e]];
    }
    g.ea = randn(E * A, rng);
    TensorView vi, va;
    vi.name = "ei"; vi.dtype = DType::I64; vi.shape = {2, E}; vi.data = g.ei.data(); vi.nbytes = g.ei.size() * 8;
    va.name = "ea"; va.dtype = DType::F32; va.shape = {E, A}; va.data = g.ea.data(); va.nbytes = g.ea.size() * 4;
    g.csr = build_csr_by_dst(vi, va, n_src, n_dst);
    return g;
}

template <class T>
void test_attention(std::mt19937& rng, int C, int heads, int A) {
    const bool f32 = std::is_same<T, float>::value;
    const int hd = C / heads;
    const int64_t n_src = 131, n_dst = 97, ld = C + 8;
    TestGraph g = make_graph(rng, n_src, n_dst, A);
    const auto& csr = g.csr;
    // operands as stored on the device (fp16: rounded), the edge Linear rounded as autocast does
    auto q = rounded<T>(randn(n_dst * ld, rng)), k = rounded<T>(randn(n_src * ld, rng)),
         v = rounded<T>(randn(n_src * ld, rng)), xr = rounded<T>(randn(n_dst * ld, rng));
    auto we = rounded<T>(randn(size_t(C) * A, rng, 0.5f)), be = rounded<T>(randn(C, rng, 0.5f));
    std::vector<float> attr(csr.attr.size());
    for (size_t i = 0; i < attr.size(); ++i) attr[i] = rnd<T>(csr.attr[i]);

    std::vector<float4> packed(csr.n_edges);
    for (int64_t e = 0; e < csr.n_edges; ++e) {
        float a[3] = {0, 0, 0}, sb;
        for (int f = 0; f < A; ++f) a[f] = attr[e * A + f];
        std::memcpy(&sb, &csr.src[e], 4);
        packed[e] = make_float4(a[0], a[1], a[2], sb);
    }
    std::vector<float> w4(size_t(4) * C, 0.0f);
    for (int c = 0; c < C; ++c) {
        for (int f = 0; f < A; ++f) w4[size_t(f) * C + c] = we[c * A + f];
        w4[size_t(3) * C + c] = be[c];
    }
    DeviceBuffer d_rp = raw_dev(csr.row_ptr), d_ed = raw_dev(packed), d_w = raw_dev(w4);
    DeviceBuffer d_q = to_dev<T>(q), d_k = to_dev<T>(k), d_v = to_dev<T>(v), d_xr = to_dev<T>(xr);
    DeviceBuffer d_o(n_dst * C * sizeof(T));

    // reference: e = W_e a + b_e materialised and added to both k and v, two-pass softmax, double
    std::vector<double> ref(n_dst * C);
    for (int64_t d = 0; d < n_dst; ++d) {
        const int b0 = csr.row_ptr[d], b1 = csr.row_ptr[d + 1];
        for (int h = 0; h < heads; ++h) {
            std::vector<double> s(b1 - b0);
            std::vector<std::vector<double>> e(b1 - b0, std::vector<double>(hd));
            double mx = -INFINITY;
            for (int j = b0; j < b1; ++j) {
                const int sj = csr.src[j];
                double dot = 0;
                for (int c = 0; c < hd; ++c) {
                    const int ch = h * hd + c;
                    double ev = be[ch];
                    for (int f = 0; f < A; ++f) ev += double(we[ch * A + f]) * attr[j * A + f];
                    e[j - b0][c] = ev;
                    dot += double(q[d * ld + ch]) * (k[sj * ld + ch] + ev);
                }
                s[j - b0] = dot / std::sqrt(double(hd));
                mx = std::max(mx, s[j - b0]);
            }
            double sum = 0;
            for (auto& x : s) sum += (x = std::exp(x - mx));
            for (int c = 0; c < hd; ++c) {
                const int ch = h * hd + c;
                double o = 0;
                for (int j = b0; j < b1; ++j) o += s[j - b0] / sum * (v[csr.src[j] * ld + ch] + e[j - b0][c]);
                ref[d * C + ch] = o + xr[d * ld + ch];
            }
        }
    }

    AttentionArgs a;
    a.g.n_src = n_src; a.g.n_dst = n_dst; a.g.n_edges = csr.n_edges; a.g.attr_dim = A;
    a.g.row_ptr = d_rp.as<int32_t>(); a.g.edges = d_ed.as<float4>();
    a.q = d_q.get(); a.ldq = ld; a.k = d_k.get(); a.ldk = ld; a.v = d_v.get(); a.ldv = ld;
    a.xr = d_xr.get(); a.ldxr = ld; a.out = d_o.get(); a.ldo = C;
    a.w_e = d_w.as<float>(); a.C = C; a.heads = heads; a.head_dim = hd;
    a.dst_lo = 0; a.n_rows = n_dst;

    std::vector<float> first;
    for (int vpl : {1, 2, 4}) {
        if (C % (256 * vpl)) continue;
        VARDA_CUDA_CHECK(cudaMemset(d_o.get(), 0xff, d_o.bytes()));
        attention<T>(a, nullptr, vpl);
        auto out = from_dev<T>(d_o, n_dst * C);
        const std::string name = std::string("attention ") + tname(f32) + " C=" + std::to_string(C) + " hd=" +
                                 std::to_string(hd) + " A=" + std::to_string(A) + " vpl=" + std::to_string(vpl);
        if (first.empty()) {
            first = out;
            const double err = rel_absmax(out, ref), tol = f32 ? 1e-5 : 2e-3;
            check(err <= tol, name, err, tol);
            // zero in-degree destinations: attention output 0, so out = xr exactly
            bool zero_ok = true;
            for (int64_t d = 0; d < n_dst; ++d)
                if (csr.row_ptr[d] == csr.row_ptr[d + 1])
                    for (int c = 0; c < C; ++c) zero_ok &= out[d * C + c] == xr[d * ld + c];
            check(zero_ok, name + " zero in-degree -> xr", zero_ok ? 0 : 1, 0);
        } else {
            const bool same = std::memcmp(out.data(), first.data(), out.size() * 4) == 0;
            check(same, name + " bitwise == vpl 1", same ? 0 : 1, 0);
        }
    }
    // destination sub-range [lo, lo + n): rows are local; bitwise equal to the full launch
    const int64_t lo = 13, n = 51;
    DeviceBuffer d_o2(n * C * sizeof(T));
    AttentionArgs b = a;
    b.dst_lo = lo; b.n_rows = n;
    b.q = static_cast<const T*>(d_q.get()) + lo * ld;
    b.xr = static_cast<const T*>(d_xr.get()) + lo * ld;
    b.out = d_o2.get();
    attention<T>(b, nullptr, 1);
    auto part = from_dev<T>(d_o2, n * C);
    const bool same = std::memcmp(part.data(), first.data() + lo * C, part.size() * 4) == 0;
    check(same, std::string("attention ") + tname(f32) + " C=" + std::to_string(C) + " sub-range bitwise", same ? 0 : 1, 0);
}

// ------------------------------------------------------------------ LayerNorm

template <class T>
void test_layernorm(std::mt19937& rng, int C) {
    const bool f32 = std::is_same<T, float>::value;
    const int64_t rows = 37;
    const float eps = 1e-5f;
    auto y = rounded<T>(randn(rows * C, rng, 3.0f)), r1 = rounded<T>(randn(rows * C, rng)),
         r2 = rounded<T>(randn(rows * C, rng));
    for (int c = 0; c < C; ++c) y[c] = rnd<T>(y[c] + 50.0f);  // a row with a large mean
    auto w = randn(C, rng), b = randn(C, rng);
    DeviceBuffer dy = to_dev<T>(y), d1 = to_dev<T>(r1), d2 = to_dev<T>(r2), dw = raw_dev(w), db = raw_dev(b);
    DeviceBuffer o1(rows * C * sizeof(T)), o2(rows * C * sizeof(T)), ln(rows * C * sizeof(T));
    auto lnref = [&](const std::vector<double>& x) {
        std::vector<double> out(x.size());
        for (int64_t r = 0; r < rows; ++r) {
            double m = 0, v = 0;
            for (int c = 0; c < C; ++c) m += x[r * C + c];
            m /= C;
            for (int c = 0; c < C; ++c) v += (x[r * C + c] - m) * (x[r * C + c] - m);
            v /= C;
            for (int c = 0; c < C; ++c) out[r * C + c] = (x[r * C + c] - m) / std::sqrt(v + eps) * w[c] + b[c];
        }
        return out;
    };
    const double tol = f32 ? 2e-6 : 1e-3;
    const std::string tag = std::string(tname(f32)) + " C=" + std::to_string(C);
    {
        std::vector<double> x(y.begin(), y.end());
        layernorm<T>(dy.as<T>(), C, dw.as<float>(), db.as<float>(), ln.as<T>(), C, rows, C, eps, nullptr);
        const double err = rel_absmax(from_dev<T>(ln, rows * C), lnref(x));
        check(err <= tol, "layernorm " + tag, err, tol);
    }
    {
        // out1 = T(y + r1), out2 = T(out1 + r2), ln = LN(out2): host mirror of the roundings
        std::vector<float> e1(rows * C), e2(rows * C);
        for (size_t i = 0; i < e1.size(); ++i) {
            e1[i] = rnd<T>(y[i] + r1[i]);
            e2[i] = rnd<T>(e1[i] + r2[i]);
        }
        add_layernorm<T>(dy.as<T>(), d1.as<T>(), o1.as<T>(), d2.as<T>(), o2.as<T>(), dw.as<float>(), db.as<float>(),
                         ln.as<T>(), C, rows, C, eps, nullptr);
        auto g1 = from_dev<T>(o1, rows * C), g2 = from_dev<T>(o2, rows * C);
        const bool b1 = g1 == e1, b2 = g2 == e2;
        check(b1 && b2, "add_layernorm " + tag + " residual sums bitwise", (b1 && b2) ? 0 : 1, 0);
        std::vector<double> x(e2.begin(), e2.end());
        const double err = rel_absmax(from_dev<T>(ln, rows * C), lnref(x));
        check(err <= tol, "add_layernorm " + tag + " LN", err, tol);
        // single residual, in place (out1 aliases y)
        DeviceBuffer dy2 = to_dev<T>(y);
        add_layernorm<T>(dy2.as<T>(), d1.as<T>(), dy2.as<T>(), nullptr, nullptr, dw.as<float>(), db.as<float>(),
                         ln.as<T>(), C, rows, C, eps, nullptr);
        const bool b3 = from_dev<T>(dy2, rows * C) == e1;
        check(b3, "add_layernorm " + tag + " in place, one residual", b3 ? 0 : 1, 0);
    }
}

// ------------------------------------------------------------------ GELU

template <class T>
void test_gelu(std::mt19937& rng, bool tanh_approx) {
    const bool f32 = std::is_same<T, float>::value;
    const int64_t n = 8 * 1001;
    std::uniform_real_distribution<float> u(-6.0f, 6.0f);
    std::vector<float> x(n);
    for (auto& v : x) v = rnd<T>(u(rng));
    DeviceBuffer d = to_dev<T>(x);
    gelu_inplace<T>(d.as<T>(), n, tanh_approx, nullptr);
    auto out = from_dev<T>(d, n);
    std::vector<double> ref(n);
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i];
        ref[i] = tanh_approx ? 0.5 * v * (1 + std::tanh(std::sqrt(2 / 3.14159265358979323846) * (v + 0.044715 * v * v * v)))
                             : 0.5 * v * (1 + std::erf(v / std::sqrt(2.0)));
    }
    const double err = rel_absmax(out, ref), tol = f32 ? 1e-6 : 1e-3;
    check(err <= tol, std::string("gelu ") + tname(f32) + (tanh_approx ? " tanh" : " erf"), err, tol);
}

// ------------------------------------------------------------------ normaliser, assembly, output

void test_normaliser(std::mt19937& rng) {
    const int V = 13;
    const int64_t n = 2 * 101 * V;
    auto x = randn(n, rng, 100.0f), mul = randn(V, rng), add = randn(V, rng);
    DeviceBuffer dx = raw_dev(x), dm = raw_dev(mul), da = raw_dev(add), dn(n * 4), dy(n * 4);
    normalise(dx.as<float>(), dm.as<float>(), da.as<float>(), dn.as<float>(), n, V, nullptr);
    denormalise(dn.as<float>(), da.as<float>(), dm.as<float>(), dy.as<float>(), n, V, nullptr);
    auto xn = from_dev<float>(dn, n), y = from_dev<float>(dy, n);
    bool ok_n = true, ok_d = true;
    for (int64_t i = 0; i < n; ++i) {
        volatile float t = x[i] * mul[i % V];  // two separately rounded ops
        const float e = t + add[i % V];
        volatile float u = xn[i] - add[i % V];
        const float f = u / mul[i % V];
        ok_n &= xn[i] == e;
        ok_d &= y[i] == f;
    }
    check(ok_n, "normalise bitwise (x*mul, rounded, +add)", ok_n ? 0 : 1, 0);
    check(ok_d, "denormalise bitwise ((y-add), rounded, /mul)", ok_d ? 0 : 1, 0);
}

template <class T>
void test_assemble(std::mt19937& rng) {
    const bool f32 = std::is_same<T, float>::value;
    const int Tin = 2, V = 11, A = 4, F = Tin * V + A, ld = 32;
    const int64_t G = 77;
    auto xn = randn(Tin * G * V, rng), attr = randn(G * A, rng);
    DeviceBuffer dx = raw_dev(xn), da = raw_dev(attr), d_out(G * ld * sizeof(T));
    VARDA_CUDA_CHECK(cudaMemset(d_out.get(), 0xff, d_out.bytes()));
    assemble_x_data<T>(dx.as<float>(), da.as<float>(), Tin, G, V, A, d_out.as<T>(), ld, nullptr);
    auto out = from_dev<T>(d_out, G * ld);
    bool ok = true;
    for (int64_t g = 0; g < G; ++g)
        for (int f = 0; f < ld; ++f) {
            float e = 0;
            if (f < Tin * V) e = xn[(f / V) * G * V + g * V + f % V];
            else if (f < F) e = attr[g * A + f - Tin * V];
            ok &= out[g * ld + f] == rnd<T>(e);
        }
    check(ok, std::string("assemble_x_data bitwise ") + tname(f32), ok ? 0 : 1, 0);
}

template <class T>
void test_output(std::mt19937& rng) {
    const bool f32 = std::is_same<T, float>::value;
    const int V_in = 10, V_out = 9, ld = 16;
    const int64_t G = 45;
    auto dec = rounded<T>(randn(G * ld, rng)), xs = randn(G * V_in, rng, 0.5f);
    std::vector<int> res = {3, 0, -1, 1, 2, 9, 4, 5, -1};  // output v gets xs[res[v]]
    std::vector<BoundingDev> bd;
    std::vector<int> idx;
    auto add_b = [&](int type, std::vector<int> i, std::vector<int> t, float lo, float hi) {
        BoundingDev b;
        b.type = type; b.n = static_cast<int>(i.size()); b.idx_off = static_cast<int>(idx.size());
        idx.insert(idx.end(), i.begin(), i.end());
        b.tot_off = static_cast<int>(idx.size()); b.n_tot = static_cast<int>(t.size());
        idx.insert(idx.end(), t.begin(), t.end());
        b.lo = lo; b.hi = hi;
        bd.push_back(b);
    };
    add_b(0, {2}, {}, 0, 0);               // relu
    add_b(1, {3}, {}, 0.0f, 1.0f);         // hardtanh
    add_b(2, {4, 5}, {3}, 0.0f, 1.0f);     // fraction, one total
    add_b(2, {6, 7}, {0, 1}, -0.5f, 0.5f); // fraction, paired totals
    DeviceBuffer d_dec = to_dev<T>(dec), d_xs = raw_dev(xs), d_res = raw_dev(res), d_bd = raw_dev(bd),
                 d_idx = raw_dev(idx), d_pre(G * V_out * 4), d_y(G * V_out * 4);
    OutputArgs a;
    a.dec = d_dec.get(); a.ld_dec = ld; a.xn_step = d_xs.as<float>(); a.V_in = V_in; a.V_out = V_out; a.G = G;
    a.res_src = d_res.as<int>(); a.bounds = d_bd.as<BoundingDev>(); a.n_bounds = static_cast<int>(bd.size());
    a.bidx = d_idx.as<int>(); a.pre_bound = d_pre.as<float>(); a.y = d_y.as<float>();
    output_assemble<T>(a, nullptr);
    auto pre = from_dev<float>(d_pre, G * V_out), y = from_dev<float>(d_y, G * V_out);
    bool ok_pre = true, ok_y = true;
    for (int64_t g = 0; g < G; ++g) {
        std::vector<float> r(V_out);
        for (int v = 0; v < V_out; ++v) r[v] = res[v] >= 0 ? dec[g * ld + v] + xs[g * V_in + res[v]] : dec[g * ld + v];
        for (int v = 0; v < V_out; ++v) ok_pre &= pre[g * V_out + v] == r[v];
        auto clamp = [](float x, float lo, float hi) { return std::min(std::max(x, lo), hi); };
        r[2] = std::max(r[2], 0.0f);
        r[3] = clamp(r[3], 0, 1);
        r[4] = clamp(r[4], 0, 1); r[5] = clamp(r[5], 0, 1);
        r[4] *= r[3]; r[5] *= r[3];
        r[6] = clamp(r[6], -0.5f, 0.5f); r[7] = clamp(r[7], -0.5f, 0.5f);
        const float t0 = r[0], t1 = r[1];
        r[6] *= t0; r[7] *= t1;
        for (int v = 0; v < V_out; ++v) ok_y &= y[g * V_out + v] == r[v];
    }
    check(ok_pre, std::string("output_assemble residual bitwise ") + tname(f32), ok_pre ? 0 : 1, 0);
    check(ok_y, std::string("output_assemble bounding bitwise ") + tname(f32), ok_y ? 0 : 1, 0);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 0;
    }
    std::mt19937 rng(1234);
    const int layouts[][3] = {{1024, 16, 3}, {256, 8, 3}, {512, 4, 2}, {256, 32, 1}, {768, 12, 3}};  // C, heads, attr dim
    for (const auto& l : layouts) {
        test_attention<float>(rng, l[0], l[1], l[2]);
        test_attention<__half>(rng, l[0], l[1], l[2]);
    }
    for (int C : {256, 768, 1024, 1280, 1536, 1792, 2048}) {
        test_layernorm<float>(rng, C);
        test_layernorm<__half>(rng, C);
    }
    for (bool t : {false, true}) {
        test_gelu<float>(rng, t);
        test_gelu<__half>(rng, t);
    }
    test_normaliser(rng);
    test_assemble<float>(rng);
    test_assemble<__half>(rng);
    test_output<float>(rng);
    test_output<__half>(rng);
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
