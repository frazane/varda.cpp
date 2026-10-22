// Unit tests of the cuBLASLt Linear wrapper (src/varda/cuda/gemm.h) against a double-precision
// host reference: fp32 and fp16, with and without bias, odd row counts, padded and unpadded
// output widths (N = 67 in a row of 72), tiny K (8) and wide K (4096), plus bitwise
// repeatability of a plan.
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "varda/cuda/common.h"
#include "varda/cuda/gemm.h"

using namespace varda::cuda;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, double err, double tol) {
    std::printf("%-64s err %.3e tol %.1e %s\n", what.c_str(), err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

template <class T> float rnd(float x) { return std::is_same<T, float>::value ? x : __half2float(__float2half_rn(x)); }

template <class T>
void test_linear(std::mt19937& rng, Gemm& gemm, int64_t M, int64_t N, int64_t K, int64_t ldy, bool bias) {
    const bool f32 = std::is_same<T, float>::value;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(M * K), w(N * K), b(N);
    for (auto& v : x) v = rnd<T>(nd(rng));
    for (auto& v : w) v = rnd<T>(nd(rng) / std::sqrt(float(K)));
    for (auto& v : b) v = rnd<T>(nd(rng));
    auto dev = [](const std::vector<float>& v) {
        std::vector<T> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = static_cast<T>(v[i]);
        DeviceBuffer d(h.size() * sizeof(T));
        VARDA_CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));
        return d;
    };
    DeviceBuffer dx = dev(x), dw = dev(w), db = dev(b), dy(M * ldy * sizeof(T)), dy2(M * ldy * sizeof(T));
    VARDA_CUDA_CHECK(cudaMemset(dy.get(), 0, dy.bytes()));
    VARDA_CUDA_CHECK(cudaMemset(dy2.get(), 0, dy2.bytes()));
    gemm.linear(dx.get(), K, dw.get(), K, bias ? db.get() : nullptr, dy.get(), ldy, M, N, K, nullptr);
    gemm.linear(dx.get(), K, dw.get(), K, bias ? db.get() : nullptr, dy2.get(), ldy, M, N, K, nullptr);
    VARDA_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<T> hy(M * ldy), hy2(M * ldy);
    VARDA_CUDA_CHECK(cudaMemcpy(hy.data(), dy.get(), dy.bytes(), cudaMemcpyDeviceToHost));
    VARDA_CUDA_CHECK(cudaMemcpy(hy2.data(), dy2.get(), dy2.bytes(), cudaMemcpyDeviceToHost));
    double m = 0, am = 0;
    bool pad_ok = true;
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t n = 0; n < N; ++n) {
            double r = bias ? b[n] : 0.0;
            for (int64_t k = 0; k < K; ++k) r += double(x[i * K + k]) * w[n * K + k];
            m = std::max(m, std::fabs(float(hy[i * ldy + n]) - r));
            am = std::max(am, std::fabs(r));
        }
        for (int64_t n = N; n < ldy; ++n) pad_ok &= float(hy[i * ldy + n]) == 0.0f;  // untouched
    }
    const double err = m / am, tol = f32 ? 1e-5 : 2e-3;
    const std::string name = std::string("linear ") + (f32 ? "fp32" : "fp16") + " M=" + std::to_string(M) + " N=" +
                             std::to_string(N) + " K=" + std::to_string(K) + " ldy=" + std::to_string(ldy) +
                             (bias ? " bias" : " no bias");
    check(err <= tol, name, err, tol);
    check(pad_ok, name + " leaves columns >= N", pad_ok ? 0 : 1, 0);
    const bool same = std::memcmp(hy.data(), hy2.data(), hy.size() * sizeof(T)) == 0;
    check(same, name + " repeat bitwise", same ? 0 : 1, 0);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 0;
    }
    std::mt19937 rng(7);
    for (bool autotune : {false, true}) {
        Gemm g32(Elt::F32, size_t(32) << 20, autotune), g16(Elt::F16, size_t(32) << 20, autotune);
        test_linear<float>(rng, g32, 1001, 67, 168, 72, true);
        test_linear<__half>(rng, g16, 1001, 67, 168, 72, true);
        test_linear<float>(rng, g32, 513, 1024, 8, 1024, true);
        test_linear<__half>(rng, g16, 513, 1024, 8, 1024, true);
        test_linear<float>(rng, g32, 37, 512, 4096, 512, false);
        test_linear<__half>(rng, g16, 37, 512, 4096, 512, false);
        test_linear<__half>(rng, g16, 4099, 72, 1024, 72, true);
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
