// Device state update of the CUDA rollout (varda::cuda::rollout_update, src/varda/cuda/rollout_cuda.h)
// vs the host StateUpdate::apply (src/varda/rollout.h): bitwise equal on synthetic windows (T = 2
// and 3, G not a multiple of the block size, permuted prognostic indices, values distinct per
// slot), over two consecutive updates. Skipped (exit 0) without a CUDA device.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "varda/cuda/common.h"
#include "varda/cuda/rollout_cuda.h"
#include "varda/rollout.h"

using namespace varda;

static int failures = 0;

static void run_case(int T, int64_t G) {
    const int V_in = 11, V_out = 8;
    const std::vector<int> prog_in = {0, 2, 3, 8, 10}, prog_out = {7, 1, 0, 4, 5}, dyn_in = {1, 6, 9},
                           const_in = {4, 5, 7};
    StateUpdate u = StateUpdate::make(T, G, V_in, V_out, prog_in, prog_out, dyn_in, const_in);
    std::vector<float> w(size_t(T) * G * V_in), y(G * V_out), f(G * u.F);
    unsigned x = 12345;
    auto rnd = [&]() { x = x * 1664525u + 1013904223u; return float(x >> 8) / 65536.0f - 128.0f; };
    for (auto& v : w) v = rnd();
    for (auto& v : y) v = rnd();
    for (auto& v : f) v = rnd();

    float *dw, *dy, *df;
    int8_t* dk;
    int* ds;
    VARDA_CUDA_CHECK(cudaMalloc(&dw, w.size() * 4));
    VARDA_CUDA_CHECK(cudaMalloc(&dy, y.size() * 4));
    VARDA_CUDA_CHECK(cudaMalloc(&df, f.size() * 4));
    VARDA_CUDA_CHECK(cudaMalloc(&dk, V_in));
    VARDA_CUDA_CHECK(cudaMalloc(&ds, V_in * 4));
    VARDA_CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(df, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(dk, u.kind.data(), V_in, cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(ds, u.src.data(), V_in * 4, cudaMemcpyHostToDevice));
    for (int rep = 0; rep < 2; ++rep) {
        u.apply(w.data(), y.data(), f.data());
        cuda::rollout_update(dw, dy, df, dk, ds, T, G, V_in, V_out, u.F, nullptr);
        VARDA_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(w.size());
        VARDA_CUDA_CHECK(cudaMemcpy(got.data(), dw, w.size() * 4, cudaMemcpyDeviceToHost));
        const bool eq = std::memcmp(got.data(), w.data(), w.size() * 4) == 0;
        std::printf("T=%d G=%lld update %d: device %s host\n", T, (long long)G, rep + 1, eq ? "==" : "!=");
        if (!eq) ++failures;
    }
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(df);
    cudaFree(dk);
    cudaFree(ds);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("skip: no CUDA device\n");
        return 0;
    }
    run_case(2, 1);
    run_case(2, 40321);
    run_case(3, 777);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_rollout_cuda: all checks passed\n");
    return 0;
}
