// cuBLASLt wrapper for the Linear layers: y[M, N] = x[M, K] W[N, K]^T (+ bias[N]), row-major with
// leading dimensions. fp32: SGEMM semantics (CUBLAS_COMPUTE_32F, no TF32). fp16: fp16 inputs,
// weights, bias and output, fp32 accumulation and epilogue (what autocast's Linear does).
// One plan (descriptors + algorithm) per distinct shape, chosen once and reused, so every call of a
// shape runs the same kernel: results are bitwise repeatable, in and outside CUDA graphs.
// Algorithm choice: with `autotune`, the first call of a shape times up to 16 cuBLASLt heuristic
// candidates on its own operands and keeps the fastest (timing-based: two processes may pick
// different kernels, i.e. different roundings); otherwise the top heuristic candidate
// (deterministic across processes).
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <tuple>

#include "varda/cuda/common.h"

namespace varda::cuda {

class Gemm {
public:
    Gemm(Elt type, size_t workspace_bytes, bool autotune);
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    // Plans are created on first use of a shape (host work only, no device allocation), so the
    // first forward pass, run eagerly, prepares everything a later graph capture needs.
    void linear(const void* x, int64_t ldx, const void* W, int64_t ldw, const void* bias, void* y, int64_t ldy,
                int64_t M, int64_t N, int64_t K, cudaStream_t s);

    Elt type() const { return type_; }
    size_t num_plans() const { return plans_.size(); }

private:
    struct Plan;
    using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int>;
    Plan& plan(const Key& k);

    Elt type_;
    bool autotune_;
    struct Operands { const void *x = nullptr, *W = nullptr, *bias = nullptr; void* y = nullptr; cudaStream_t s = nullptr; };
    Operands tune_;  // operands of the call creating a plan (autotuning)
    cublasLtHandle_t lt_ = nullptr;
    DeviceBuffer workspace_;
    std::map<Key, std::unique_ptr<Plan>> plans_;
};

}  // namespace varda::cuda
