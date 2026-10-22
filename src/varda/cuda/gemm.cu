#include "varda/cuda/gemm.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace varda::cuda {

namespace {

void lt_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLASLt error ") + std::to_string(static_cast<int>(st)) + " in " + what);
}
#define LT(expr) lt_check((expr), #expr)

int alignment_of(const void* p) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    int al = 256;
    while (al > 1 && (a % al)) al >>= 1;
    return al;
}

}  // namespace

struct Gemm::Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    int tuned_index = 0;   // position of the chosen algorithm in the heuristic list
    float tuned_ms = 0;    // its time per call when autotuned
    ~Plan() {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

Gemm::Gemm(Elt type, size_t workspace_bytes, bool autotune)
    : type_(type), autotune_(autotune), workspace_(workspace_bytes) {
    LT(cublasLtCreate(&lt_));
}

Gemm::~Gemm() {
    plans_.clear();
    if (lt_) cublasLtDestroy(lt_);
}

Gemm::Plan& Gemm::plan(const Key& k) {
    auto it = plans_.find(k);
    if (it != plans_.end()) return *it->second;
    const auto [M, N, K, ldx, ldw, ldy, bias, align] = k;
    auto p = std::make_unique<Plan>();
    const cudaDataType_t dt = type_ == Elt::F32 ? CUDA_R_32F : CUDA_R_16F;
    LT(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    if (bias) {
        const cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
        LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
        LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &dt, sizeof(dt)));
    }
    // Column-major view: D (N x M, ld ldy) = W^T (N x K) * x^T (K x M).
    LT(cublasLtMatrixLayoutCreate(&p->a, dt, K, N, ldw));
    LT(cublasLtMatrixLayoutCreate(&p->b, dt, K, M, ldx));
    LT(cublasLtMatrixLayoutCreate(&p->c, dt, N, M, ldy));

    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    const uint64_t ws = workspace_.bytes();
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    const uint32_t al = static_cast<uint32_t>(align);
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        LT(cublasLtMatmulPreferenceSetAttribute(pref, attr, &al, sizeof(al)));
    constexpr int kMaxAlgos = 16;
    cublasLtMatmulHeuristicResult_t res[kMaxAlgos];
    int n = 0;
    const cublasStatus_t st =
        cublasLtMatmulAlgoGetHeuristic(lt_, p->op, p->a, p->b, p->c, p->c, pref, autotune_ ? kMaxAlgos : 1, res, &n);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || n == 0)
        throw std::runtime_error("cuBLASLt: no algorithm for GEMM M=" + std::to_string(M) + " N=" + std::to_string(N) +
                                 " K=" + std::to_string(K));
    p->algo = res[0].algo;
    if (autotune_ && n > 1 && tune_.x) {
        // Time every candidate on the call's own operands (the real call follows and overwrites y).
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        VARDA_CUDA_CHECK(cudaStreamIsCapturing(tune_.s, &cap));
        if (cap == cudaStreamCaptureStatusNone) {
            if (tune_.bias)
                LT(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &tune_.bias,
                                                  sizeof(tune_.bias)));
            cudaEvent_t e0, e1;
            VARDA_CUDA_CHECK(cudaEventCreate(&e0));
            VARDA_CUDA_CHECK(cudaEventCreate(&e1));
            const float one = 1.0f, zero = 0.0f;
            float best = 1e30f;
            int best_i = 0;
            for (int i = 0; i < n; ++i) {
                if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > workspace_.bytes()) continue;
                auto run = [&] {
                    return cublasLtMatmul(lt_, p->op, &one, tune_.W, p->a, tune_.x, p->b, &zero, tune_.y, p->c,
                                          tune_.y, p->c, &res[i].algo, workspace_.get(), workspace_.bytes(), tune_.s);
                };
                if (run() != CUBLAS_STATUS_SUCCESS) continue;
                VARDA_CUDA_CHECK(cudaEventRecord(e0, tune_.s));
                const int reps = 3;
                for (int r = 0; r < reps; ++r) LT(run());
                VARDA_CUDA_CHECK(cudaEventRecord(e1, tune_.s));
                VARDA_CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                VARDA_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                // Ties (within 1%) keep the earlier, heuristically preferred candidate.
                if (ms < best * 0.99f) {
                    best = ms;
                    best_i = i;
                }
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            p->algo = res[best_i].algo;
            p->tuned_index = best_i;
            p->tuned_ms = best / 3;
        }
    }
    if (const char* e = std::getenv("VARDA_GEMM_LOG"); e && *e == '1')
        std::fprintf(stderr, "gemm plan M=%lld N=%lld K=%lld ld=%lld/%lld/%lld bias=%d align=%d: %d candidates, chose #%d (%.3f ms)\n",
                     (long long)M, (long long)N, (long long)K, (long long)ldx, (long long)ldw, (long long)ldy, (int)bias,
                     align, n, p->tuned_index, p->tuned_ms);
    auto& ref = *p;
    plans_.emplace(k, std::move(p));
    return ref;
}

void Gemm::linear(const void* x, int64_t ldx, const void* W, int64_t ldw, const void* bias, void* y, int64_t ldy,
                  int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    if (M <= 0) return;
    int al = alignment_of(x);
    for (const void* p : {W, static_cast<const void*>(y), bias})
        if (p) al = std::min(al, alignment_of(p));
    const size_t es = elt_size(type_);
    for (int64_t ld : {ldx, ldw, ldy}) al = std::min(al, alignment_of(reinterpret_cast<const void*>(ld * es)));
    tune_ = {x, W, bias, y, s};
    Plan& p = plan(Key{M, N, K, ldx, ldw, ldy, bias != nullptr, al});
    tune_ = {};
    if (bias) LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    const float one = 1.0f, zero = 0.0f;
    LT(cublasLtMatmul(lt_, p.op, &one, W, p.a, x, p.b, &zero, y, p.c, y, p.c, &p.algo, workspace_.get(),
                      workspace_.bytes(), s));
}

}  // namespace varda::cuda
