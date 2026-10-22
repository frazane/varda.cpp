// Internal CUDA helpers: error checks, device buffers, element types.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace varda::cuda {

[[noreturn]] inline void cuda_fail(cudaError_t e, const char* what, const char* file, int line) {
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) +
                             ") in " + what + " at " + file + ":" + std::to_string(line));
}

#define VARDA_CUDA_CHECK(expr)                                                         \
    do {                                                                               \
        cudaError_t _e = (expr);                                                       \
        if (_e != cudaSuccess) ::varda::cuda::cuda_fail(_e, #expr, __FILE__, __LINE__); \
    } while (0)

// Checks the last kernel launch (launch errors only; no synchronisation).
#define VARDA_CUDA_LAUNCH() VARDA_CUDA_CHECK(cudaGetLastError())

// Element type of activations and weights on the device.
enum class Elt { F32, F16 };
inline size_t elt_size(Elt t) { return t == Elt::F32 ? 4 : 2; }

// Owning device allocation (cudaMalloc, 256-byte aligned).
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { alloc(bytes); }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
    void alloc(size_t bytes) {
        reset();
        if (bytes) VARDA_CUDA_CHECK(cudaMalloc(&p_, bytes));
        n_ = bytes;
    }
    void reset() {
        if (p_) cudaFree(p_);
        p_ = nullptr;
        n_ = 0;
    }
    void* get() const { return p_; }
    template <class T> T* as() const { return static_cast<T*>(p_); }
    size_t bytes() const { return n_; }

private:
    void* p_ = nullptr;
    size_t n_ = 0;
};

inline size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

}  // namespace varda::cuda
