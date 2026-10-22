// Device-level interface of the CUDA backend (docs/cuda.md §6), for callers that keep data on the
// GPU, e.g. the autoregressive rollout:
//
//   auto model = varda::Model::load(path, opts);            // opts.device = Device::CUDA
//   varda::cuda::CudaBackend* be = varda::cuda::as_cuda(*model);
//   be->normalise(x_dev, xn_dev, s);                        // physical -> normalised window
//   be->forward_normalised(xn_dev, yn_dev, s);              // one step, normalised in and out
//   be->denormalise(yn_dev, y_dev, s);                      // normalised -> physical
//
// Every call only enqueues work on `s`: no host synchronisation, no allocation, no host-side
// state change after the first call (cuBLASLt plans are created on the first eager run of a
// shape), so a caller may capture any sequence of calls into its own CUDA graph once the model
// ran one forward pass eagerly (prepare() does that). The calls share the backend's internal
// scratch buffers: at most one forward may be in flight per backend, and all calls must be
// ordered on one stream (or with events). Pointers are device pointers, fp32, contiguous.
//
//   predict_step_device(x, y, s) == denormalise(forward_normalised(normalise(x)))   bitwise
//   Model::predict_step (host pointers)  == predict_step_device on device copies          bitwise
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "varda/backend.h"
#include "varda/varda.h"

namespace varda::cuda {

class CudaBackend : public Backend {
public:
    // Element counts of the device tensors.
    virtual int64_t input_numel() const = 0;       // x, xn: (1, T_in, [1,] G, V_in)
    virtual int64_t output_numel() const = 0;      // yn, y: (1, 1, 1, G, V_out)

    // x (1, T_in, G, V_in) physical -> xn (1, T_in, 1, G, V_in): x * mul, rounded, then + add.
    virtual void normalise(const float* x, float* xn, cudaStream_t s) = 0;
    // xn normalised window -> yn (1, 1, 1, G, V_out) normalised, after the residual and bounding.
    // xn and yn must not alias the backend's own buffers; xn is only read.
    virtual void forward_normalised(const float* xn, float* yn, cudaStream_t s) = 0;
    // yn normalised -> y physical: (yn - add) / mul.
    virtual void denormalise(const float* yn, float* y, cudaStream_t s) = 0;
    // The composition of the three, through internal xn / yn buffers.
    virtual void predict_step_device(const float* x, float* y, cudaStream_t s) = 0;

    // Runs one eager forward pass on internal buffers (if none ran yet), so that every cuBLASLt plan
    // exists before a caller's graph capture. Synchronises.
    virtual void prepare() = 0;

    virtual int device() const = 0;               // CUDA device ordinal
    virtual size_t weight_bytes() const = 0;      // device bytes held by weights and graphs
    virtual size_t activation_bytes() const = 0;  // device bytes of activation buffers
    virtual int64_t decoder_chunk_rows() const = 0;
};

// The CUDA backend of a loaded model, or nullptr if it runs on another device.
CudaBackend* as_cuda(Model& m);

}  // namespace varda::cuda
