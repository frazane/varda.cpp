// Internal: what a compute backend provides. Model::load builds one per device.
#pragma once

#include <memory>

#include "varda/config.h"
#include "varda/format.h"
#include "varda/varda.h"

namespace varda {

class Backend {
public:
    virtual ~Backend() = default;
    // Same contract as Model::predict_step (host pointers, physical units).
    virtual void predict_step(const float* x, float* y, const ActivationHook* hook) = 0;
};

// Each factory take()s every tensor it reads from `file`; Model::load then requires that
// every tensor was consumed. `file` outlives the backend, so host weights may point into
// the mapping.
std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, SafeTensorsFile& file,
                                          const Options& opts);
#ifdef VARDA_WITH_CUDA
std::unique_ptr<Backend> make_cuda_backend(const Config& cfg, SafeTensorsFile& file,
                                           const Options& opts);
#endif

}  // namespace varda
