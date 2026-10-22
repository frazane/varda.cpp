// Placeholder: replaced by the CUDA backend.
#include <stdexcept>

#include "varda/backend.h"

namespace varda {

std::unique_ptr<Backend> make_cuda_backend(const Config&, SafeTensorsFile&, const Options&) {
    throw std::runtime_error("CUDA backend not implemented yet");
}

}  // namespace varda
