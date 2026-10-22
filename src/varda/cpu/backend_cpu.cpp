// Placeholder: replaced by the CPU fp32 backend.
#include <stdexcept>

#include "varda/backend.h"

namespace varda {

std::unique_ptr<Backend> make_cpu_backend(const Config&, SafeTensorsFile&, const Options&) {
    throw std::runtime_error("CPU backend not implemented yet");
}

}  // namespace varda
