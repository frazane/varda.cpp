#pragma once
#include <cstdint>
#include <string>

namespace varda::metal {
struct DeviceInfo {
    std::string name;
    uint64_t max_buffer_length = 0;
    uint64_t recommended_working_set = 0;
};
// No CPU fallback: false means there is no Metal device with unified memory.
bool available();
DeviceInfo device_info();
}
