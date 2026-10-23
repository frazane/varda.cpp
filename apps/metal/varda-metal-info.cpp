// Exit 77 explicitly distinguishes a GPU-less runner from a successful GPU test.
#include <cstdio>
#include <exception>
#include "varda/metal/metal_backend.h"

int main() {
    if (!varda::metal::available()) {
        std::fprintf(stderr, "no unified-memory Metal GPU is available\n");
        return 77;
    }
    try {
        const auto d = varda::metal::device_info();
        std::printf("Metal GPU: %s\nmax buffer: %.2f GiB\nrecommended working set: %.2f GiB\n",
                    d.name.c_str(), d.max_buffer_length / 1073741824.0,
                    d.recommended_working_set / 1073741824.0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
