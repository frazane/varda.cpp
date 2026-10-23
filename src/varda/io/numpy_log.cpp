#include "varda/io/numpy_log.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace varda::io {

namespace {

// numpy/_core/src/umath/npy_simd_data.h (NPY_COEFF_{P,Q}{0..5}_LOGf), float literals.
constexpr float P0 = 0.000000000000000000000e+00f;
constexpr float P1 = 9.999999999999998702752e-01f;
constexpr float P2 = 2.112677543073053063722e+00f;
constexpr float P3 = 1.480000633576506585156e+00f;
constexpr float P4 = 3.808837741388407920751e-01f;
constexpr float P5 = 2.589979117907922693523e-02f;
constexpr float Q0 = 1.000000000000000000000e+00f;
constexpr float Q1 = 2.612677543073109236779e+00f;
constexpr float Q2 = 2.453006071784736363091e+00f;
constexpr float Q3 = 9.864942958519418960339e-01f;
constexpr float Q4 = 1.546476374983906719538e-01f;
constexpr float Q5 = 5.875095403124574342950e-03f;
constexpr float LOGE2 = 0.693147180559945309417232121458176568f;   // NPY_LOGE2f
constexpr float SQRT1_2 = 0.707106781186547524400844362104849039f; // NPY_SQRT1_2f

inline uint32_t bits_of(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
inline float float_of(uint32_t u) {
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

}  // namespace

float logf_np(float x_in) {
    // Special-case lanes (applied last in numpy, in this order: nan, x < 0, x == 0, +inf).
    if (x_in != x_in) return float_of(0x7fc00000u);         // NPY_NANF
    if (x_in < 0.0f) return float_of(0xffc00000u);          // -NPY_NANF
    if (x_in == 0.0f) return -std::numeric_limits<float>::infinity();
    if (x_in == std::numeric_limits<float>::infinity()) return x_in;

    // fma_get_exponent / fma_get_mantissa: denormals are multiplied by 2^100 (exact), their
    // exponent reduced by 100.
    const bool denormal = x_in < std::numeric_limits<float>::min();
    const float x = denormal ? x_in * float_of(0x71800000u) : x_in;
    const uint32_t b = bits_of(x);
    float e = static_cast<float>(static_cast<int32_t>(b >> 23) - 0x7E);
    if (denormal) e = e - 100.0f;
    float m = float_of((b & 0x7fffffu) | (126u << 23));  // in [0.5, 1)

    if (m <= SQRT1_2) {
        m = m + m;
        e = e - 1.0f;
    }
    m = m - 1.0f;

    float num = std::fmaf(P5, m, P4);
    num = std::fmaf(num, m, P3);
    num = std::fmaf(num, m, P2);
    num = std::fmaf(num, m, P1);
    num = std::fmaf(num, m, P0);
    float den = std::fmaf(Q5, m, Q4);
    den = std::fmaf(den, m, Q3);
    den = std::fmaf(den, m, Q2);
    den = std::fmaf(den, m, Q1);
    den = std::fmaf(den, m, Q0);
    const float poly = num / den;
    return std::fmaf(e, LOGE2, poly);
}

void logf_np(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = logf_np(x[i]);
}

}  // namespace varda::io
