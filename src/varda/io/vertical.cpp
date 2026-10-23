#include "varda/io/vertical.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "varda/io/numpy_log.h"

namespace varda::io {

namespace {

// VI:233-238: Python floats, applied to float32 arrays (numpy's weak scalars -> float32).
constexpr float LAPSE_RATE = 0.0065f;
constexpr float H1 = 2000.0f;
constexpr float H2 = 2500.0f;
constexpr float T1 = 298.0f;
constexpr float G = 9.80665f;
constexpr float R_D = 287.05f;

}  // namespace

void clamp_top_pressure(float* p_top, size_t N) {
    // p.where(p < 5000, 5000 - 1e-5): the Python float 4999.99999 is stored into a float32 array.
    const float c = static_cast<float>(5000.0 - 1e-5);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < int64_t(N); ++n)
        if (!(p_top[n] < 5000.0f)) p_top[n] = c;
}

void check_pressure_increasing(const float* p, size_t K, size_t N) {
    int64_t bad = 0, first = -1;
#pragma omp parallel for reduction(+ : bad) schedule(static)
    for (int64_t n = 0; n < int64_t(N); ++n) {
        for (size_t k = 0; k + 1 < K; ++k)
            if (!(p[(k + 1) * N + n] - p[k * N + n] > 0.0f)) {  // dtdz > 0, NaN fails
                ++bad;
                break;
            }
    }
    if (bad) {
        for (size_t n = 0; n < N && first < 0; ++n)
            for (size_t k = 0; k + 1 < K; ++k)
                if (!(p[(k + 1) * N + n] - p[k * N + n] > 0.0f)) {
                    first = int64_t(n);
                    break;
                }
        throw std::runtime_error("model-level pressure is not strictly increasing downwards at " + std::to_string(bad) +
                                 " points (first: point " + std::to_string(first) +
                                 "); earthkit-meteo raises here (docs/spec-io.md §2.2)");
    }
}

void geopotential_from_hhl(const float* hhl, size_t K_half, size_t N, float* fi) {
    if (K_half < 2) throw std::runtime_error("geopotential_from_hhl: need at least 2 half levels");
    const size_t K = K_half - 1;
#pragma omp parallel for schedule(static)
    for (int64_t kn = 0; kn < int64_t(K * N); ++kn) {
        const size_t k = size_t(kn) / N, n = size_t(kn) % N;
        const float s = hhl[k * N + n] + hhl[(k + 1) * N + n];
        const float half = 0.5f * s;
        fi[kn] = half * 9.80665f;
    }
}

LogPWeights log_p_weights(const float* p, size_t K, size_t N, const std::vector<int>& levels_hpa) {
    LogPWeights w;
    w.N = N;
    w.levels_hpa = levels_hpa;
    const size_t L = levels_hpa.size();
    w.k.assign(L * N, -1);
    w.r.assign(L * N, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> t0(L), log_t0(L);
    for (size_t j = 0; j < L; ++j) {
        t0[j] = double(levels_hpa[j]) * 100.0;  // np.array(sorted(target_p)) * 100.0
        log_t0[j] = std::log(t0[j]);             // np.log of a Python float: glibc log
    }
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < int64_t(N); ++n) {
        for (size_t j = 0; j < L; ++j) {
            // coords.where((coords >= t0) & (tkm1 <= t0)).fillna(inf).argmin: with p increasing,
            // the smallest qualifying k (tkm1 is NaN at k = 0, so k >= 1).
            int32_t ks = -1;
            for (size_t k = 1; k < K; ++k) {
                const double pk = p[k * N + n], pkm1 = p[(k - 1) * N + n];
                if (pk >= t0[j] && pkm1 <= t0[j]) {
                    ks = int32_t(k);
                    break;
                }
            }
            if (ks < 0) continue;
            const float l2 = logf_np(p[size_t(ks) * N + n]);
            const float l1 = logf_np(p[size_t(ks - 1) * N + n]);
            const double num = log_t0[j] - double(l1);  // np.float64 - float32 array -> float64
            const float den = l2 - l1;                  // float32 - float32
            w.k[j * N + n] = ks;
            w.r[j * N + n] = num / double(den);
        }
    }
    return w;
}

float extrapolate_temperature(float t2m, float hsurf, float ps, int p_target_pa) {
    const float lg = logf_np(float(p_target_pa) / ps);  // np.log(p_target / p_sfc), int / float32
    // _vertical_extrapolation_t_zero_prime
    const float t = t2m + LAPSE_RATE * hsurf;
    const float t_min = std::isnan(t) ? t : std::min(t, T1);  // np.minimum propagates NaN
    const float tzp = hsurf > H2 ? t_min : t_min * 0.5f + t * 0.5f;
    // _vertical_extrapolation_lapse_rate (both branches are float32)
    float lr;
    if (hsurf < H1) {
        lr = LAPSE_RATE;
    } else {
        const float d = tzp - t2m;
        const float mx = std::isnan(d) ? d : std::max(d, 0.0f);  // np.maximum propagates NaN
        lr = (1.0f / hsurf) * mx;
    }
    const float y = ((lr * R_D) / G) * lg;
    const float y3 = std::pow(y, 3.0f);  // y**3: numpy's float32 power loop -> powf
    return t2m * (((1.0f + y) + (y * y) / 2.0f) + y3 / 6.0f);
}

float extrapolate_geopotential(float hsurf, float t2m, float ps, int p_target_pa) {
    const float lg = logf_np(float(p_target_pa) / ps);
    // lapse_rate * r_d / g with lapse_rate = LAPSE_RATE: Python floats, rounded once to float32.
    const float c = static_cast<float>((0.0065 * 287.05) / 9.80665);
    const float y = c * lg;
    return hsurf * G - ((R_D * t2m) * lg) * ((1.0f + y / 2.0f) + (y * y) / 6.0f);
}

void interpolate_extrapolate(const float* x, size_t K, const LogPWeights& w, const float* t2m, const float* ps,
                             const float* hsurf, ExtrapKind kind, const std::vector<int>& extrap_hpa, float* out) {
    const size_t N = w.N, L = w.levels_hpa.size();
    std::vector<char> extrap(L, 0);
    for (size_t j = 0; j < L; ++j)
        extrap[j] = std::find(extrap_hpa.begin(), extrap_hpa.end(), w.levels_hpa[j]) != extrap_hpa.end();
    const float nan = std::numeric_limits<float>::quiet_NaN();
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < int64_t(N); ++n) {
        for (size_t j = 0; j < L; ++j) {
            const int32_t k = w.k[j * N + size_t(n)];
            float v = nan;
            if (k > 0) {
                const double r = w.r[j * N + size_t(n)];
                const double a = (1.0 - r) * double(x[size_t(k - 1) * N + size_t(n)]);
                const double b = r * double(x[size_t(k) * N + size_t(n)]);
                v = static_cast<float>(a + b);
            }
            if (std::isnan(v) && extrap[j]) {
                const int pt = w.levels_hpa[j] * 100;
                switch (kind) {
                    case ExtrapKind::Temperature:
                        v = extrapolate_temperature(t2m[n], hsurf[n], ps[n], pt);
                        break;
                    case ExtrapKind::Geopotential:
                        v = extrapolate_geopotential(hsurf[n], t2m[n], ps[n], pt);
                        break;
                    case ExtrapKind::LowestLevel:
                        v = x[(K - 1) * N + size_t(n)];
                        break;
                }
            }
            out[j * N + size_t(n)] = v;
        }
    }
}

}  // namespace varda::io
