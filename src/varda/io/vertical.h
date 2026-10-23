// KENDA-CH1 model levels -> pressure levels, as anemoi-plugins-meteoswiss' `model_to_pressure_level`
// filter does it (docs/spec-io.md §2.2), bit for bit: linear interpolation in log p
// (earthkit-meteo interpolate_monotonic), extrapolation below the surface for the configured
// levels, and geopotential on full levels from HHL.
//
// Arrays on model levels are [K][N] row-major, k = 0 is model level 1 (the top), as the
// filter's xarray sorts them. Float32 in and out; every operation is rounded where the reference
// rounds it (no FMA contraction: the library is built with -ffp-contract=off). Logs of float32
// pressures are numpy's (logf_np), logs of the double targets glibc's log, cubes glibc's powf.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace varda::io {

// Which extrapolation rule a variable gets below the lowest model level (VI:217-228).
enum class ExtrapKind {
    Temperature,  // T: extrapolate_temperature_sfc2p
    Geopotential, // FI: extrapolate_geopotential_sfc2p
    LowestLevel,  // U, V, QV, ...: the value at the lowest model level
};

// Step 1 of §2.2: p[0][n] = p[0][n] < 5000 ? p[0][n] : float(5000 - 1e-5) (= 5000.0f).
void clamp_top_pressure(float* p_top, size_t N);

// Throws std::runtime_error unless p[k][n] < p[k+1][n] for every k and n (earthkit raises
// "target data is not monotonic" otherwise; a decreasing p is not supported here).
void check_pressure_increasing(const float* p, size_t K, size_t N);

// Step 3: fi[k][n] = f32(f32(0.5f * f32(hhl[k][n] + hhl[k+1][n])) * 9.80665f), k = 0 .. K_half-2.
void geopotential_from_hhl(const float* hhl, size_t K_half, size_t N, float* fi);

// The interpolation weights in log p for a set of target levels (hPa), shared by every variable
// of one valid time: for target j and point n, k[j][n] is the smallest model level k >= 1 with
// p[k] >= t0 && p[k-1] <= t0 (t0 = level * 100.0 Pa), or -1 if none (the value is then NaN), and
//   r[j][n] = (log(t0) - f64(logf_np(p[k-1]))) / f64(logf_np(p[k]) - logf_np(p[k-1]))   (double).
// A std::allocator whose value-initialisation leaves trivial types uninitialised, so that large
// buffers are first written (and paged in) by the parallel loops that fill them.
template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    template <class U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };
    DefaultInitAllocator() = default;
    template <class U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... A>
    void construct(U* p, A&&... a) {
        ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
    }
};

struct LogPWeights {
    size_t N = 0;
    std::vector<int> levels_hpa;
    std::vector<int32_t, DefaultInitAllocator<int32_t>> k;  // [L][N]
    std::vector<double, DefaultInitAllocator<double>> r;    // [L][N]
};
LogPWeights log_p_weights(const float* p, size_t K, size_t N, const std::vector<int>& levels_hpa);

// interpolate_extrapolate (VI:193-229) for one variable x [K][N] -> out [L][N] (L = levels of `w`):
//   out = f32((1.0 - r) * f64(x[k-1]) + r * f64(x[k]))  where bracketed, NaN elsewhere;
//   then, for the levels in `extrap_hpa` and only where out is NaN, the extrapolation of `kind`
//   from t2m, ps, hsurf [N] (float32 throughout, numpy's promotion rules).
void interpolate_extrapolate(const float* x, size_t K, const LogPWeights& w, const float* t2m, const float* ps,
                             const float* hsurf, ExtrapKind kind, const std::vector<int>& extrap_hpa, float* out);
// The same, level j written to out_levels[j][0 .. N).
void interpolate_extrapolate(const float* x, size_t K, const LogPWeights& w, const float* t2m, const float* ps,
                             const float* hsurf, ExtrapKind kind, const std::vector<int>& extrap_hpa,
                             float* const* out_levels);

// The single-point extrapolations (VI:248-347), p_target in Pa.
float extrapolate_temperature(float t2m, float hsurf, float ps, int p_target_pa);
float extrapolate_geopotential(float hsurf, float t2m, float ps, int p_target_pa);

}  // namespace varda::io
