// numpy's float32 natural logarithm, bit for bit (docs/spec-io.md §2.5).
//
// On the AVX2+FMA3 nodes the reference runs on (AMD EPYC 7713, numpy 2.5.4 dispatch X86_V3),
// np.log on float32 is numpy's own SIMD routine (simd_log_FLOAT in
// numpy/_core/src/umath/loops_exponent_log.dispatch.c.src), not glibc logf: they differ on about
// 3% of pressure values. The vertical interpolation (§2.2) takes its logs through this function.
// Verified against numpy over every positive finite float32 (docs/io-input.md).
#pragma once

#include <cstddef>

namespace varda::io {

// x > 0 finite: numpy's rational approximation (exponent/mantissa split, fused multiply-adds);
// denormals are scaled by 2^100 first, as numpy does. NaN -> NaN, x < 0 -> -NaN, ±0 -> -inf,
// +inf -> +inf (numpy's special-case lanes).
float logf_np(float x);

// y[i] = logf_np(x[i]); x and y may alias.
void logf_np(const float* x, float* y, size_t n);

}  // namespace varda::io
