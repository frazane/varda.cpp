// logf_np vs numpy's float32 np.log (docs/spec-io.md §2.5, docs/io-input.md §5).
//
// Without arguments (ctest): special values and a table of np.log results recorded with numpy
// 2.5.4 (reference env) on an AMD EPYC 7713 (dispatch X86_V3).
// With --stdin: the exhaustive check. stdin carries np.log(x) as raw little-endian float32 for
// every positive finite float32 x in bit order (0x00000001 .. 0x7f7fffff, 2 139 095 039 values),
// e.g. from the reference env:
//   python -I -c "import numpy as np,sys; o=sys.stdout.buffer
//   for s in range(1, 0x7f800000, 1<<24):
//       o.write(np.log(np.arange(s, min(s+(1<<24), 0x7f800000), dtype=np.uint32).view(np.float32)).tobytes())"
//   | test_io_numpy_log --stdin
// Every value must be bitwise equal; mismatches are counted and the first ones printed.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "varda/io/numpy_log.h"

using varda::io::logf_np;

namespace {

uint32_t bits_of(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
float float_of(uint32_t u) {
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

// {bits(x), bits(np.log(x))}: numpy 2.5.4, EPYC 7713.
const uint32_t kTable[][2] = {
    {0x00000001, 0xc2ce8ed0}, {0x00800000, 0xc2aeac50}, {0x0da24260, 0xc28a27b5}, {0x3a83126f, 0xc0dd0c55},
    {0x3f000000, 0xbf317218}, {0x3f3504f3, 0xbeb17218}, {0x3f3504f4, 0xbeb17215}, {0x3f800000, 0x00000000},
    {0x3f800001, 0x33fffffe}, {0x40000000, 0x3f317218}, {0x40400000, 0x3f8c9f54}, {0x41200000, 0x40135d8e},
    {0x45960000, 0x41079f37}, {0x4597f0fc, 0x4107d3e3}, {0x459c4000, 0x4108466c}, {0x47435000, 0x412d1dd0},
    {0x47a60400, 0x41359b44}, {0x47c5e680, 0x41386adc}, {0x7f7fffff, 0x42b17218}, {0x00105564, 0xc2b2ca6c},
};

int run_stdin() {
    const uint32_t first = 1, end = 0x7f800000u;  // positive finite: [first, end)
    const size_t chunk = size_t(1) << 24;
    std::vector<float> ref(chunk);
    uint64_t checked = 0, bad = 0;
    for (uint64_t s = first; s < end; s += chunk) {
        const size_t n = size_t(std::min<uint64_t>(chunk, end - s));
        if (std::fread(ref.data(), 4, n, stdin) != n) {
            std::printf("FAIL short read at x = 0x%08llx (%llu values checked)\n", (unsigned long long)s,
                        (unsigned long long)checked);
            return 1;
        }
        uint64_t bad_chunk = 0;
#pragma omp parallel for reduction(+ : bad_chunk) schedule(static)
        for (int64_t i = 0; i < int64_t(n); ++i) {
            const float x = float_of(uint32_t(s + uint64_t(i)));
            if (bits_of(logf_np(x)) != bits_of(ref[size_t(i)])) ++bad_chunk;
        }
        for (size_t i = 0, shown = bad; bad_chunk && i < n && shown < 20; ++i) {
            const float x = float_of(uint32_t(s + i));
            const float y = logf_np(x);
            if (bits_of(y) != bits_of(ref[i])) {
                std::printf("  differ: x = %.9g (0x%08x): logf_np 0x%08x, numpy 0x%08x\n", x, bits_of(x), bits_of(y),
                            bits_of(ref[i]));
                ++shown;
            }
        }
        bad += bad_chunk;
        checked += n;
    }
    char extra;
    if (std::fread(&extra, 1, 1, stdin) == 1) {
        std::printf("FAIL stdin has more data than the positive finite floats\n");
        return 1;
    }
    std::printf("%s exhaustive: %llu positive finite float32 values, %llu differ from numpy\n", bad ? "FAIL" : "ok  ",
                (unsigned long long)checked, (unsigned long long)bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--stdin") return run_stdin();

    int fails = 0;
    auto expect = [&](bool ok, const std::string& what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
        fails += !ok;
    };
    int table_bad = 0;
    for (const auto& r : kTable) {
        const float y = logf_np(float_of(r[0]));
        if (bits_of(y) != r[1]) {
            std::printf("  x 0x%08x: logf_np 0x%08x, numpy 0x%08x\n", r[0], bits_of(y), r[1]);
            ++table_bad;
        }
    }
    expect(table_bad == 0, "numpy table (" + std::to_string(sizeof(kTable) / sizeof(kTable[0])) + " values)");
    // A value where numpy and glibc disagree: log(1 + 2^-23).
    expect(bits_of(logf_np(float_of(0x3f800001))) == 0x33fffffe, "log(1+ulp) is numpy's 0x33fffffe");
    const float inf = INFINITY;
    expect(std::isnan(logf_np(NAN)), "nan -> nan");
    expect(std::isnan(logf_np(-1.0f)) && std::signbit(logf_np(-1.0f)), "x < 0 -> -nan");
    expect(std::isnan(logf_np(-inf)), "-inf -> nan");
    expect(logf_np(0.0f) == -inf && logf_np(-0.0f) == -inf, "±0 -> -inf");
    expect(logf_np(inf) == inf, "inf -> inf");
    std::vector<float> v = {1.0f, 2.0f, 4862.123f}, w(3);
    varda::io::logf_np(v.data(), w.data(), v.size());
    expect(bits_of(w[2]) == 0x4107d3e3 && w[0] == 0.0f, "array form");
    return fails ? 1 : 0;
}
