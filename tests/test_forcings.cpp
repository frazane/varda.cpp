// Dynamic forcings (src/varda/forcings.h, docs/forcings.md):
//  1. date helpers and error handling (no data needed);
//  2. bitwise comparison, per model / date / forcing, with anemoi's own ComputedForcings output
//     ($VARDA_SCRATCH/forcings/<model>.safetensors, written by python/reference/make_forcings_fixture.py)
//     computed from the grid.latitudes / grid.longitudes of the converted file
//     ($VARDA_SCRATCH/converted/<model>/model.safetensors);
//  3. thread-count invariance.
// Reports max abs difference, number of elements that are not bitwise equal, and the max ulp distance.
// Environment: VARDA_SCRATCH (tools/env.sh; unset: the data part is skipped), VARDA_FORCINGS_DIR (fixtures),
// VARDA_FORCINGS_DUMP=<file.safetensors> writes the Varda forcings for 24 six-hourly dates from
// 2024-06-01T06:00 (tensor "values" F32 [23, G, 5]) for python/reference/compare_forcings_recorded.py.
// Fixtures missing: the data part is skipped with a message (exit 0).
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "varda/forcings.h"
#include "varda/format.h"

using namespace varda;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return v && *v ? v : d;
}

bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

void test_dates() {
    auto iso = [](const char* s, int64_t h) { return format_iso8601_utc(add_hours(parse_iso8601_utc(s), h)); };
    expect(iso("2024-02-28T18:00:00", 6) == "2024-02-29T00:00:00", "leap day reached");
    expect(iso("2024-02-29T18:00:00", 6) == "2024-03-01T00:00:00", "leap day left");
    expect(iso("2023-02-28T18:00:00", 6) == "2023-03-01T00:00:00", "non-leap February");
    expect(iso("2024-12-31T18:00:00", 6) == "2025-01-01T00:00:00", "year rollover");
    expect(iso("2025-01-01T00:00:00", -1) == "2024-12-31T23:00:00", "negative hours");
    expect(iso("1900-02-28T12:00:00", 12) == "1900-03-01T00:00:00", "1900 is not a leap year");
    expect(iso("2000-02-28T12:00:00", 12) == "2000-02-29T00:00:00", "2000 is a leap year");
    expect(iso("2024-06-01T00:00:00", 24 * 366) == "2025-06-02T00:00:00", "366 days");
    expect(to_unix_seconds(parse_iso8601_utc("1970-01-01T00:00:00")) == 0, "epoch");
    expect(to_unix_seconds(parse_iso8601_utc("2024-06-01T06:00:00Z")) == 1717221600, "unix seconds");
    expect(format_iso8601_utc(parse_iso8601_utc("2024-06-01 06:00")) == "2024-06-01T06:00:00", "short / space form");
    for (const char* bad : {"", "2024-06-01", "2024-13-01T00:00:00", "2023-02-29T00:00:00", "2024-06-01T24:00:00",
                            "2024-06-01T00:00:0x", "2024/06/01T00:00:00", "2024-06-01T00:00:00.5"})
        expect(throws([&] { parse_iso8601_utc(bad); }), std::string("bad date rejected: '") + bad + "'");
    std::vector<double> lat{10.0}, lon{20.0};
    float o[2];
    expect(throws([&] { compute_forcings("2024-06-01T00:00:00", {"cos_julian_day", "bogus"}, lat.data(), lon.data(), 1, o); }),
           "unknown forcing name throws");
    expect(is_forcing_name("insolation") && !is_forcing_name("Insolation"), "is_forcing_name");
    std::printf("date helpers: %s\n", failures ? "FAIL" : "ok");
}

// Distance in representable floats between two finite floats (0 when bitwise equal; +0 vs -0 is 0).
int64_t ulp_distance(float a, float b) {
    auto key = [](float x) {
        int32_t i;
        std::memcpy(&i, &x, 4);
        return i >= 0 ? int64_t(i) : int64_t(INT32_MIN) - i;  // monotone in the float value, -0 == +0
    };
    return std::llabs(key(a) - key(b));
}

struct Totals {
    size_t elements = 0, mismatches = 0;
    double max_abs = 0;
    int64_t max_ulp = 0;
};

void test_model(const std::string& model, const std::string& scratch, const std::string& fixdir, bool dump) {
    std::string fix = fixdir + "/" + model + ".safetensors";
    const std::string conv = scratch + "/converted/" + model + "/model.safetensors";
    if (!file_exists(fix) || !file_exists(conv)) {
        std::printf("[%s] SKIPPED: need %s and %s\n", model.c_str(), fix.c_str(), conv.c_str());
        return;
    }
    SafeTensorsFile model_file(conv), fixture(fix);
    if (!model_file.has("grid.latitudes")) {
        std::printf("[%s] SKIPPED: %s has no grid.latitudes (old converter)\n", model.c_str(), conv.c_str());
        return;
    }
    const TensorView& lat = model_file.peek("grid.latitudes");
    const TensorView& lon = model_file.peek("grid.longitudes");
    expect(lat.dtype == DType::F64 && lon.dtype == DType::F64 && lat.shape.size() == 1 && lat.shape == lon.shape,
           model + ": grid dtype / shape");
    const size_t G = size_t(lat.numel());

    // the fixture was written with the coordinates read by anemoi from the checkpoint
    const TensorView& flat = fixture.peek("lat");
    const TensorView& flon = fixture.peek("lon");
    expect(flat.nbytes == lat.nbytes && std::memcmp(flat.data, lat.data, lat.nbytes) == 0 &&
               std::memcmp(flon.data, lon.data, lon.nbytes) == 0,
           model + ": converted grid.* bitwise equal to the coordinates anemoi used");

    auto names = nlohmann::json::parse(fixture.metadata("names")).get<std::vector<std::string>>();
    auto dates = nlohmann::json::parse(fixture.metadata("dates")).get<std::vector<std::string>>();
    const size_t nf = names.size();
    std::vector<float> out(G * nf);
    std::printf("[%s] G=%zu, %zu dates x %zu forcings (%s)\n", model.c_str(), G, dates.size(), nf, conv.c_str());
    std::printf("  %-21s %-15s %12s %12s %8s\n", "date", "forcing", "max_abs", "not_bitwise", "max_ulp");
    Totals total;
    std::vector<Totals> per(nf);
    for (const auto& d : dates) {
        const TensorView& ref = fixture.peek("forcings/" + d);
        expect(ref.dtype == DType::F32 && ref.shape == std::vector<int64_t>{int64_t(G), int64_t(nf)}, "fixture shape " + d);
        compute_forcings(d, names, lat.as<double>(), lon.as<double>(), G, out.data());
        const float* r = ref.as<float>();
        for (size_t k = 0; k < nf; ++k) {
            Totals t;
            for (size_t g = 0; g < G; ++g) {
                const float a = out[g * nf + k], b = r[g * nf + k];
                ++t.elements;
                if (std::memcmp(&a, &b, 4) != 0) {
                    ++t.mismatches;
                    t.max_abs = std::max(t.max_abs, std::fabs(double(a) - double(b)));
                    t.max_ulp = std::max(t.max_ulp, ulp_distance(a, b));
                }
            }
            std::printf("  %-21s %-15s %12.3e %12zu %8lld\n", d.c_str(), names[k].c_str(), t.max_abs, t.mismatches,
                        (long long)t.max_ulp);
            for (Totals* p : {&total, &per[k]}) {
                p->elements += t.elements, p->mismatches += t.mismatches;
                p->max_abs = std::max(p->max_abs, t.max_abs), p->max_ulp = std::max(p->max_ulp, t.max_ulp);
            }
        }
    }
    for (size_t k = 0; k < nf; ++k)
        std::printf("  total %-15s elements %zu not_bitwise %zu max_abs %.3e max_ulp %lld\n", names[k].c_str(),
                    per[k].elements, per[k].mismatches, per[k].max_abs, (long long)per[k].max_ulp);
    std::printf("[%s] TOTAL elements %zu, not bitwise equal %zu, max_abs %.3e, max_ulp %lld\n", model.c_str(),
                total.elements, total.mismatches, total.max_abs, (long long)total.max_ulp);
    expect(total.mismatches == 0, model + ": forcings bitwise equal to anemoi's ComputedForcings");

    // thread-count invariance (last date)
    {
        std::vector<float> a(G * nf), b(G * nf);
        const int saved = omp_get_max_threads();
        omp_set_num_threads(1);
        compute_forcings(dates.back(), names, lat.as<double>(), lon.as<double>(), G, a.data());
        omp_set_num_threads(std::max(2, saved));
        compute_forcings(dates.back(), names, lat.as<double>(), lon.as<double>(), G, b.data());
        omp_set_num_threads(saved);
        expect(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, model + ": thread-count invariance");
    }

    // column order is the caller's: a permutation / subset gives the same values
    {
        std::vector<std::string> sub{names.back(), names.front()};
        std::vector<float> s(G * 2);
        compute_forcings(dates.front(), sub, lat.as<double>(), lon.as<double>(), G, s.data());
        compute_forcings(dates.front(), names, lat.as<double>(), lon.as<double>(), G, out.data());
        bool same = true;
        for (size_t g = 0; g < G && same; ++g)
            same = std::memcmp(&s[g * 2], &out[g * nf + nf - 1], 4) == 0 && std::memcmp(&s[g * 2 + 1], &out[g * nf], 4) == 0;
        expect(same, model + ": subset / permuted names");
    }

    if (dump && model == "varda") {
        const std::string path = env_or("VARDA_FORCINGS_DUMP", "");
        const int N = 23;
        std::vector<float> all(size_t(N) * G * nf);
        DateTime t = parse_iso8601_utc("2024-06-01T06:00:00");
        std::vector<std::string> ds;
        for (int i = 0; i < N; ++i, t = add_hours(t, 6)) {
            ds.push_back(format_iso8601_utc(t));
            compute_forcings(t, names, lat.as<double>(), lon.as<double>(), G, all.data() + size_t(i) * G * nf);
        }
        write_safetensors(path, {{"values", DType::F32, {N, int64_t(G), int64_t(nf)}, all.data()}},
                          {{"names", nlohmann::json(names).dump()}, {"dates", nlohmann::json(ds).dump()}});
        std::printf("[%s] dumped %d dates to %s\n", model.c_str(), N, path.c_str());
    }
}

}  // namespace

int main() {
    test_dates();
    const std::string scratch = env_or("VARDA_SCRATCH", "");
    const std::string fixdir = env_or("VARDA_FORCINGS_DIR", scratch + "/forcings");
    const bool dump = !env_or("VARDA_FORCINGS_DUMP", "").empty();
    if (scratch.empty())
        std::printf("data part SKIPPED: VARDA_SCRATCH is not set (source tools/env.sh)\n");
    else
        for (const char* m : {"dev", "varda"}) test_model(m, scratch, fixdir, dump);
    std::printf(failures ? "test_forcings: %d FAILURE(S)\n" : "test_forcings: all ok (failures %d)\n", failures);
    return failures ? 1 : 0;
}
