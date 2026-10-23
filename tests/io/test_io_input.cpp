// Input-side building blocks (docs/io-input.md): dates, states, the regrid matrix files and
// y = W x in stored order. The full pipeline on real data is varda-prepare --compare.
//
// Optional real-data check (skipped unless all are set): VARDA_REGRID_MATRIX (the exported
// .safetensors), VARDA_N320_TEMPLATE (HF templates/ifs-levtype=sfc.grib), VARDA_REGRID_REF_DUMPS
// (a reference run's dumps/ with regrid.1.{in,out}.grib2) and VARDA_COSMO_DEFINITIONS: MIR's own
// input messages through our regridder must decode bitwise like MIR's output messages.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "nlohmann/json.hpp"
#include "varda/format.h"
#include "varda/io/grib.h"
#include "varda/io/input.h"
#include "varda/io/regrid.h"

using namespace varda::io;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    fails += !ok;
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

// A 2 x 3 matrix whose second row sums differently in stored and sorted order.
//   row 0: cols {1}        weights {0.5}
//   row 1: cols {0, 2, 1}  weights {1, 1, 1}   x = {1, 1e-16, -1}: stored ((1 + -1) + 1e-16) = 1e-16,
//                                              sorted ((1 + 1e-16) + -1) = 0
const std::vector<int64_t> kIndptr = {0, 1, 4};
const std::vector<int32_t> kIndices = {1, 0, 2, 1};
const std::vector<double> kWeights = {0.5, 1.0, 1.0, 1.0};

RegridGeometry tiny_geometry() {
    RegridGeometry g = RegridGeometry::ifs_open_data_0p25();
    g.Ni = 3;
    g.Nj = 1;
    g.numberOfDataPoints = 3;
    return g;
}

std::string write_tiny_safetensors(const std::string& path) {
    const RegridGeometry g = tiny_geometry();
    nlohmann::json geo = {{"gridType", g.gridType},
                          {"Ni", g.Ni},
                          {"Nj", g.Nj},
                          {"numberOfDataPoints", g.numberOfDataPoints},
                          {"latitudeOfFirstGridPointInDegrees", g.latitudeOfFirstGridPointInDegrees},
                          {"longitudeOfFirstGridPointInDegrees", g.longitudeOfFirstGridPointInDegrees},
                          {"latitudeOfLastGridPointInDegrees", g.latitudeOfLastGridPointInDegrees},
                          {"longitudeOfLastGridPointInDegrees", g.longitudeOfLastGridPointInDegrees},
                          {"iDirectionIncrementInDegrees", g.iDirectionIncrementInDegrees},
                          {"jDirectionIncrementInDegrees", g.jDirectionIncrementInDegrees},
                          {"iScansNegatively", g.iScansNegatively},
                          {"jScansPositively", g.jScansPositively},
                          {"jPointsAreConsecutive", g.jPointsAreConsecutive},
                          {"bitmapPresent", 0}};
    nlohmann::json meta = {{"source_geometry", geo}, {"n_source", 3}, {"n_target", 2}};
    varda::write_safetensors(path,
                             {{"regrid.indptr", varda::DType::I64, {3}, kIndptr.data()},
                              {"regrid.indices", varda::DType::I32, {4}, kIndices.data()},
                              {"regrid.weights", varda::DType::F64, {4}, kWeights.data()}},
                             {{"regrid", meta.dump()}});
    return path;
}

// The same matrix as MIR's eckit-Stream cache file (big-endian tags/ints, little-endian blobs).
std::string write_tiny_mat(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](int tag, uint32_t v) {
        const unsigned char b[5] = {(unsigned char)tag, (unsigned char)(v >> 24), (unsigned char)(v >> 16),
                                    (unsigned char)(v >> 8), (unsigned char)v};
        f.write(reinterpret_cast<const char*>(b), 5);
    };
    auto blob = [&](const void* p, uint64_t n) {
        unsigned char b[9] = {21};
        for (int i = 0; i < 8; ++i) b[1 + i] = (unsigned char)(n >> (56 - 8 * i));
        f.write(reinterpret_cast<const char*>(b), 9);
        f.write(static_cast<const char*>(p), std::streamsize(n));
    };
    u32(10, 2);
    u32(10, 3);
    u32(10, 4);
    u32(5, 1);  // little endian
    u32(10, 4);
    u32(10, 8);
    u32(10, 8);
    const std::vector<uint32_t> outer = {0, 1, 4}, inner = {1, 0, 2, 1};
    blob(outer.data(), outer.size() * 4);
    blob(inner.data(), inner.size() * 4);
    blob(kWeights.data(), kWeights.size() * 8);
    return path;
}

void real_regrid_check() {
    const char* mat = std::getenv("VARDA_REGRID_MATRIX");
    const char* tmpl = std::getenv("VARDA_N320_TEMPLATE");
    const char* dumps = std::getenv("VARDA_REGRID_REF_DUMPS");
    const char* defs = std::getenv("VARDA_COSMO_DEFINITIONS");
    if (!mat || !tmpl || !dumps || !defs) {
        std::printf("skip real-data regrid check (VARDA_REGRID_MATRIX / VARDA_N320_TEMPLATE / "
                    "VARDA_REGRID_REF_DUMPS / VARDA_COSMO_DEFINITIONS unset)\n");
        return;
    }
    set_definitions_path(defs);
    const RegridMatrix m = RegridMatrix::load(mat);
    const Regridder rg(m, read_first_message(tmpl));
    const auto ins = read_messages(std::string(dumps) + "/regrid.1.in.grib2");
    const auto outs = read_messages(std::string(dumps) + "/regrid.1.out.grib2");
    expect(ins.size() == outs.size() && !ins.empty(), "regrid.1: " + std::to_string(ins.size()) + " fields");
    for (size_t i = 0; i < ins.size() && i < outs.size(); ++i) {
        std::vector<float> f;
        std::vector<double> d;
        rg.regrid(ins[i], &f, &d);
        const auto rf = outs[i].get_float_array("values");
        const auto rd = outs[i].get_double_array("values");
        const bool ok = f.size() == rf.size() && std::memcmp(f.data(), rf.data(), f.size() * 4) == 0 &&
                        std::memcmp(d.data(), rd.data(), d.size() * 8) == 0;
        expect(ok, "MIR input -> regrid + 16-bit re-encode == MIR output, bitwise: " + ins[i].get_string("shortName"));
    }
    // A message from another grid is refused.
    expect(throws([&] { m.check_message(outs[0]); }), "N320 message refused as regrid source (geometry check)");
}

}  // namespace

int main() {
    // dates
    const varda::DateTime d = parse_input_date("2026-10-22T15");
    expect(d.year == 2026 && d.month == 10 && d.day == 22 && d.hour == 15 && d.minute == 0, "date YYYY-MM-DDTHH");
    expect(varda::format_iso8601_utc(parse_input_date("2026-10-22T15:00:00")) == "2026-10-22T15:00:00",
           "date YYYY-MM-DDTHH:MM:SS");
    expect(throws([] { parse_input_date("2026-10-22"); }), "bad date throws");

    // states
    FieldState s(2, 3);
    std::vector<float>& a = s.add("t_500");
    expect(a.size() == 6 && std::isnan(a[0]), "FieldState::add is NaN-filled [n_dates * n_points]");
    expect(throws([&] { s.add("t_500"); }), "duplicate field throws");
    expect(throws([&] { s.at("u_500"); }), "missing field throws");

    // levels of model_to_pressure_level
    expect(kenda_interpolate_levels().size() == 13 && kenda_extrapolate_levels().size() == 6, "configured levels");

    // regrid matrix files and stored-order summation
    const std::string dir = ".";
    const std::string st_path = write_tiny_safetensors(dir + "/test_io_input.regrid.safetensors");
    const std::string mat_path = write_tiny_mat(dir + "/test_io_input.regrid.mat");
    const RegridMatrix m = RegridMatrix::load(st_path);
    const RegridMatrix mm = RegridMatrix::load_mir_mat(mat_path, tiny_geometry());
    expect(m.n_target() == 2 && m.n_source() == 3 && m.nnz() == 4, "safetensors matrix shape");
    expect(m.same_matrix(mm), "safetensors matrix == eckit .mat matrix (stored order kept)");
    const double x[3] = {1.0, 1e-16, -1.0};
    double y[2];
    m.apply(x, y);
    expect(y[0] == 0.5e-16 && y[1] == 1e-16, "y = W x summed in stored order ((1 + -1) + 1e-16 = 1e-16, not 0)");
    std::remove(st_path.c_str());
    std::remove(mat_path.c_str());

    real_regrid_check();
    return fails ? 1 : 0;
}
