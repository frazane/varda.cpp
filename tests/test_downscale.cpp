// Temporal downscaling loop (src/varda/downscale.h, docs/downscaler.md), on tiny synthetic name lists
// and values, without a model file:
//  1. the assembly table: coverage, name mapping (the downscaler's channel order differs from the
//     forecaster's input and output orders), constants, forcing columns;
//  2. fill_slot against the literal definition: slot 0 of window 0 from x0's newest slot (forecaster input
//     names), slot 0 of window k >= 1 from y_{k-1}, slot 1 from y_k (forecaster output names), constants
//     from x0's newest slot in both slots, forcing columns from the given forcing array;
//  3. invalid inputs (uncovered channel, listed twice, wrong index, missing constant, unknown forcing name,
//     prognostic absent from the forecaster input) are rejected with a message naming the problem;
//  4. dates and step hours: window starts date0 + 6k h, outputs start + 1..6 h, across a day, a month and a
//     leap day; a general (input [0, 3], output [1, 2], 2 h) configuration;
//  5. DownscaleTimes::from_config / DownscaleAssembly::from_config on a hand-built interpolator Config,
//     including the grid and the node-count checks of ForecasterInfo.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "varda/downscale.h"

using namespace varda;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

// Runs f, returns true if it throws and the message contains `needle`.
template <class F>
bool throws_with(F f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        const bool ok = std::string(e.what()).find(needle) != std::string::npos;
        if (!ok) std::printf("  (message was: %s)\n", e.what());
        return ok;
    }
    return false;
}

// Forecaster: inputs  a b c sdor cos_julian_day slor d   (7), outputs d c b a (4, reversed order).
// Downscaler: inputs  b sin_julian_day sdor a cos_julian_day d slor c (8, mixed order).
const std::vector<std::string> FIN = {"a", "b", "c", "sdor", "cos_julian_day", "slor", "d"};
const std::vector<std::string> FOUT = {"d", "c", "b", "a"};
const std::vector<std::string> DIN = {"b", "sin_julian_day", "sdor", "a", "cos_julian_day", "d", "slor", "c"};
const std::vector<std::string> DYN = {"cos_julian_day", "sin_julian_day"};
const std::vector<int> DYNI = {4, 1};
const std::vector<std::string> CON = {"sdor", "slor"};
const std::vector<int> CONI = {2, 6};
constexpr int64_t G = 5;

float xval(int64_t g, int v) { return 10.0f * g + v + 0.25f; }              // x0 newest slot
float yval(int s, int64_t g, int v) { return -1000.0f * (s + 1) - 10.0f * g - v - 0.5f; }  // y_s
float fval(int e, int64_t g, int k) { return 7.0f * e + g + 0.125f * k + 0.5f; }  // forcing, edge e

}  // namespace

int main() {
    // ---- 1. table
    const DownscaleAssembly a = DownscaleAssembly::make(G, DIN, DYN, DYNI, CON, CONI, FIN, FOUT);
    expect(a.V_in == 8 && a.V_fin == 7 && a.V_fout == 4 && a.F == 2, "sizes");
    const std::vector<int8_t> kinds = {0, 2, 1, 0, 2, 0, 1, 0};
    expect(a.kind == kinds, "kinds");
    // prognostic b -> out 2 / in 1; a -> out 3 / in 0; d -> out 0 / in 6; c -> out 1 / in 2
    expect(a.from_out[0] == 2 && a.from_in[0] == 1, "b mapping");
    expect(a.from_out[3] == 3 && a.from_in[3] == 0, "a mapping");
    expect(a.from_out[5] == 0 && a.from_in[5] == 6, "d mapping");
    expect(a.from_out[7] == 1 && a.from_in[7] == 2, "c mapping");
    expect(a.from_in[2] == 3 && a.from_in[6] == 5 && a.from_out[2] == -1 && a.from_out[6] == -1, "constants map into x0 by name");
    expect(a.forcing_col[4] == 0 && a.forcing_col[1] == 1 && a.forcing_col[0] == -1, "forcing columns");
    expect(a.forcing_names == DYN, "forcing names");

    // ---- 2. fill_slot against the definition
    std::vector<float> x0(G * a.V_fin);
    for (int64_t g = 0; g < G; ++g)
        for (int v = 0; v < a.V_fin; ++v) x0[g * a.V_fin + v] = xval(g, v);
    std::vector<std::vector<float>> ys(3, std::vector<float>(G * a.V_fout));
    for (int s = 0; s < 3; ++s)
        for (int64_t g = 0; g < G; ++g)
            for (int v = 0; v < a.V_fout; ++v) ys[s][g * a.V_fout + v] = yval(s, g, v);
    std::vector<std::vector<float>> fs(4, std::vector<float>(G * a.F));
    for (int e = 0; e < 4; ++e)
        for (int64_t g = 0; g < G; ++g)
            for (int k = 0; k < a.F; ++k) fs[e][g * a.F + k] = fval(e, g, k);
    auto name_in = [&](const std::string& n) { for (int i = 0; i < a.V_fin; ++i) if (FIN[i] == n) return i; return -1; };
    auto name_out = [&](const std::string& n) { for (int i = 0; i < a.V_fout; ++i) if (FOUT[i] == n) return i; return -1; };
    auto check_slot = [&](const std::vector<float>& slot, const float* state_out, bool start0, int fe, const char* tag) {
        bool ok = true;
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < a.V_in; ++c) {
                const std::string& n = DIN[c];
                float want;
                if (n == "cos_julian_day") want = fs[fe][g * a.F + 0];
                else if (n == "sin_julian_day") want = fs[fe][g * a.F + 1];
                else if (n == "sdor" || n == "slor") want = x0[g * a.V_fin + name_in(n)];
                else if (start0) want = x0[g * a.V_fin + name_in(n)];
                else want = state_out[g * a.V_fout + name_out(n)];
                ok = ok && std::memcmp(&want, &slot[g * a.V_in + c], 4) == 0;
            }
        expect(ok, std::string("fill_slot ") + tag);
    };
    std::vector<float> s0(G * a.V_in, -99.0f), s1(G * a.V_in, -99.0f);
    a.fill_slot(s0.data(), nullptr, x0.data(), x0.data(), fs[0].data());
    check_slot(s0, nullptr, true, 0, "window 0 start (from x0 by input names)");
    a.fill_slot(s1.data(), ys[0].data(), nullptr, x0.data(), fs[1].data());
    check_slot(s1, ys[0].data(), false, 1, "window 0 end (y_0 by output names)");
    a.fill_slot(s0.data(), ys[0].data(), nullptr, x0.data(), fs[1].data());
    check_slot(s0, ys[0].data(), false, 1, "window 1 start (y_0)");
    a.fill_slot(s1.data(), ys[1].data(), nullptr, x0.data(), fs[2].data());
    check_slot(s1, ys[1].data(), false, 2, "window 1 end (y_1)");
    // the start of window 0 differs from the start of window 1 where x0 and y_0 differ
    {
        std::vector<float> w0(G * a.V_in), w1(G * a.V_in);
        a.fill_slot(w0.data(), nullptr, x0.data(), x0.data(), fs[0].data());
        a.fill_slot(w1.data(), ys[0].data(), nullptr, x0.data(), fs[0].data());
        expect(w0[0] != w1[0], "k = 0 and k >= 1 start slots take different sources");
    }

    // ---- 3. invalid
    expect(throws_with([&] {  // channel 'z' is no forecaster output and no forcing
        auto bad = DIN; bad[7] = "z";
        DownscaleAssembly::make(G, bad, DYN, DYNI, CON, CONI, FIN, FOUT);
    }, "'z'"), "uncovered channel rejected");
    expect(throws_with([&] { DownscaleAssembly::make(G, DIN, DYN, {4, 4}, CON, CONI, FIN, FOUT); }, "'sin_julian_day' is listed at input channel 4"),
           "wrong index rejected");
    expect(throws_with([&] { DownscaleAssembly::make(G, DIN, DYN, DYNI, {"sdor", "sdor"}, {2, 2}, FIN, FOUT); }, "listed twice"),
           "listed twice rejected");
    expect(throws_with([&] { DownscaleAssembly::make(G, DIN, DYN, DYNI, CON, {2, 7}, FIN, FOUT); }, "which is 'c'"),
           "constant at a wrong channel rejected");
    expect(throws_with([&] {
        auto fin = FIN; fin[3] = "sdorx";
        DownscaleAssembly::make(G, DIN, DYN, DYNI, CON, CONI, fin, FOUT);
    }, "'sdor' is not a variable of the forecaster input"), "missing constant rejected");
    expect(throws_with([&] { DownscaleAssembly::make(G, DIN, {"cos_julian_day", "foo"}, DYNI, CON, CONI, FIN, FOUT); }, "not a computable forcing"),
           "unknown forcing name rejected");
    expect(throws_with([&] {
        auto fin = FIN; fin[6] = "dd";  // d is an output but not an input
        DownscaleAssembly::make(G, DIN, DYN, DYNI, CON, CONI, fin, FOUT);
    }, "window 0 starts from it"), "prognostic absent from the forecaster input rejected");
    expect(throws_with([&] { DownscaleAssembly::make(G, DIN, DYN, DYNI, CON, CONI, {"a", "a"}, FOUT); }, "duplicate"),
           "duplicate names rejected");

    // ---- 4. dates
    {
        Config c;
        c.model_kind = "enc_proc_dec_interpolator";
        c.n_input_steps = 2;
        c.n_output_steps = 6;
        c.input_times = {0, 6};
        c.output_times = {1, 2, 3, 4, 5, 6};
        c.frequency_hours = 1;
        const DownscaleTimes t = DownscaleTimes::from_config(c);
        expect(t.window_hours == 6, "window 6 h");
        const DateTime d0 = parse_iso8601_utc("2024-02-28T15:00:00");
        expect(format_iso8601_utc(t.window_start(d0, 0)) == "2024-02-28T15:00:00", "start 0");
        expect(format_iso8601_utc(t.window_start(d0, 2)) == "2024-02-29T03:00:00", "start 2 (leap day)");
        expect(format_iso8601_utc(t.window_end(d0, 1)) == "2024-02-29T03:00:00", "end 1 == start 2");
        expect(format_iso8601_utc(t.valid(d0, 3, 5)) == "2024-02-29T15:00:00", "valid (3, 5) = date0 + 24 h");
        expect(t.step_hours(0, 0) == 1 && t.step_hours(0, 5) == 6 && t.step_hours(1, 0) == 7 && t.step_hours(3, 5) == 24,
               "step hours 6k + t + 1");
        expect(format_iso8601_utc(t.valid(parse_iso8601_utc("2023-12-31T20:00:00"), 0, 4)) == "2024-01-01T01:00:00", "across year");
        // general: input [0, 3], output [1, 2], 2 h => window 6 h, outputs at +2 h and +4 h
        c.input_times = {0, 3};
        c.output_times = {1, 2};
        c.n_output_steps = 2;
        c.frequency_hours = 2;
        const DownscaleTimes g = DownscaleTimes::from_config(c);
        expect(g.window_hours == 6 && g.out_offset_hours == std::vector<int>({2, 4}), "general times");
        expect(g.step_hours(2, 1) == 16, "general step hours");
        c.output_times = {1, 4};
        expect(throws_with([&] { DownscaleTimes::from_config(c); }, "outside the window"), "output outside window rejected");
        c.model_kind = "enc_proc_dec";
        expect(throws_with([&] { DownscaleTimes::from_config(c); }, "not a temporal downscaler"), "forecaster config rejected");
    }

    // ---- 5. from_config
    {
        Config ds;
        ds.model_kind = "enc_proc_dec_interpolator";
        ds.n_data_nodes = G;
        ds.var_input = DIN;
        ds.has_forcings = true;
        ds.forcing_dynamic = DYN;
        ds.forcing_dynamic_index = DYNI;
        ds.forcing_constant = CON;
        ds.forcing_constant_index = CONI;
        ForecasterInfo fc;
        fc.in_names = FIN;
        fc.out_names = FOUT;
        fc.n_data_nodes = G;
        const DownscaleAssembly b = DownscaleAssembly::from_config(ds, fc);
        expect(b.kind == kinds && b.from_out == a.from_out && b.from_in == a.from_in, "from_config == make");
        fc.n_data_nodes = G + 1;
        expect(throws_with([&] { DownscaleAssembly::from_config(ds, fc); }, "grid mismatch"), "node count mismatch rejected");
        fc.n_data_nodes = G;
        ds.has_forcings = false;
        expect(throws_with([&] { DownscaleAssembly::from_config(ds, fc); }, "no `forcings` section"), "no forcings section rejected");
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_downscale: all checks passed\n");
    return 0;
}
