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
//  6. CUDA builds, with VARDA_TEST_TD_MODEL=<an interpolator's converted file> (e.g. the synthetic model g,
//     tests/run_synthetic.sh) and a GPU: the device-resident loop (cuda/downscale_cuda.h) against the host
//     loop on the same CUDA model, with a deterministic synthetic forecaster (names from the downscaler's
//     own, as varda-downscale --smoke): every window's assembled input and every output bitwise equal,
//     with the states fed from the host (y_at) and from device buffers (push_device, as the device rollout
//     does, with the producer reusing its two buffers only after the returned event), with and without the
//     hold buffer, with staging chunks that do not divide a state, a ring of one and of two states, fp32 and
//     fp16; with two GPUs also with the producer on the other GPU (peer copies). Skipped (with a note) otherwise.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "varda/downscale.h"
#ifdef VARDA_WITH_CUDA
#include <cuda_runtime_api.h>

#include "varda/cuda/cuda_backend.h"
#include "varda/cuda/downscale_cuda.h"
#endif

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

#ifdef VARDA_WITH_CUDA
#define CK(e)                                                                                       \
    do {                                                                                            \
        cudaError_t _e = (e);                                                                       \
        if (_e != cudaSuccess) throw std::runtime_error(std::string(#e) + ": " + cudaGetErrorString(_e)); \
    } while (0)

float sval(int64_t s, int64_t g, int64_t v) {  // deterministic, distinct per (state, point, variable)
    uint64_t h = uint64_t(s + 1) * 0x9E3779B97F4A7C15ull ^ uint64_t(g + 1) * 0xBF58476D1CE4E5B9ull ^
                 uint64_t(v + 1) * 0x94D049BB133111EBull;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return float(double(h % 2000001) / 1000000.0 - 1.0);
}

// Section 6: the device loop against the host loop on one model file.
void test_device_loop(const std::string& path, Precision prec) {
    Options opts;
    opts.device = Device::CUDA;
    opts.precision = prec;
    auto model = Model::load(path, opts);
    const Config& cfg = model->config();
    if (!cfg.has_forcings) {
        std::printf("note: %s has no `forcings` section (no assembly table): device loop test skipped\n", path.c_str());
        return;
    }
    ForecasterInfo fi;
    fi.in_names = cfg.var_input;
    fi.out_names = cfg.var_output;
    fi.n_data_nodes = cfg.n_data_nodes;
    fi.lat = model->latitudes();
    fi.lon = model->longitudes();
    Downscaler ds(*model, fi);
    const int64_t Gm = cfg.n_data_nodes, V_fin = int64_t(fi.in_names.size()), V_fout = int64_t(fi.out_names.size());
    const int N = 4, T_f = 2;
    std::vector<float> x0(size_t(T_f * Gm * V_fin));
    for (int t = 0; t < T_f; ++t)
        for (int64_t g = 0; g < Gm; ++g)
            for (int64_t v = 0; v < V_fin; ++v) x0[size_t((t * Gm + g) * V_fin + v)] = sval(1000 + t, g, v);
    std::vector<std::vector<float>> ys(N, std::vector<float>(size_t(Gm * V_fout)));
    for (int s = 0; s < N; ++s)
        for (int64_t g = 0; g < Gm; ++g)
            for (int64_t v = 0; v < V_fout; ++v) ys[size_t(s)][size_t(g * V_fout + v)] = sval(s, g, v);
    const DateTime date0 = parse_iso8601_utc("2024-02-28T18:00:00");  // crosses the leap day
    struct Run {
        std::vector<std::vector<float>> x, y;
        std::vector<int> steps;
    };
    auto request = [&](Run& r) {
        DownscaleRequest rq;
        rq.x0 = x0.data();
        rq.forecaster_T = T_f;
        rq.date0 = date0;
        rq.n_windows = N;
        rq.y_at = [&](int s) { return ys[size_t(s)].data(); };
        const size_t xn = size_t(2 * Gm * cfg.n_input_vars), yn = size_t(Gm * cfg.n_output_vars);
        rq.on_input = [&r, xn](int, const float* x) { r.x.emplace_back(x, x + xn); };
        rq.on_output = [&r, yn](int, int, int step_h, const DateTime&, const float* y) {
            r.y.emplace_back(y, y + yn);
            r.steps.push_back(step_h);
        };
        return rq;
    };
    Run host;
    ds.run(request(host));
    const std::string tag = std::string(prec == Precision::FP32 ? "fp32" : "fp16") + " ";
    auto same = [&](const Run& r, const std::string& what) {
        bool okx = r.x.size() == host.x.size(), oky = r.y.size() == host.y.size() && r.steps == host.steps;
        for (size_t i = 0; okx && i < r.x.size(); ++i) okx = r.x[i] == host.x[i];
        for (size_t i = 0; oky && i < r.y.size(); ++i) oky = r.y[i] == host.y[i];
        expect(okx, tag + what + ": assembled inputs bitwise equal to the host loop's");
        expect(oky, tag + what + ": outputs and step hours bitwise equal to the host loop's");
        std::printf("  device loop %s%s: inputs %s, outputs %s\n", tag.c_str(), what.c_str(), okx ? "bitwise" : "DIFFER",
                    oky ? "bitwise" : "DIFFER");
    };
    const int dev = cuda::as_cuda(*model)->device();
    const size_t out_floats = size_t(cfg.n_output_steps * Gm * cfg.n_output_vars);
    for (int variant = 0; variant < 4; ++variant) {
        cuda::DeviceLoopOptions o;
        o.hold_output = variant % 2;                    // without / with the hold buffer
        o.ring_states = variant < 2 ? 2 : 1;
        o.staging_slots = 3;
        // chunks that do not divide a state (nor the window): 2/5 of a state + 13 floats, >= 1024 floats
        o.staging_bytes = 4 * std::max<size_t>(1024, out_floats / size_t(cfg.n_output_steps) * 2 / 5 + 13);
        o.copy_threads = 3;
        const std::string what = std::string(o.hold_output ? "hold" : "no hold") + ", ring " + std::to_string(o.ring_states);
        {  // states from the host
            cuda::DeviceDownscaler dl(ds, -1, o);
            Run r;
            dl.run(request(r));
            same(r, what + ", host feed");
        }
        // states from device buffers, as the device rollout hands them on: on the downscaler's device and,
        // with two GPUs, on another one (cudaMemcpyPeerAsync, the --downscaler-device case)
        int n_dev = 1;
        CK(cudaGetDeviceCount(&n_dev));
        std::vector<int> producers = {dev};
        if (n_dev > 1) producers.push_back((dev + 1) % n_dev);
        for (int pd : producers) {
            cuda::DeviceDownscaler dl(ds, pd, o);
            CK(cudaSetDevice(pd));
            float* dy[2] = {nullptr, nullptr};
            cudaStream_t ps = nullptr;
            cudaEvent_t ready[2], back[2] = {nullptr, nullptr};
            const size_t bytes = size_t(Gm * V_fout) * sizeof(float);
            for (int i = 0; i < 2; ++i) {
                CK(cudaMalloc(reinterpret_cast<void**>(&dy[i]), bytes));
                CK(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
            }
            CK(cudaStreamCreateWithFlags(&ps, cudaStreamNonBlocking));
            Run r;
            std::exception_ptr err;
            std::thread consumer([&] {
                try {
                    dl.run(request(r));
                } catch (...) {
                    err = std::current_exception();
                    dl.cancel();
                }
            });
            for (int s = 0; s < N; ++s) {
                const int b = s % 2;
                if (back[b]) CK(cudaStreamWaitEvent(ps, back[b], 0));  // the loop has copied state s - 2 out
                CK(cudaMemcpyAsync(dy[b], ys[size_t(s)].data(), bytes, cudaMemcpyHostToDevice, ps));
                CK(cudaEventRecord(ready[b], ps));
                back[b] = dl.push_device(s, dy[b], ready[b]);
            }
            consumer.join();
            CK(cudaStreamSynchronize(ps));
            for (int i = 0; i < 2; ++i) {
                cudaFree(dy[i]);
                cudaEventDestroy(ready[i]);
            }
            cudaStreamDestroy(ps);
            CK(cudaSetDevice(dev));
            if (err) std::rethrow_exception(err);
            same(r, what + ", device feed (" + dl.feed_path() + ")");
        }
    }
}
#endif

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

    // ---- 6. the device loop (CUDA builds, VARDA_TEST_TD_MODEL)
#ifdef VARDA_WITH_CUDA
    {
        const char* td = std::getenv("VARDA_TEST_TD_MODEL");
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess) n_dev = 0;
        if (!td || !*td) {
            std::printf("note: VARDA_TEST_TD_MODEL not set: device loop test skipped\n");
        } else if (n_dev == 0) {
            std::printf("note: no CUDA device: device loop test skipped\n");
        } else {
            for (Precision p : {Precision::FP32, Precision::FP16}) {
                try {
                    test_device_loop(td, p);
                } catch (const std::exception& e) {
                    expect(false, std::string("device loop test threw: ") + e.what());
                }
            }
        }
    }
#endif

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_downscale: all checks passed\n");
    return 0;
}
