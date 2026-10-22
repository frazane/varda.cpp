// Rollout state update and dates (src/varda/rollout.h, docs/rollout-engine.md), on tiny synthetic
// windows whose values differ between slots, so that a value taken from the wrong slot shows:
//  1. T = 2: roll, prognostic feedback (permuted output indices), forcings, and constant inputs
//     inherited from the OLDEST slot (anemoi-inference: the roll brings slot 0 round);
//  2. T = 3: the same with a deeper window;
//  3. invalid index sets (channel uncovered, listed twice, out of range) and a config without a
//     `forcings` section are rejected with an error;
//  4. valid dates: date0 + (s + 1) * timestep (1 h and 6 h, across a day, a month and a leap day);
//  5. if the converted models are present ($VARDA_SCRATCH/converted-new/<model>/model.safetensors):
//     StateUpdate::from_config on the real configs (counts, and every input covered exactly once).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "varda/config.h"
#include "varda/format.h"
#include "varda/rollout.h"

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
bool throws(F f, std::string* msg = nullptr) {
    try {
        f();
    } catch (const std::exception& e) {
        if (msg) *msg = e.what();
        return true;
    }
    return false;
}

// Distinct value per (slot, point, channel).
float wval(int t, int64_t g, int c) { return 1000.0f * (t + 1) + 10.0f * g + c + 0.125f; }
float yval(int64_t g, int o) { return -(100.0f * g + o) - 0.5f; }
float fval(int64_t g, int k) { return 0.25f + g + 0.0625f * k; }

void check_update(int T) {
    const int64_t G = 3;
    const int V_in = 7, V_out = 5;
    // channels: 0,3,6 prognostic (from outputs 2,0,4: permuted), 1,4 dynamic forcings (columns 0,1), 2,5 constant
    const std::vector<int> prog_in = {0, 3, 6}, prog_out = {2, 0, 4}, dyn_in = {1, 4}, const_in = {2, 5};
    StateUpdate u = StateUpdate::make(T, G, V_in, V_out, prog_in, prog_out, dyn_in, const_in);
    expect(u.F == 2, "F");
    std::vector<float> w(size_t(T) * G * V_in), y(G * V_out), f(G * 2);
    for (int t = 0; t < T; ++t)
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < V_in; ++c) w[(t * G + g) * V_in + c] = wval(t, g, c);
    for (int64_t g = 0; g < G; ++g)
        for (int o = 0; o < V_out; ++o) y[g * V_out + o] = yval(g, o);
    for (int64_t g = 0; g < G; ++g)
        for (int k = 0; k < 2; ++k) f[g * 2 + k] = fval(g, k);
    u.apply(w.data(), y.data(), f.data());

    const std::string tag = "T=" + std::to_string(T) + ": ";
    for (int t = 0; t + 1 < T; ++t)
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < V_in; ++c)
                expect(w[(t * G + g) * V_in + c] == wval(t + 1, g, c),
                       tag + "slot " + std::to_string(t) + " is the old slot " + std::to_string(t + 1));
    const int n = T - 1;
    for (int64_t g = 0; g < G; ++g) {
        const float* nw = &w[(n * G + g) * V_in];
        for (size_t i = 0; i < prog_in.size(); ++i)
            expect(nw[prog_in[i]] == yval(g, prog_out[i]), tag + "prognostic input from its output");
        for (size_t k = 0; k < dyn_in.size(); ++k)
            expect(nw[dyn_in[k]] == fval(g, int(k)), tag + "dynamic forcing column");
        for (int c : const_in) {
            expect(nw[c] == wval(0, g, c), tag + "constant input inherited from the OLDEST slot");
            expect(nw[c] != wval(n, g, c), tag + "constant input not the newest slot's own old value");
        }
    }
    // A second update: the constants keep rolling round; slot 0 now holds the original slot 1,
    // so that is what the newest slot gets (anemoi's roll; for truly constant inputs all slots agree).
    u.apply(w.data(), y.data(), f.data());
    for (int64_t g = 0; g < G; ++g)
        for (int c : const_in) {
            const float want = wval(1, g, c);
            expect(w[(n * G + g) * V_in + c] == want, tag + "second update: constant from the oldest slot");
        }
}

void check_errors() {
    std::string msg;
    expect(throws([] { StateUpdate::make(2, 1, 4, 2, {0}, {0}, {1}, {2}); }, &msg), "uncovered channel rejected");
    expect(msg.find("channel 3") != std::string::npos, "message names the uncovered channel: " + msg);
    expect(throws([] { StateUpdate::make(2, 1, 3, 2, {0}, {0}, {0}, {1, 2}); }), "channel listed twice rejected");
    expect(throws([] { StateUpdate::make(2, 1, 3, 2, {0}, {5}, {1}, {2}); }), "output index out of range rejected");
    expect(throws([] { StateUpdate::make(2, 1, 3, 2, {0, 1}, {0}, {}, {2}); }), "length mismatch rejected");
    Config c;
    c.n_input_steps = 2;
    c.n_input_vars = 3;
    c.n_output_vars = 2;
    expect(throws([&] { StateUpdate::from_config(c); }, &msg), "config without forcings rejected");
    expect(msg.find("forcings") != std::string::npos, "message mentions the forcings section: " + msg);
}

void check_dates() {
    const DateTime d0 = parse_iso8601_utc("2024-06-01T00:00:00");
    expect(format_iso8601_utc(valid_date(d0, 0, 1)) == "2024-06-01T01:00:00", "1 h, step 0");
    expect(format_iso8601_utc(valid_date(d0, 22, 1)) == "2024-06-01T23:00:00", "1 h, step 22");
    expect(format_iso8601_utc(valid_date(d0, 23, 1)) == "2024-06-02T00:00:00", "1 h, step 23 crosses the day");
    expect(format_iso8601_utc(valid_date(d0, 0, 6)) == "2024-06-01T06:00:00", "6 h, step 0");
    expect(format_iso8601_utc(valid_date(d0, 3, 6)) == "2024-06-02T00:00:00", "6 h, step 3");
    expect(format_iso8601_utc(valid_date(d0, 22, 6)) == "2024-06-06T18:00:00", "6 h, step 22");
    const DateTime l = parse_iso8601_utc("2024-02-28T18:00:00");
    expect(format_iso8601_utc(valid_date(l, 0, 6)) == "2024-02-29T00:00:00", "leap day");
    expect(format_iso8601_utc(valid_date(l, 4, 6)) == "2024-03-01T00:00:00", "out of February");
}

void check_real_configs() {
    const char* e = std::getenv("VARDA_SCRATCH");
    const std::string root = e && *e ? e : "$VARDA_SCRATCH";
    for (const char* m : {"dev", "varda"}) {
        const std::string p = root + "/converted-new/" + m + "/model.safetensors";
        if (!std::ifstream(p).good()) {
            std::printf("skip %s: %s not found\n", m, p.c_str());
            continue;
        }
        SafeTensorsFile f(p);
        const Config c = parse_config(f.metadata("config"));
        if (!c.has_forcings) {
            std::printf("skip %s: config has no forcings section\n", m);
            continue;
        }
        StateUpdate u = StateUpdate::from_config(c);
        int np = 0, nf = 0, nk = 0;
        for (int8_t k : u.kind) (k == StateUpdate::Prognostic ? np : k == StateUpdate::Forcing ? nf : nk)++;
        std::printf("%s: %d inputs = %d prognostic + %d dynamic forcings + %d constant; timestep %d h\n", m, c.n_input_vars,
                    np, nf, nk, c.timestep_hours);
        expect(np == int(c.residual_input.size()) && nf == 5 && nk == 8, std::string(m) + ": channel counts");
        expect(c.timestep_hours == (std::string(m) == "varda" ? 6 : 1), std::string(m) + ": timestep");
    }
}

}  // namespace

int main() {
    check_update(2);
    check_update(3);
    check_errors();
    check_dates();
    check_real_configs();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_rollout: all checks passed\n");
    return 0;
}
