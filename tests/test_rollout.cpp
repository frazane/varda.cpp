// Rollout state update and dates (src/varda/rollout.h, docs/rollout-engine.md), on tiny synthetic
// windows whose values differ between slots, so that a value taken from the wrong slot shows:
//  1. T = 2..5, both constant sources, three consecutive updates, each bitwise equal to the literal
//     upstream definition executed on whole slots: anemoi-inference 9505dd0's advance_map
//     (inin = [(i, i-m) for i in range(m, n)], in order; outin = [(m-i-1, n-i-1)], m = 1, n = T) for
//     the default ConstantSource::Newest, and eaeb36fa's roll(-1) for ConstantSource::Oldest; plus
//     explicit slot-by-slot expectations: slot t <- old slot t+1, prognostic inputs from their (permuted)
//     outputs, forcings, and constants that keep the NEWEST slot's value (default) or take the OLDEST
//     slot's (roll); with constants equal in all slots, both semantics give the same window;
//  2. the default is ConstantSource::Newest;
//  3. invalid index sets (channel uncovered, listed twice, out of range) and a config without a
//     `forcings` section are rejected with an error;
//  4. valid dates: date0 + (s + 1) * timestep (1 h and 6 h, across a day, a month and a leap day);
//  5. if the converted models are present ($VARDA_SCRATCH/converted/<model>/model.safetensors):
//     StateUpdate::from_config on the real configs (counts, and every input covered exactly once).
#include <algorithm>
#include <cstdio>
#include <cstring>
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

bool rep_print_once(int T, ConstantSource cs) {
    static std::vector<std::pair<int, int>> seen;
    const std::pair<int, int> k{T, int(cs)};
    for (auto& p : seen) if (p == k) return false;
    seen.push_back(k);
    return true;
}

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

// The two upstream definitions, executed literally on whole slots (independent of StateUpdate):
//  - 9505dd0 (Metadata.advance_map, m = multi_step_output = 1, n = T):
//      inin  = [(i, i - m) for i in range(m, n)]   ->  window[:, new] = window[:, old], in list order
//      outin = [(m - i - 1, n - i - 1) for i in range(min(n, m))]   ->  newest slot: prognostic <- y, forcings
//  - eaeb36fa: window = roll(window, -1, dim=1), then the same newest-slot writes.
std::vector<float> reference_update(std::vector<float> w, int T, int64_t G, int V_in, int V_out,
                                    const std::vector<int>& prog_in, const std::vector<int>& prog_out,
                                    const std::vector<int>& dyn_in, const std::vector<float>& y,
                                    const std::vector<float>& f, ConstantSource cs, std::string* map_text) {
    const int64_t slot = G * V_in;
    const int n = T, m = 1;
    std::vector<std::pair<int, int>> inin, outin;
    for (int i = m; i < n; ++i) inin.push_back({i, i - m});
    for (int i = 0; i < std::min(n, m); ++i) outin.push_back({m - i - 1, n - i - 1});
    if (map_text) {
        *map_text = "inin";
        for (auto& p : inin) *map_text += " (" + std::to_string(p.first) + "->" + std::to_string(p.second) + ")";
        *map_text += " outin";
        for (auto& p : outin) *map_text += " (y" + std::to_string(p.first) + "->" + std::to_string(p.second) + ")";
    }
    if (cs == ConstantSource::Newest) {
        for (auto& p : inin)  // sequential, in place, like the torch assignments
            std::copy(w.begin() + p.first * slot, w.begin() + (p.first + 1) * slot, w.begin() + p.second * slot);
    } else {
        std::vector<float> r(w.size());
        for (int t = 0; t < T; ++t)  // roll(-1): r[t] = w[(t + 1) % T]
            std::copy(w.begin() + ((t + 1) % T) * slot, w.begin() + ((t + 1) % T + 1) * slot, r.begin() + t * slot);
        w = r;
    }
    for (auto& p : outin) {
        float* nw = &w[p.second * slot];
        for (int64_t g = 0; g < G; ++g) {
            for (size_t i = 0; i < prog_in.size(); ++i) nw[g * V_in + prog_in[i]] = y[g * V_out + prog_out[i]];
            for (size_t k = 0; k < dyn_in.size(); ++k) nw[g * V_in + dyn_in[k]] = f[g * dyn_in.size() + k];
        }
    }
    return w;
}

void check_update(int T, ConstantSource cs) {
    const int64_t G = 3;
    const int V_in = 7, V_out = 5;
    // channels: 0,3,6 prognostic (from outputs 2,0,4: permuted), 1,4 dynamic forcings (columns 0,1), 2,5 constant
    const std::vector<int> prog_in = {0, 3, 6}, prog_out = {2, 0, 4}, dyn_in = {1, 4}, const_in = {2, 5};
    StateUpdate u = StateUpdate::make(T, G, V_in, V_out, prog_in, prog_out, dyn_in, const_in, cs);
    expect(u.F == 2, "F");
    std::vector<float> w(size_t(T) * G * V_in), y(G * V_out), f(G * 2);
    for (int t = 0; t < T; ++t)
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < V_in; ++c) w[(t * G + g) * V_in + c] = wval(t, g, c);
    for (int64_t g = 0; g < G; ++g)
        for (int o = 0; o < V_out; ++o) y[g * V_out + o] = yval(g, o);
    for (int64_t g = 0; g < G; ++g)
        for (int k = 0; k < 2; ++k) f[g * 2 + k] = fval(g, k);

    const bool newest_cs = cs == ConstantSource::Newest;
    const std::string tag = "T=" + std::to_string(T) + (newest_cs ? " in place: " : " roll: ");
    std::string map_text;
    std::vector<float> want = w;
    for (int rep = 1; rep <= 3; ++rep) {  // three consecutive updates, each against the literal reference
        want = reference_update(want, T, G, V_in, V_out, prog_in, prog_out, dyn_in, y, f, cs, &map_text);
        u.apply(w.data(), y.data(), f.data());
        expect(std::memcmp(w.data(), want.data(), w.size() * 4) == 0,
               tag + "update " + std::to_string(rep) + " bitwise equal to the literal upstream definition");
    }
    if (rep_print_once(T, cs))
        std::printf("T=%d %s: %s\n", T, to_string(cs),
                    newest_cs ? ("advance_map " + map_text).c_str() : "roll(-1), then outin");

    // Explicit expectations after ONE update (fresh window): which slot each value comes from.
    for (int t = 0; t < T; ++t)
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < V_in; ++c) w[(t * G + g) * V_in + c] = wval(t, g, c);
    u.apply(w.data(), y.data(), f.data());
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
            if (newest_cs) {
                expect(nw[c] == wval(n, g, c), tag + "constant input keeps the NEWEST slot's own value");
                expect(nw[c] != wval(0, g, c), tag + "constant input not from the oldest slot");
            } else {
                expect(nw[c] == wval(0, g, c), tag + "constant input inherited from the OLDEST slot");
                expect(nw[c] != wval(n, g, c), tag + "constant input not the newest slot's own old value");
            }
        }
    }
    // A second update. In place: the newest slot's constants never change; roll: slot 0 now holds the
    // original slot 1, so that is what comes round (for truly constant inputs all slots agree).
    u.apply(w.data(), y.data(), f.data());
    for (int64_t g = 0; g < G; ++g)
        for (int c : const_in) {
            const float want2 = newest_cs ? wval(n, g, c) : wval(1, g, c);
            expect(w[(n * G + g) * V_in + c] == want2, tag + "second update: constant source");
        }
}

// With constants equal in every slot (real inputs), both semantics give the same window.
void check_equal_constants(int T) {
    const int64_t G = 5;
    const int V_in = 7, V_out = 5;
    const std::vector<int> prog_in = {0, 3, 6}, prog_out = {2, 0, 4}, dyn_in = {1, 4}, const_in = {2, 5};
    StateUpdate a = StateUpdate::make(T, G, V_in, V_out, prog_in, prog_out, dyn_in, const_in, ConstantSource::Newest);
    StateUpdate b = StateUpdate::make(T, G, V_in, V_out, prog_in, prog_out, dyn_in, const_in, ConstantSource::Oldest);
    std::vector<float> w(size_t(T) * G * V_in), y(G * V_out), f(G * 2);
    for (int t = 0; t < T; ++t)
        for (int64_t g = 0; g < G; ++g)
            for (int c = 0; c < V_in; ++c) w[(t * G + g) * V_in + c] = (c == 2 || c == 5) ? wval(0, g, c) : wval(t, g, c);
    for (int64_t g = 0; g < G; ++g)
        for (int o = 0; o < V_out; ++o) y[g * V_out + o] = yval(g, o);
    for (int64_t g = 0; g < G; ++g)
        for (int k = 0; k < 2; ++k) f[g * 2 + k] = fval(g, k);
    std::vector<float> wb = w;
    for (int rep = 0; rep < 3; ++rep) {
        a.apply(w.data(), y.data(), f.data());
        b.apply(wb.data(), y.data(), f.data());
    }
    expect(std::memcmp(w.data(), wb.data(), w.size() * 4) == 0,
           "T=" + std::to_string(T) + ": equal constants in all slots -> in place == roll, bitwise");
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
    if (!(e && *e)) {
        std::printf("skip the converted models: VARDA_SCRATCH is not set (source tools/env.sh)\n");
        return;
    }
    const std::string root = e;
    for (const char* m : {"dev", "varda"}) {
        const std::string p = root + "/converted/" + m + "/model.safetensors";
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
    for (int T = 2; T <= 5; ++T) {
        check_update(T, ConstantSource::Newest);
        check_update(T, ConstantSource::Oldest);
        check_equal_constants(T);
    }
    {  // the default is in place (anemoi-inference 9505dd0)
        StateUpdate d = StateUpdate::make(2, 1, 3, 1, {0}, {0}, {1}, {2});
        expect(d.constants == ConstantSource::Newest, "default constant source is in place (9505dd0)");
    }
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
