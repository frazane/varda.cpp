// Autoregressive rollout (src/varda/rollout.h, src/varda/cuda/rollout_cuda.h, docs/rollout-engine.md).
//
//   varda-rollout --model M.safetensors --input IN [--key K] [--date ISO] [--steps N]
//                 [--device cpu|cuda] [--precision fp32|fp16] [--threads N] [--cuda-device I]
//                 [--engine host|device] [--no-cuda-graph]
//                 [--forcings computed|replay] [--replay-from ROLLOUT]
//                 [--compare REF]... [--tolerances FILE] [--check-forcings]
//                 [--verify-replay] [--verify-host] [--verify-constants] [--constants newest|oldest]
//                 [--bench R] [--forward-iters N]
//                 [--json OUT.json] [--out Y.safetensors]
//   varda-rollout --judge OLD.json [--tolerances FILE] [--json NEW.json]     (re-judge an earlier report)
//
// IN is a rollout file (python/export_rollout.py / python/reference/export_rollout.py: x0 [2, G, V_in]
// and its start_date) or a golden file (`in.x` [1, T, G, V_in]; --date required). --date is the date of
// the newest input slot. --steps defaults to the steps of the first --compare reference.
//
// Engines: `host` = varda::Rollout (Model::predict_step, any device); `device` = varda::cuda::DeviceRollout
// (CUDA only, default there; one CUDA graph per step unless --no-cuda-graph).
// Forcings: `computed` (default; compute_forcings at each valid date) or `replay` (the forcings array of
// --replay-from, default the input file if it is a rollout file, else the first --compare reference).
//
// --compare REF (file, or a directory holding rollout.safetensors): per step s, the contract metric
// (docs/format.md §2) of y (the step's physical output), y_norm (y * mul_out + add_out, float64), window
// (the window step s sees) and window_norm (docs/rollout-engine.md §4). The table of the
// rollout tolerance file (tolerances/rollout/<model>.json, found by checkpoint uuid, or --tolerances) is
// chosen from the reference tag and this run's precision:
//     cpu-fp32-*   fp32: fast-fp32-vs-cpu     fp16: fast-fp16-vs-cpu
//     cuda-fp16-*  fp16: fast-fp16-vs-ref-fp16
//     other        reported only
// fp32 tables: every step within its tolerance. fp16 tables (fp16_criterion of the file): step 0 within the
// tolerance; steps >= 1 within fp16_allowance x the tolerance; per key, the mean over steps within
// fp16_mean_tolerance. Entries above the bare per-step tolerance are counted.
//
// Checks (each reported, exit 1 on failure):
//   dates            valid date of output s == the reference's recorded dates[s];
//   --check-forcings compute_forcings(valid date s) bitwise == the replay source's forcings[s], every s;
//                    with --compare, also the newest-slot forcing columns of the window each step sees;
//   --verify-replay  a second rollout with the other forcing mode: y and windows bitwise equal;
//   --verify-host    (device engine) the host-level Rollout on the same model: y and windows bitwise equal;
//   --verify-constants a second rollout with the other constant source: y and windows bitwise equal (holds
//                    when the constant inputs are equal in every slot, reported as constants_across_slots).
// --constants: where the newest slot's constant-in-time inputs come from (src/varda/rollout.h): `newest`
// (default; kept in place, anemoi-inference 9505dd0's advance_map) or `oldest` (roll(-1), eaeb36fa).
// --bench R: after the checked run, R more rollouts of N steps (no window copies, no compare): per-step
// GPU time (CUDA events around each step graph), wall time per step; plus --forward-iters replays of the
// forward-only graph (normalise -> forward -> denormalise) for the like-for-like one-step time.
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "common.h"
#include "varda/forcings.h"
#include "varda/format.h"
#include "varda/rollout.h"
#include "varda/varda.h"

#ifdef VARDA_WITH_CUDA
#include <cuda_runtime_api.h>

#include "varda/cuda/cuda_backend.h"
#include "varda/cuda/rollout_cuda.h"
#endif

using app::json;
using varda::DateTime;

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-rollout --model M --input IN [--key K] [--date ISO] [--steps N] [--device cpu|cuda]\n"
                 "         [--precision fp32|fp16] [--threads N] [--cuda-device I] [--engine host|device]\n"
                 "         [--no-cuda-graph] [--forcings computed|replay] [--replay-from ROLLOUT] [--compare REF]...\n"
                 "         [--tolerances FILE] [--check-forcings] [--verify-replay] [--verify-host] [--verify-constants]\n"
                 "         [--constants newest|oldest] [--bench R]\n"
                 "         [--forward-iters N] [--json OUT] [--out Y.safetensors]\n"
                 "       varda-rollout --judge OLD.json [--tolerances FILE] [--json NEW.json]\n");
    std::exit(2);
}

bool is_dir(const std::string& p) {
    std::ifstream f(p + "/rollout.safetensors");
    return f.good();
}

// A reference (or input) rollout file: docs/rollout-engine.md §"Files".
struct RefRollout {
    std::unique_ptr<varda::SafeTensorsFile> file;
    std::string path, tag, start_date;
    int steps = 0;
    int64_t G = 0;
    int V_in = 0, V_out = 0, F = 0;
    std::vector<std::string> dates, forcing_vars, in_vars, out_vars;
    std::string uuid;
    json run;
    const float *x0 = nullptr, *y = nullptr, *window = nullptr, *forcings = nullptr;
    const float *mul_in = nullptr, *add_in = nullptr, *mul_out = nullptr, *add_out = nullptr;

    explicit RefRollout(std::string p) {
        if (is_dir(p)) p += "/rollout.safetensors";
        path = p;
        file.reset(new varda::SafeTensorsFile(p));
        const auto& md = file->metadata();
        if (!md.count("varda_rollout") || md.at("varda_rollout") != "1")
            throw std::runtime_error(p + ": not a varda rollout file (metadata varda_rollout != 1)");
        tag = md.at("tag");
        start_date = md.at("start_date");
        steps = std::stoi(md.at("steps"));
        dates = json::parse(md.at("dates")).get<std::vector<std::string>>();
        forcing_vars = json::parse(md.at("forcing_vars")).get<std::vector<std::string>>();
        in_vars = json::parse(md.at("in_vars")).get<std::vector<std::string>>();
        out_vars = json::parse(md.at("out_vars")).get<std::vector<std::string>>();
        if (md.count("checkpoint_uuid")) uuid = md.at("checkpoint_uuid");
        if (md.count("run")) run = json::parse(md.at("run"));
        const auto& tx = file->peek("x0");
        const auto& ty = file->peek("y");
        G = tx.dim(1);
        V_in = int(tx.dim(2));
        V_out = int(ty.dim(2));
        F = int(forcing_vars.size());
        auto f32 = [&](const char* n, std::vector<int64_t> shape) {
            const auto& t = file->peek(n);
            if (t.dtype != varda::DType::F32 || t.shape != shape)
                throw std::runtime_error(path + ": " + n + " is " + t.shape_str() + ", expected " + app::shape_str(shape));
            return t.as<float>();
        };
        x0 = f32("x0", {2, G, V_in});
        y = f32("y", {steps, G, V_out});
        window = f32("window", {steps, 2, G, V_in});
        forcings = f32("forcings", {steps - 1, G, F});
        mul_in = f32("mul_in", {V_in});
        add_in = f32("add_in", {V_in});
        mul_out = f32("mul_out", {V_out});
        add_out = f32("add_out", {V_out});
    }

    // Refuses a reference of another model: same variables, grid size and (if recorded) checkpoint uuid.
    void check_pairs(const varda::Config& c) const {
        if (!uuid.empty() && uuid != c.uuid)
            throw std::runtime_error(path + ": checkpoint uuid " + uuid + " is not the model's " + c.uuid);
        if (in_vars != c.var_input || out_vars != c.var_output || G != c.n_data_nodes)
            throw std::runtime_error(path + ": variables or grid differ from the model's: another checkpoint");
    }
};

struct Metric {
    double max_abs = 0, absmax = 0, rel = 0;
    int64_t nan = 0;
};

Metric finish(double max_abs, double absmax, int64_t nan) {
    Metric m;
    m.max_abs = max_abs;
    m.absmax = absmax;
    m.nan = nan;
    m.rel = absmax > 0 ? max_abs / absmax : (max_abs > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    if (nan) m.rel = std::numeric_limits<double>::infinity();
    return m;
}

// Contract metric of a vs r (n elements, V channels per row), raw and normalised (v * mul[c] + add[c]),
// in float64: one pass for both.
std::pair<Metric, Metric> compare2(const float* a, const float* r, int64_t n, int V, const float* mul,
                                   const float* add) {
    double ma = 0, ra = 0, mn = 0, rn = 0;
    int64_t nan = 0;
#pragma omp parallel for reduction(max : ma, ra, mn, rn) reduction(+ : nan) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double av = a[i], rv = r[i];
        if (std::isnan(av)) { ++nan; continue; }
        const int c = int(i % V);
        const double m = mul[c], d = add[c];
        const double an = av * m + d, rnv = rv * m + d;
        ma = std::max(ma, std::fabs(av - rv));
        ra = std::max(ra, std::fabs(rv));
        mn = std::max(mn, std::fabs(an - rnv));
        rn = std::max(rn, std::fabs(rnv));
    }
    return {finish(ma, ra, nan), finish(mn, rn, nan)};
}

json metric_json(const Metric& m) {
    return {{"rel_absmax", m.rel}, {"max_abs", m.max_abs}, {"absmax", m.absmax}, {"bitwise_equal", m.max_abs == 0 && !m.nan}};
}

// Bitwise comparison: number of differing float elements and max abs difference.
std::pair<int64_t, double> bitdiff(const float* a, const float* b, int64_t n) {
    int64_t bad = 0;
    double mx = 0;
#pragma omp parallel for reduction(+ : bad) reduction(max : mx) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, a + i, 4);
        std::memcpy(&ub, b + i, 4);
        if (ua != ub) {
            ++bad;
            mx = std::max(mx, std::fabs(double(a[i]) - double(b[i])));
        }
    }
    return {bad, mx};
}

// 64-bit hash of a float array's bytes, independent of the thread count (fixed 256 chunks).
uint64_t hash_bytes(const float* p, int64_t n) {
    const int K = 256;
    std::vector<uint64_t> h(K);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < K; ++k) {
        const int64_t lo = n * k / K, hi = n * (k + 1) / K;
        uint64_t x = 1469598103934665603ull ^ uint64_t(k);
        for (int64_t i = lo; i < hi; ++i) {
            uint32_t u;
            std::memcpy(&u, p + i, 4);
            x = (x ^ u) * 1099511628211ull;
        }
        h[k] = x;
    }
    uint64_t r = 0;
    for (int k = 0; k < K; ++k) r = (r ^ h[k]) * 1099511628211ull + 0x9e3779b97f4a7c15ull;
    return r;
}

// Tolerance file tolerances/rollout/*.json whose checkpoint.uuid is the model's.
json load_rollout_tolerances(const std::string& explicit_path, const std::string& uuid, std::string* used) {
    std::vector<std::string> cands;
    if (!explicit_path.empty()) {
        cands.push_back(explicit_path);
    } else {
        const std::string dir = std::string(VARDA_SOURCE_DIR) + "/tolerances/rollout";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                std::string n = e->d_name;
                if (n.size() > 5 && n.substr(n.size() - 5) == ".json") cands.push_back(dir + "/" + n);
            }
            closedir(d);
        }
    }
    for (const auto& c : cands) {
        std::ifstream f(c);
        json t = json::parse(f);
        const std::string tu = t.contains("checkpoint") ? t["checkpoint"].value("uuid", "") : "";
        if (tu == uuid) {
            *used = c;
            return t;
        }
        if (!explicit_path.empty()) throw std::runtime_error(c + ": checkpoint uuid " + tu + " is not the model's " + uuid);
    }
    return json();  // none: comparisons are reported without a verdict (and the run fails)
}

const char* KEYS[4] = {"y", "y_norm", "window", "window_norm"};

// One --compare reference with its per-step metrics.
struct Comparison {
    std::unique_ptr<RefRollout> ref;  // null when re-judging a report (--judge)
    std::string path, tag;
    std::string table;  // tolerance table, or "" (report only)
    std::vector<std::array<Metric, 4>> m;
    int64_t forcing_cols_bad = 0;  // newest-slot forcing columns of the window != ref forcings (s >= 1)
    int const_unequal_steps = 0;   // reference windows whose constant inputs differ between slots
    double forcing_cols_maxabs = 0;
};

std::string pick_table(const std::string& tag, varda::Precision p) {
    const bool fp16 = p == varda::Precision::FP16;
    if (tag.rfind("cpu-fp32", 0) == 0) return fp16 ? "fast-fp16-vs-cpu" : "fast-fp32-vs-cpu";
    if (tag.rfind("cuda-fp16", 0) == 0 && fp16) return "fast-fp16-vs-ref-fp16";
    return "";
}

// Judges one comparison (docs at the top); prints the per-step table; returns the report entry.
json judge(const Comparison& c, const json& tols, int n, bool* ok) {
    json out;
    out["reference"] = c.path;
    out["tag"] = c.tag;
    out["table"] = c.table.empty() ? json(nullptr) : json(c.table);
    json mj;
    for (int k = 0; k < 4; ++k) {
        json arr = json::array();
        for (int s = 0; s < n; ++s) arr.push_back(metric_json(c.m[s][k]));
        mj[KEYS[k]] = arr;
    }
    out["metrics"] = mj;
    const bool have = !c.table.empty() && tols.is_object() && tols.contains(c.table);
    std::vector<std::array<double, 4>> tol(n, {0, 0, 0, 0});
    if (have) {
        for (int k = 0; k < 4; ++k) {
            const auto& t = tols[c.table][KEYS[k]];
            if (int(t.size()) < n) throw std::runtime_error("tolerance table " + c.table + " has fewer steps than the run");
            for (int s = 0; s < n; ++s) tol[s][k] = t[s].get<double>();
        }
    }
    const bool fp16 = c.table.find("fp16") != std::string::npos;
    double allow = 1.0;
    std::array<double, 4> mean_tol = {0, 0, 0, 0};
    if (have && fp16) {
        allow = tols["fp16_allowance"][c.table].get<double>();
        for (int k = 0; k < 4; ++k) mean_tol[k] = tols["fp16_mean_tolerance"][c.table][KEYS[k]].get<double>();
    }
    std::printf("\n%s (%s)  [tolerance: %s]\n", c.tag.c_str(), c.path.c_str(),
                have ? c.table.c_str() : "none: reported only");
    std::printf("%4s", "step");
    for (int k = 0; k < 4; ++k) std::printf(" %13s %11s%s", (std::string(KEYS[k]) + " rel").c_str(), "max_abs", have ? "  x tol" : "");
    std::printf("\n");
    for (int s = 0; s < n; ++s) {
        std::printf("%4d", s);
        for (int k = 0; k < 4; ++k) {
            std::printf(" %13.3e %11.3e", c.m[s][k].rel, c.m[s][k].max_abs);
            if (have) {
                if (tol[s][k] > 0) std::printf(" %6.2f", c.m[s][k].rel / tol[s][k]);
                else std::printf(" %6s", "-");
            }
        }
        std::printf("\n");
    }
    std::array<double, 4> mean = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        for (int s = 0; s < n; ++s) mean[k] += c.m[s][k].rel;
        mean[k] /= n;
    }
    if (!have) {
        out["pass"] = nullptr;
        return out;
    }
    json viol = json::array();
    int above = 0;
    double worst = -1;
    int wk = 0, ws = 0;
    for (int k = 0; k < 4; ++k)
        for (int s = 0; s < n; ++s) {
            const double r = c.m[s][k].rel, t = tol[s][k];
            if (!(r <= t)) ++above;
            const double lim = t * (fp16 && s > 0 ? allow : 1.0);
            if (!(r <= lim)) viol.push_back({{"key", KEYS[k]}, {"step", s}, {"rel_absmax", r}, {"limit", lim}});
            if (t > 0 && r / t > worst) worst = r / t, wk = k, ws = s;
        }
    char line[512];
    int len = std::snprintf(line, sizeof line, "worst %.2f x per-step tolerance (%s step %d); %d/%d entries above it; allowance %g",
                            worst, KEYS[wk], ws, above, 4 * n, fp16 ? allow : 1.0);
    json means;
    if (fp16) {
        len += std::snprintf(line + len, sizeof line - len, "; mean over steps / mean tolerance:");
        for (int k = 0; k < 4; ++k) {
            len += std::snprintf(line + len, sizeof line - len, " %s %.2f", KEYS[k], mean[k] / mean_tol[k]);
            means[KEYS[k]] = {{"mean", mean[k]}, {"mean_tolerance", mean_tol[k]}};
            if (!(mean[k] <= mean_tol[k]))
                viol.push_back({{"key", std::string("mean ") + KEYS[k]}, {"rel_absmax", mean[k]}, {"limit", mean_tol[k]}});
        }
    }
    const bool pass = viol.empty();
    std::printf("%s -> %s\n", line, pass ? "PASS" : "FAIL");
    out["tolerance"] = tols[c.table];
    out["summary"] = line;
    out["worst_ratio"] = {{"ratio", worst}, {"key", KEYS[wk]}, {"step", ws}};
    out["entries_above_per_step_tolerance"] = above;
    if (fp16) out["fp16"] = {{"allowance", allow}, {"means", means}};
    out["violations"] = viol;
    out["pass"] = pass;
    if (!pass) *ok = false;
    return out;
}

// --judge OLD.json: re-applies the rollout tolerances to the per-step metrics of an earlier report (same table
// choice, same criterion), without running anything. Everything else of the old report is kept.
int rejudge(const std::string& path, const std::string& tol_path, const std::string& out_path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "error: cannot read %s\n", path.c_str());
        return 1;
    }
    json rep = json::parse(f);
    bool ok = true;
    try {
        const varda::Precision prec = rep.at("precision") == "fp16" ? varda::Precision::FP16 : varda::Precision::FP32;
        const int n = rep.at("steps").get<int>();
        std::string used;
        const json tols = load_rollout_tolerances(tol_path, rep.at("uuid").get<std::string>(), &used);
        std::printf("re-judging %s (%s %s, %d steps) with %s\n", path.c_str(), rep["device"].get<std::string>().c_str(),
                    rep["precision"].get<std::string>().c_str(), n, used.empty() ? "NO tolerance file" : used.c_str());
        if (tols.is_null()) ok = false;
        for (const char* k : {"verify_replay", "verify_host", "verify_constants"})
            if (rep.contains(k) && !rep[k].value("pass", false)) ok = false;
        if (rep.contains("forcings_check") && rep["forcings_check"].value("not_bitwise_equal", 1) != 0) ok = false;
        for (auto& d : rep.value("dates_check", json::array())) if (d.value("mismatches", 1) != 0) ok = false;
        json comps = json::array();
        for (auto& old : rep.at("comparisons")) {
            Comparison c;
            c.path = old.at("reference");
            c.tag = old.at("tag");
            c.table = pick_table(c.tag, prec);
            c.m.assign(n, {});
            for (int k = 0; k < 4; ++k)
                for (int s = 0; s < n; ++s) {
                    const json& e = old["metrics"][KEYS[k]][s];
                    Metric& m = c.m[s][k];
                    m.rel = e["rel_absmax"].is_number() ? e["rel_absmax"].get<double>() : std::numeric_limits<double>::infinity();
                    m.max_abs = e["max_abs"].get<double>();
                    m.absmax = e["absmax"].get<double>();
                }
            json j = judge(c, tols, n, &ok);
            if (old.contains("forcing_columns_vs_reference")) {
                j["forcing_columns_vs_reference"] = old["forcing_columns_vs_reference"];
                if (old["forcing_columns_vs_reference"].value("not_bitwise_equal", 1) != 0) ok = false;
            }
            comps.push_back(j);
        }
        rep["comparisons"] = comps;
        rep["tolerances"] = used;
        rep["rejudged_from"] = path;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        ok = false;
    }
    rep["pass"] = ok;
    if (!out_path.empty()) {
        const std::string tmp = out_path + ".tmp";
        std::ofstream(tmp) << rep.dump(1) << "\n";
        std::rename(tmp.c_str(), out_path.c_str());
    }
    std::printf("%s\n", ok ? "varda-rollout: OK" : "varda-rollout: FAILED");
    return ok ? 0 : 1;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, input_path, key, date_s, engine, forcing_mode = "computed", replay_from, tol_path, json_path,
        out_path, judge_path;
    std::vector<std::string> compare_paths;
    int steps = 0, bench = 0, forward_iters = 0;
    bool check_forcings = false, verify_replay = false, verify_host = false, verify_constants = false;
    varda::ConstantSource csrc = varda::ConstantSource::Newest;
    varda::Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--key") key = next();
        else if (a == "--date") date_s = next();
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--device") { std::string d = next(); if (d != "cpu" && d != "cuda") usage(); opts.device = d == "cuda" ? varda::Device::CUDA : varda::Device::CPU; }
        else if (a == "--precision") { std::string p = next(); if (p != "fp32" && p != "fp16") usage(); opts.precision = p == "fp16" ? varda::Precision::FP16 : varda::Precision::FP32; }
        else if (a == "--threads") opts.num_threads = std::stoi(next());
        else if (a == "--cuda-device") opts.cuda_device = std::stoi(next());
        else if (a == "--engine") engine = next();
        else if (a == "--no-cuda-graph") opts.cuda_graph = false;
        else if (a == "--forcings") forcing_mode = next();
        else if (a == "--replay-from") replay_from = next();
        else if (a == "--compare") compare_paths.push_back(next());
        else if (a == "--tolerances") tol_path = next();
        else if (a == "--check-forcings") check_forcings = true;
        else if (a == "--verify-replay") verify_replay = true;
        else if (a == "--verify-constants") verify_constants = true;
        else if (a == "--constants") {
            const std::string c = next();
            if (c != "newest" && c != "oldest") usage();
            csrc = c == "newest" ? varda::ConstantSource::Newest : varda::ConstantSource::Oldest;
        }
        else if (a == "--verify-host") verify_host = true;
        else if (a == "--bench") bench = std::stoi(next());
        else if (a == "--forward-iters") forward_iters = std::stoi(next());
        else if (a == "--json") json_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--judge") judge_path = next();
        else usage();
    }
    if (!judge_path.empty()) return rejudge(judge_path, tol_path, json_path);
    if (model_path.empty() || input_path.empty()) usage();
    if (forcing_mode != "computed" && forcing_mode != "replay") usage();
    if (engine.empty()) engine = opts.device == varda::Device::CUDA ? "device" : "host";
    if (engine != "host" && engine != "device") usage();
    if (engine == "device" && opts.device != varda::Device::CUDA) {
        std::fprintf(stderr, "error: --engine device needs --device cuda\n");
        return 2;
    }
    if (opts.num_threads > 0) omp_set_num_threads(opts.num_threads);

    json report;
    bool ok = true;
    try {
        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        const varda::Config& cfg = model->config();
        varda::check_rollout_model(*model);  // clear error without a forcings section or grid coordinates
        const int64_t G = cfg.n_data_nodes;
        const int T = cfg.n_input_steps, V_in = cfg.n_input_vars, V_out = cfg.n_output_vars;
        const int F = int(cfg.forcing_dynamic.size());
        const int64_t in_n = int64_t(T) * G * V_in, out_n = G * V_out, f_n = G * F;
        std::printf("model %s (uuid %s), %s %s, engine %s, cuda graph %s, threads %d, load %.2f s\n", model_path.c_str(),
                    cfg.uuid.c_str(), varda::to_string(opts.device).c_str(), varda::to_string(opts.precision).c_str(),
                    engine.c_str(), opts.device == varda::Device::CUDA ? (opts.cuda_graph ? "on" : "off") : "-",
                    omp_get_max_threads(), load_s);

        // ---- input window and date
        std::unique_ptr<RefRollout> in_roll;
        std::unique_ptr<varda::SafeTensorsFile> in_file;
        const float* x0 = nullptr;
        {
            varda::SafeTensorsFile probe(input_path + (is_dir(input_path) ? "/rollout.safetensors" : ""));
            const bool rollout_file = probe.metadata().count("varda_rollout") != 0;
            if (rollout_file) {
                in_roll.reset(new RefRollout(input_path));
                in_roll->check_pairs(cfg);
                x0 = in_roll->x0;
                if (date_s.empty()) date_s = in_roll->start_date;
            }
        }
        if (!x0) {
            in_file.reset(new varda::SafeTensorsFile(input_path));
            const auto& t = in_file->peek(key.empty() ? "in.x" : key);
            if (t.dtype != varda::DType::F32 || t.numel() != in_n)
                throw std::runtime_error(input_path + ": input tensor " + t.shape_str() + " is not (1, T, G, V_in) F32");
            x0 = t.as<float>();
            if (date_s.empty()) throw std::runtime_error("--date is required with a non-rollout input file");
        }
        const DateTime date0 = varda::parse_iso8601_utc(date_s);

        // ---- references
        std::vector<Comparison> cmp;
        for (const auto& p : compare_paths) {
            Comparison c;
            c.ref.reset(new RefRollout(p));
            c.ref->check_pairs(cfg);
            c.path = c.ref->path;
            c.tag = c.ref->tag;
            c.table = pick_table(c.ref->tag, opts.precision);
            cmp.push_back(std::move(c));
        }
        if (steps <= 0) steps = cmp.empty() ? (in_roll ? in_roll->steps : 1) : cmp[0].ref->steps;
        for (auto& c : cmp) {
            if (c.ref->steps < steps) throw std::runtime_error(c.ref->path + " has only " + std::to_string(c.ref->steps) + " steps");
            if (std::memcmp(c.ref->x0, x0, in_n * sizeof(float)) != 0)
                std::printf("note: %s has another x0 than the input\n", c.ref->tag.c_str());
            c.m.assign(steps, {});
        }
        std::string tol_used;
        const json tols = cmp.empty() ? json() : load_rollout_tolerances(tol_path, cfg.uuid, &tol_used);
        if (!cmp.empty()) std::printf("rollout tolerances: %s\n", tol_used.empty() ? "NONE for this checkpoint" : tol_used.c_str());
        if (!cmp.empty() && tols.is_object() && tols.value("steps", 0) < steps)
            throw std::runtime_error("the tolerance file covers fewer steps than requested");

        // ---- forcings source for replay (config `forcings.dynamic` order)
        RefRollout* rsrc = nullptr;
        std::unique_ptr<RefRollout> replay_file;
        if (!replay_from.empty()) {
            replay_file.reset(new RefRollout(replay_from));
            replay_file->check_pairs(cfg);
            rsrc = replay_file.get();
        } else if (in_roll) {
            rsrc = in_roll.get();
        } else if (!cmp.empty()) {
            rsrc = cmp[0].ref.get();
        }
        std::vector<float> replay_buf;
        const float* replay = nullptr;
        if (rsrc && rsrc->steps - 1 >= steps - 1) {
            std::vector<int> perm(F);
            for (int k = 0; k < F; ++k) {
                auto it = std::find(rsrc->forcing_vars.begin(), rsrc->forcing_vars.end(), cfg.forcing_dynamic[k]);
                if (it == rsrc->forcing_vars.end())
                    throw std::runtime_error(rsrc->path + " has no forcing " + cfg.forcing_dynamic[k]);
                perm[k] = int(it - rsrc->forcing_vars.begin());
            }
            bool identity = rsrc->F == F;
            for (int k = 0; k < F; ++k) identity = identity && perm[k] == k;
            if (identity) {
                replay = rsrc->forcings;
            } else {
                replay_buf.resize(size_t(std::max(steps - 1, 0)) * f_n);
                for (int64_t s = 0; s < steps - 1; ++s)
                    for (int64_t g = 0; g < G; ++g)
                        for (int k = 0; k < F; ++k)
                            replay_buf[(s * G + g) * F + k] = rsrc->forcings[(s * G + g) * rsrc->F + perm[k]];
                replay = replay_buf.data();
            }
        }
        if ((forcing_mode == "replay" || verify_replay || check_forcings) && !replay)
            throw std::runtime_error("replayed forcings needed but no rollout file with enough steps (--replay-from)");

        report["model"] = model_path;
        report["uuid"] = cfg.uuid;
        report["device"] = varda::to_string(opts.device);
        report["precision"] = varda::to_string(opts.precision);
        report["engine"] = engine;
        report["cuda_graph"] = opts.device == varda::Device::CUDA ? json(opts.cuda_graph) : json(nullptr);
        report["threads"] = omp_get_max_threads();
        report["load_s"] = load_s;
        report["input"] = input_path;
        report["date0"] = varda::format_iso8601_utc(date0);
        report["timestep_hours"] = cfg.timestep_hours;
        report["steps"] = steps;
        report["forcings"] = forcing_mode;
        report["constants"] = csrc == varda::ConstantSource::Newest ? "newest" : "oldest";
        std::printf("constant inputs: %s\n", varda::to_string(csrc));
        if (replay) report["replay_source"] = rsrc->path;
        if (!tol_used.empty()) report["tolerances"] = tol_used;
        if (const char* j = std::getenv("SLURM_JOB_ID")) report["slurm_job_id"] = j;

        // ---- dates vs the references' recorded dates
        {
            json dj = json::array();
            std::vector<RefRollout*> refs;
            auto add = [&](RefRollout* r) {
                for (RefRollout* q : refs) if (q->path == r->path) return;
                refs.push_back(r);
            };
            if (rsrc) add(rsrc);
            for (auto& c : cmp) add(c.ref.get());
            for (RefRollout* r : refs) {
                int bad = 0;
                for (int s = 0; s + 1 < steps && s < int(r->dates.size()); ++s) {
                    const DateTime want = varda::parse_iso8601_utc(r->dates[s]);
                    if (varda::to_unix_seconds(want) != varda::to_unix_seconds(varda::valid_date(date0, s, cfg.timestep_hours))) ++bad;
                }
                if (varda::to_unix_seconds(varda::parse_iso8601_utc(r->start_date)) != varda::to_unix_seconds(date0)) ++bad;
                std::printf("dates: %s: start %s, %d forcing dates checked, %d differ from date0 + (s+1)*%d h\n", r->tag.c_str(),
                            r->start_date.c_str(), std::min(steps - 1, int(r->dates.size())), bad, cfg.timestep_hours);
                dj.push_back({{"reference", r->path}, {"start_date", r->start_date}, {"mismatches", bad}});
                if (bad) ok = false;
            }
            report["dates_check"] = dj;
        }

        // ---- computed forcings vs the recorded ones, bitwise (independent of the rollout)
        if (check_forcings) {
            std::vector<float> f(f_n);
            int64_t bad = 0;
            double mx = 0;
            json per = json::array();
            for (int s = 0; s + 1 < steps; ++s) {
                varda::compute_forcings(varda::valid_date(date0, s, cfg.timestep_hours), cfg.forcing_dynamic,
                                        model->latitudes().data(), model->longitudes().data(), size_t(G), f.data());
                auto d = bitdiff(f.data(), replay + size_t(s) * f_n, f_n);
                bad += d.first;
                mx = std::max(mx, d.second);
                per.push_back({{"step", s}, {"date", varda::format_iso8601_utc(varda::valid_date(date0, s, cfg.timestep_hours))},
                               {"not_bitwise_equal", d.first}, {"max_abs", d.second}});
            }
            std::printf("forcings: computed vs %s, %d steps x %lld x %d: %lld elements not bitwise equal, max abs %.3g\n",
                        rsrc->tag.c_str(), steps - 1, (long long)G, F, (long long)bad, mx);
            report["forcings_check"] = {{"source", rsrc->path}, {"elements", int64_t(steps - 1) * f_n},
                                        {"not_bitwise_equal", bad}, {"max_abs", mx}, {"per_step", per}};
            if (bad) ok = false;
        }

        // ---- the rollout
        std::vector<float> y_keep;  // all steps, for the bitwise verifications
        std::vector<uint64_t> win_hash;
        const bool keep = verify_replay || verify_host || verify_constants;
        if (keep) y_keep.resize(size_t(steps) * out_n);
        std::vector<float> y_out;
        if (!out_path.empty()) y_out.resize(size_t(steps) * out_n);
        std::vector<double> step_wall;
        double t_prev = 0;

        varda::RolloutRequest req;
        req.x0 = x0;
        req.date0 = date0;
        req.n_steps = steps;
        req.replay_forcings = forcing_mode == "replay" ? replay : nullptr;
        req.on_step = [&](int s, const DateTime& valid, const float* y) {
            const double t = app::now_s();
            step_wall.push_back(t - t_prev);
            t_prev = t;
            for (auto& c : cmp) {
                auto r = compare2(y, c.ref->y + size_t(s) * out_n, out_n, V_out, c.ref->mul_out, c.ref->add_out);
                c.m[s][0] = r.first;
                c.m[s][1] = r.second;
            }
            if (keep) std::memcpy(&y_keep[size_t(s) * out_n], y, out_n * sizeof(float));
            if (!y_out.empty()) std::memcpy(&y_out[size_t(s) * out_n], y, out_n * sizeof(float));
            std::printf("step %2d valid %s%s", s, varda::format_iso8601_utc(valid).c_str(), cmp.empty() ? "" : ":");
            for (auto& c : cmp) std::printf("  %s y %.3e y_norm %.3e", c.ref->tag.c_str(), c.m[s][0].rel, c.m[s][1].rel);
            std::printf("\n");
            std::fflush(stdout);
        };
        // Are the constant-in-time inputs bitwise equal in every slot of a window? (If so, the in-place
        // and roll semantics give the same update.)
        int const_unequal_steps = 0;
        int64_t win_seen = 0;
        auto const_unequal = [&](const float* w) {
            int64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static)
            for (int64_t g = 0; g < G; ++g)
                for (int c : cfg.forcing_constant_index)
                    for (int t = 1; t < T; ++t)
                        if (std::memcmp(&w[(int64_t(t) * G + g) * V_in + c], &w[g * V_in + c], 4) != 0) ++bad;
            return bad != 0;
        };
        const bool want_window = !cmp.empty() || keep;
        if (want_window) {
            req.on_window = [&](int s, const float* w) {
                if (keep) win_hash.push_back(hash_bytes(w, in_n));
                ++win_seen;
                if (const_unequal(w)) ++const_unequal_steps;
                for (auto& c : cmp) {
                    if (const_unequal(c.ref->window + size_t(s) * in_n)) ++c.const_unequal_steps;
                    auto r = compare2(w, c.ref->window + size_t(s) * in_n, in_n, V_in, c.ref->mul_in, c.ref->add_in);
                    c.m[s][2] = r.first;
                    c.m[s][3] = r.second;
                    if (s >= 1) {  // what the update wrote into the newest slot's forcing columns
                        const float* nw = w + int64_t(T - 1) * G * V_in;
                        const float* rf = c.ref->forcings + size_t(s - 1) * G * c.ref->F;
                        for (int k = 0; k < F; ++k) {
                            const int ci = cfg.forcing_dynamic_index[k];
                            const int rk = int(std::find(c.ref->forcing_vars.begin(), c.ref->forcing_vars.end(),
                                                         cfg.forcing_dynamic[k]) - c.ref->forcing_vars.begin());
                            for (int64_t g = 0; g < G; ++g) {
                                const float a = nw[g * V_in + ci], b = rf[g * c.ref->F + rk];
                                if (std::memcmp(&a, &b, 4) != 0) {
                                    ++c.forcing_cols_bad;
                                    c.forcing_cols_maxabs = std::max(c.forcing_cols_maxabs, std::fabs(double(a) - b));
                                }
                            }
                        }
                    }
                }
            };
        }

        auto run_cs = [&](const varda::RolloutRequest& q, const std::string& eng, varda::ConstantSource cs) -> double {
            const double ts = app::now_s();
            t_prev = ts;
            if (eng == "host") {
                varda::Rollout r(*model, cs);
                r.run(q);
            } else {
#ifdef VARDA_WITH_CUDA
                varda::cuda::DeviceRollout r(*model, opts.cuda_graph, cs);
                r.run(q);
#endif
            }
            return app::now_s() - ts;
        };
        auto run_with = [&](const varda::RolloutRequest& q, const std::string& eng) { return run_cs(q, eng, csrc); };

        const double run_s = run_with(req, engine);
        std::printf("rollout: %d steps in %.3f s (%.3f s/step, incl. %s)\n", steps, run_s, run_s / steps,
                    want_window ? "per-step window copies and comparisons" : "callbacks");
        report["run_s"] = run_s;
        report["step_wall_s"] = step_wall;

        json comps = json::array();
        for (auto& c : cmp) {
            json j = judge(c, tols, steps, &ok);
            j["forcing_columns_vs_reference"] = {{"steps", steps - 1}, {"not_bitwise_equal", c.forcing_cols_bad},
                                                 {"max_abs", c.forcing_cols_maxabs}};
            std::printf("window forcing columns vs %s forcings (steps 1..%d): %lld not bitwise equal\n", c.ref->tag.c_str(),
                        steps - 1, (long long)c.forcing_cols_bad);
            comps.push_back(j);
        }
        if (!cmp.empty()) {
            report["comparisons"] = comps;
            if (tols.is_null()) ok = false;  // a comparison without tolerances cannot pass
        }

        // ---- bitwise verifications against a second rollout
        auto verify_against = [&](varda::RolloutRequest q, const std::string& eng, const std::string& name,
                                  varda::ConstantSource cs) {
            std::vector<float> y2(size_t(steps) * out_n);
            std::vector<uint64_t> h2;
            q.on_step = [&](int s, const DateTime&, const float* y) { std::memcpy(&y2[size_t(s) * out_n], y, out_n * 4); };
            q.on_window = [&](int, const float* w) { h2.push_back(hash_bytes(w, in_n)); };
            run_cs(q, eng, cs);
            json per = json::array();
            int64_t bad_total = 0;
            int win_bad = 0;
            for (int s = 0; s < steps; ++s) {
                auto d = bitdiff(&y_keep[size_t(s) * out_n], &y2[size_t(s) * out_n], out_n);
                const bool weq = win_hash[s] == h2[s];
                bad_total += d.first;
                win_bad += !weq;
                per.push_back({{"step", s}, {"y_not_bitwise_equal", d.first}, {"y_max_abs", d.second}, {"window_hash_equal", weq}});
            }
            const bool pass = bad_total == 0 && win_bad == 0;
            std::printf("%s: y %s at every step (%lld elements differ), windows %s (%d of %d steps differ, 64-bit hash)\n",
                        name.c_str(), bad_total ? "NOT bitwise equal" : "bitwise equal", (long long)bad_total,
                        win_bad ? "DIFFER" : "equal", win_bad, steps);
            if (!pass) ok = false;
            return json{{"pass", pass}, {"y_elements_not_bitwise_equal", bad_total}, {"window_steps_differ", win_bad}, {"per_step", per}};
        };
        if (verify_replay) {
            varda::RolloutRequest q = req;
            q.replay_forcings = req.replay_forcings ? nullptr : replay;
            report["verify_replay"] = verify_against(q, engine, std::string("computed vs replayed forcings (") +
                                                                     (req.replay_forcings ? "second run computed" : "second run replayed") + ")", csrc);
        }
        if (verify_host) {
            if (engine != "device") throw std::runtime_error("--verify-host needs the device engine");
            report["verify_host"] = verify_against(req, "host", "device rollout vs host-level Rollout (same backend)", csrc);
        }
        if (verify_constants) {
            const varda::ConstantSource other =
                csrc == varda::ConstantSource::Newest ? varda::ConstantSource::Oldest : varda::ConstantSource::Newest;
            report["verify_constants"] = verify_against(req, engine, std::string("constants ") + varda::to_string(csrc) +
                                                                         " vs " + varda::to_string(other), other);
        }
        {
            json cj = {{"run_windows_steps_with_unequal_constants", const_unequal_steps}};
            for (auto& c : cmp) cj[c.ref->tag] = c.const_unequal_steps;
            report["constants_across_slots"] = cj;
            std::printf("constant inputs unequal across slots: this run's windows %d of %d steps", const_unequal_steps,
                        int(win_seen));
            for (auto& c : cmp) std::printf(", %s windows %d", c.ref->tag.c_str(), c.const_unequal_steps);
            std::printf("\n");
        }

        // ---- benchmark
#ifdef VARDA_WITH_CUDA
        if (bench > 0 && engine == "device") {
            varda::cuda::DeviceRollout r(*model, opts.cuda_graph, csrc);
            varda::RolloutRequest q = req;
            q.on_window = nullptr;
            q.on_step = [](int, const DateTime&, const float*) {};  // outputs to pinned host memory, untouched
            r.run(q);  // warm-up
            std::vector<double> walls, med_steps;
            std::vector<float> all;
            for (int b = 0; b < bench; ++b) {
                r.run(q);
                walls.push_back(r.wall_s() / steps);
                std::vector<double> v(r.step_ms().begin(), r.step_ms().end());
                all.insert(all.end(), r.step_ms().begin(), r.step_ms().end());
                med_steps.push_back(median(v));
            }
            std::vector<double> allv(all.begin(), all.end());
            const double fstep_s = median(std::vector<double>(r.forcing_host_s().begin(), r.forcing_host_s().end() - 1));
            json bj = {{"rollouts", bench},
                       {"steps", steps},
                       {"forcings", forcing_mode},
                       {"wall_ms_per_step_median", 1e3 * median(walls)},
                       {"gpu_ms_per_step_median", median(allv)},
                       {"gpu_ms_per_step_max", *std::max_element(allv.begin(), allv.end())},
                       {"gpu_ms_per_step_min", *std::min_element(allv.begin(), allv.end())},
                       {"host_forcing_ms_per_step_median", 1e3 * fstep_s},
                       {"last_run_step_ms", r.step_ms()},
                       {"rollout_device_gb", r.device_bytes() / 1e9},
                       {"rollout_pinned_gb", r.pinned_bytes() / 1e9}};
            std::printf("bench (%s %s, graph %s, %d x %d steps, forcings %s): wall %.2f ms/step, GPU %.2f ms/step (median; "
                        "min %.2f max %.2f), host forcings %.2f ms/step (overlapped)\n",
                        varda::to_string(opts.device).c_str(), varda::to_string(opts.precision).c_str(),
                        opts.cuda_graph ? "on" : "off", bench, steps, forcing_mode.c_str(), 1e3 * median(walls), median(allv),
                        *std::min_element(allv.begin(), allv.end()), *std::max_element(allv.begin(), allv.end()), 1e3 * fstep_s);
            if (forward_iters > 0) {
                r.time_forward(std::min(forward_iters, 3));  // warm
                auto ms = r.time_forward(forward_iters);
                std::vector<double> v(ms.begin(), ms.end());
                const double fwd = median(v);
                bj["forward_only_ms_median"] = fwd;
                bj["forward_only_ms_min"] = *std::min_element(v.begin(), v.end());
                bj["update_overhead_ms"] = median(allv) - fwd;
                std::printf("forward-only graph (normalise -> forward -> denormalise, %d replays): %.2f ms median; "
                            "rollout step - forward = %.3f ms (%.2f%%)\n",
                            forward_iters, fwd, median(allv) - fwd, 100.0 * (median(allv) - fwd) / fwd);
            }
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            auto* be = varda::cuda::as_cuda(*model);
            bj["device_used_gb"] = (tot - fr) / 1e9;
            bj["backend_weight_gb"] = be->weight_bytes() / 1e9;
            bj["backend_activation_gb"] = be->activation_bytes() / 1e9;
            report["bench"] = bj;
            // the host-level loop on the same backend, for comparison (host copies every step)
            {
                varda::Rollout h(*model, csrc);
                varda::RolloutRequest qh = q;
                const double ts = app::now_s();
                h.run(qh);
                const double hs = (app::now_s() - ts) / steps;
                report["bench"]["host_rollout_wall_ms_per_step"] = 1e3 * hs;
                std::printf("host-level Rollout on the same backend: %.2f ms/step (wall, H2D + forward + D2H + host update)\n",
                            1e3 * hs);
            }
        }
#endif
        if (bench > 0 && engine == "host") {
            varda::Rollout r(*model, csrc);
            varda::RolloutRequest q = req;
            q.on_window = nullptr;
            q.on_step = nullptr;
            std::vector<double> per;
            for (int b = 0; b < bench; ++b) {
                const double ts = app::now_s();
                r.run(q);
                per.push_back((app::now_s() - ts) / steps);
            }
            double pred = 0, upd = 0, frc = 0;
            for (auto& t : r.times()) pred += t.predict, upd += t.update, frc += t.forcings;
            report["bench"] = {{"rollouts", bench}, {"wall_s_per_step_median", median(per)},
                               {"predict_s_per_step", pred / steps}, {"update_s_per_step", upd / std::max(1, steps - 1)},
                               {"forcings_s_per_step", frc / std::max(1, steps - 1)}};
            std::printf("bench host (%s): %.3f s/step; predict %.3f, update %.4f, forcings %.4f s/step\n",
                        varda::to_string(opts.device).c_str(), median(per), pred / steps, upd / std::max(1, steps - 1),
                        frc / std::max(1, steps - 1));
        }

        if (!out_path.empty()) {
            varda::write_safetensors(out_path, {{"y", varda::DType::F32, {steps, G, V_out}, y_out.data()}},
                                     {{"model", model_path}, {"uuid", cfg.uuid}, {"date0", varda::format_iso8601_utc(date0)},
                                      {"device", varda::to_string(opts.device)}, {"precision", varda::to_string(opts.precision)},
                                      {"engine", engine}, {"forcings", forcing_mode}});
            std::printf("wrote %s\n", out_path.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        report["error"] = e.what();
        ok = false;
    }
    report["pass"] = ok;
    if (!json_path.empty()) {
        const std::string tmp = json_path + ".tmp";
        std::ofstream(tmp) << report.dump(1) << "\n";
        std::rename(tmp.c_str(), json_path.c_str());
    }
    std::printf("%s\n", ok ? "varda-rollout: OK" : "varda-rollout: FAILED");
    return ok ? 0 : 1;
}
