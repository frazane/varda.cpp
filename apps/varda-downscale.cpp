// varda-downscale: runs the 1-hourly temporal downscaler (docs/downscaler.md) and judges it.
//
//   (a) pre-assembled inputs, judged against a reference dump (the model alone):
//   varda-downscale --model TD.safetensors --ref model_io.safetensors [--windows N] [--date ISO]
//                   [--device cpu|cuda|metal] [--precision fp32|fp16] [--threads N] [--cuda-device I]
//                   [--tolerances FILE] [--table NAME] [--json OUT.json]
//   (b) the full loop over a forecaster run (GRIB-free: the forecaster's outputs in memory):
//   varda-downscale --model TD.safetensors --forecast FORECAST.safetensors [--x0 X0.safetensors]
//                   [--forecaster-model F.safetensors] [--windows N] [--date ISO]
//                   [--out HOURLY.safetensors] [--dump-x X.safetensors] [--ref model_io.safetensors] ...
//   (c) smoke test on any interpolator model, with a synthetic forecaster derived from its own names:
//   varda-downscale --model TD.safetensors --smoke [--windows N] [--date ISO] ...
//
// Reference dump (--ref): `x.<k>` F32 (2, G, V_in) and `y.<k>` F32 (6, G, V_out) per window k, physical
// units; metadata optional (a `date`/`date0` entry, flat or inside a `meta` JSON, supplies --date).
// Mode (a) feeds x.<k> to the model and compares y per output step (contract metric, docs/format.md §2:
// float64, rel_absmax = max|a-r| / max|r| over the whole (G, V_out) tensor, plus max_abs; per variable
// as well). Mode (b) assembles x itself (src/varda/downscale.h); with --ref it also compares the
// assembled x.<k> per slot and channel (memory vs GRIB-packed inputs) and y.<k>.
//
// Forecast file (--forecast): `y.<s>` F32 [G, V_fout], s = 1..N, valid at date0 + s * W (what
// `varda-forecast --dump-y` writes); `x0` F32 [1, T, G, V_fin] or [T, G, V_fin] (here, or in --x0
// FILE under `x0` or `in.x`); metadata `meta` JSON {out_vars, in_vars, date} (in_vars, or the grid, may
// come from --forecaster-model instead; the date from --date).
//
// Tolerances (--tolerances): JSON {"<table>": {"y": T or [T_0 .. T_5] or [[per step] per window], "y_var": same}}
// (tolerances/downscale/<model>.json, python/reference/calibrate_downscale_tolerances.py; `y` the overall
// rel_absmax, `y_var` the worst per-variable one; default table fast-fp32-vs-ref / fast-fp16-vs-ref, also
// fast-fp16-vs-ref-fp16 with --table); without them, comparisons are reported only. Exit 1 on a
// failed verdict or a failed self-check, 2 on usage.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "common.h"
#include "varda/downscale.h"
#include "varda/forcings.h"

using app::json;
using varda::DateTime;

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-downscale --model TD --ref model_io.safetensors | --forecast F.safetensors | --smoke\n"
                 "         [--x0 X0.safetensors] [--forecaster-model F] [--windows N] [--date ISO]\n"
                 "         [--device cpu|cuda|metal] [--precision fp32|fp16] [--threads N] [--cuda-device I]\n"
                 "         [--out HOURLY.safetensors] [--dump-x X.safetensors] [--tolerances FILE] [--table NAME]\n"
                 "         [--json OUT.json]\n");
    std::exit(2);
}

// ---- metrics ---------------------------------------------------------------------------------

struct VarMetric {
    double rel = 0, max_abs = 0, absmax = 0;
    int64_t n_nan = 0, n_ref_bad = 0;
    int worst = -1;  // variable with the largest rel
    std::vector<double> var_rel, var_max_abs, var_absmax;
};

// Contract metric of a (G, V) float tensor against a reference, overall and per variable.
VarMetric metric_vars(const float* a, const float* r, int64_t G, int V) {
    VarMetric m;
    m.var_max_abs.assign(V, 0.0);
    m.var_absmax.assign(V, 0.0);
    int64_t nan = 0, bad = 0;
#pragma omp parallel
    {
        std::vector<double> lm(V, 0.0), la(V, 0.0);
        int64_t lnan = 0, lbad = 0;
#pragma omp for schedule(static) nowait
        for (int64_t g = 0; g < G; ++g)
            for (int v = 0; v < V; ++v) {
                const double av = a[g * V + v], rv = r[g * V + v];
                if (!std::isfinite(rv)) { ++lbad; continue; }
                if (std::isnan(av)) { ++lnan; continue; }
                lm[v] = std::max(lm[v], std::fabs(av - rv));
                la[v] = std::max(la[v], std::fabs(rv));
            }
#pragma omp critical
        {
            for (int v = 0; v < V; ++v) {
                m.var_max_abs[v] = std::max(m.var_max_abs[v], lm[v]);
                m.var_absmax[v] = std::max(m.var_absmax[v], la[v]);
            }
            nan += lnan;
            bad += lbad;
        }
    }
    const double inf = std::numeric_limits<double>::infinity();
    auto rel = [&](double ma, double am) { return am > 0 ? ma / am : (ma > 0 ? inf : 0.0); };
    m.n_nan = nan;
    m.n_ref_bad = bad;
    m.var_rel.resize(V);
    double wr = -1;
    for (int v = 0; v < V; ++v) {
        m.max_abs = std::max(m.max_abs, m.var_max_abs[v]);
        m.absmax = std::max(m.absmax, m.var_absmax[v]);
        m.var_rel[v] = rel(m.var_max_abs[v], m.var_absmax[v]);
        if (m.var_rel[v] > wr) wr = m.var_rel[v], m.worst = v;
    }
    m.rel = rel(m.max_abs, m.absmax);
    return m;
}

json num(double x) { return std::isfinite(x) ? json(x) : json(nullptr); }

json to_json(const VarMetric& m, const std::vector<std::string>& names, bool per_var) {
    json j = {{"rel_absmax", num(m.rel)}, {"max_abs", num(m.max_abs)}, {"ref_absmax", num(m.absmax)},
              {"n_nan", m.n_nan}, {"n_ref_nonfinite", m.n_ref_bad},
              {"worst_variable", m.worst >= 0 ? names.at(m.worst) : std::string()},
              {"worst_variable_rel_absmax", m.worst >= 0 ? num(m.var_rel[m.worst]) : json(nullptr)}};
    if (per_var) {
        json v = json::object();
        for (size_t i = 0; i < names.size(); ++i)
            v[names[i]] = {{"rel_absmax", num(m.var_rel[i])}, {"max_abs", num(m.var_max_abs[i])}};
        j["variables"] = v;
    }
    return j;
}

// ---- files -----------------------------------------------------------------------------------

const float* f32(varda::SafeTensorsFile& f, const std::string& key, int64_t numel) {
    const auto& t = f.peek(key);
    if (t.dtype != varda::DType::F32 || t.numel() != numel)
        throw std::runtime_error(f.path() + ": '" + key + "' is " + t.shape_str() + " " + varda::dtype_name(t.dtype) +
                                 ", expected " + std::to_string(numel) + " F32 values");
    return t.as<float>();
}

// Metadata entry `key`, flat or inside the `meta` JSON; "" if absent.
std::string meta_get(const varda::SafeTensorsFile& f, const std::vector<std::string>& keys) {
    for (const auto& k : keys)
        if (f.metadata().count(k)) return f.metadata().at(k);
    if (f.metadata().count("meta")) {
        try {
            const json m = json::parse(f.metadata().at("meta"));
            for (const auto& k : keys)
                if (m.contains(k) && m[k].is_string()) return m[k].get<std::string>();
        } catch (...) {}
    }
    return "";
}

std::vector<std::string> meta_names(const varda::SafeTensorsFile& f, const std::string& key) {
    auto parse = [&](const json& m) {
        std::vector<std::string> v;
        if (m.contains(key) && m[key].is_array()) v = m[key].get<std::vector<std::string>>();
        return v;
    };
    if (f.metadata().count(key)) {
        try { return json::parse(f.metadata().at(key)).get<std::vector<std::string>>(); } catch (...) {}
    }
    if (f.metadata().count("meta")) {
        try { return parse(json::parse(f.metadata().at("meta"))); } catch (...) {}
    }
    return {};
}

int count_windows(const varda::SafeTensorsFile& f, const std::string& prefix, int first) {
    int n = 0;
    while (f.has(prefix + std::to_string(first + n))) ++n;
    return n;
}

// ---- tolerance judging -----------------------------------------------------------------------

// y tolerance for (window k, step t); <= 0 = none.
double tol_at(const json& y, int k, int t) {
    if (y.is_number()) return y.get<double>();
    if (y.is_array() && !y.empty() && y[0].is_array()) {
        if (size_t(k) >= y.size())
            throw std::runtime_error("tolerances: window " + std::to_string(k) + " beyond the " + std::to_string(y.size()) +
                                     " calibrated windows (use --windows, or a per-step table)");
        return y.at(k).at(t).get<double>();
    }
    if (y.is_array()) return y.at(t).get<double>();
    return -1;
}

}  // namespace

// Gathers everything one run produces, to report it after the loop.
struct Results {
    int N = 0, T_out = 0;
    std::vector<std::vector<VarMetric>> y;           // [window][output step], vs --ref
    std::vector<std::vector<VarMetric>> x;           // [window][slot], vs --ref (assembled inputs, mode b)
    std::vector<std::vector<float>> hourly;          // --out / smoke: y per output, in call order
    std::vector<int> hours;
    std::vector<std::vector<float>> xkeep;           // --dump-x / smoke: assembled x per window
    int64_t nonfinite = 0;
    bool dates_ok = true;
};

int main(int argc, char** argv) {
    std::string model_path, ref_path, forecast_path, x0_path, fmodel_path, date_s, out_path, dumpx_path, tol_path, table,
        json_path;
    int windows = 0;
    bool smoke = false;
    varda::Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "--forecast") forecast_path = next();
        else if (a == "--x0") x0_path = next();
        else if (a == "--forecaster-model") fmodel_path = next();
        else if (a == "--smoke") smoke = true;
        else if (a == "--windows") windows = std::stoi(next());
        else if (a == "--date") date_s = next();
        else if (a == "--device") { try { opts.device = app::parse_device(next()); } catch (...) { usage(); } }
        else if (a == "--precision") { try { opts.precision = app::parse_precision(next()); } catch (...) { usage(); } }
        else if (a == "--threads") opts.num_threads = std::stoi(next());
        else if (a == "--cuda-device") opts.cuda_device = std::stoi(next());
        else if (a == "--no-cuda-graph") opts.cuda_graph = false;
        else if (a == "--out") out_path = next();
        else if (a == "--dump-x") dumpx_path = next();
        else if (a == "--tolerances") tol_path = next();
        else if (a == "--table") table = next();
        else if (a == "--json") json_path = next();
        else usage();
    }
    if (model_path.empty() || (ref_path.empty() && forecast_path.empty() && !smoke)) usage();
    if (!forecast_path.empty() && smoke) usage();
    if (opts.num_threads > 0) omp_set_num_threads(opts.num_threads);

    bool ok = true;
    json report;
    try {
        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        const varda::Config& cfg = model->config();
        if (!cfg.is_interpolator())
            throw std::runtime_error(model_path + ": not a temporal downscaler (model.kind " + cfg.model_kind + ")");
        const int64_t G = cfg.n_data_nodes;
        const int V_in = cfg.n_input_vars, V_out = cfg.n_output_vars, T_out = cfg.n_output_steps;
        const size_t x_n = size_t(2) * G * V_in, y_n = size_t(G) * V_out;
        std::printf("model %s (uuid %s), %s %s, threads %d, load %.2f s\n", model_path.c_str(), cfg.uuid.c_str(),
                    varda::to_string(opts.device).c_str(), varda::to_string(opts.precision).c_str(), omp_get_max_threads(), load_s);

        std::unique_ptr<varda::SafeTensorsFile> ref, fc_file;
        if (!ref_path.empty()) ref.reset(new varda::SafeTensorsFile(ref_path));
        if (!forecast_path.empty()) fc_file.reset(new varda::SafeTensorsFile(forecast_path));

        // ---- date and window count
        if (date_s.empty() && fc_file) date_s = meta_get(*fc_file, {"date", "date0"});
        if (date_s.empty() && ref) date_s = meta_get(*ref, {"date0", "date", "start_date", "reference_date"});
        if (date_s.empty()) date_s = "2026-10-22T15:00:00";
        const DateTime date0 = varda::parse_iso8601_utc(date_s);
        int n_avail;
        if (fc_file) n_avail = count_windows(*fc_file, "y.", 1);
        else if (ref) n_avail = std::min(count_windows(*ref, "x.", 0), count_windows(*ref, "y.", 0));
        else n_avail = windows > 0 ? windows : 3;  // smoke: any number
        const int N = windows > 0 ? windows : n_avail;
        if (N < 1 || N > n_avail)
            throw std::runtime_error("windows: " + std::to_string(N) + " requested, " + std::to_string(n_avail) + " available");
        if (ref && fc_file && std::min(count_windows(*ref, "x.", 0), count_windows(*ref, "y.", 0)) < N)
            throw std::runtime_error("reference has fewer x.<k> / y.<k> than windows");
        const char* mode = fc_file ? "forecast" : smoke ? "smoke" : "reference_inputs";
        std::printf("mode %s, %d window(s), date0 %s\n", mode, N, varda::format_iso8601_utc(date0).c_str());

        report["model"] = model_path;
        report["uuid"] = cfg.uuid;
        report["device"] = varda::to_string(opts.device);
        report["precision"] = varda::to_string(opts.precision);
        report["threads"] = omp_get_max_threads();
        report["load_s"] = load_s;
        report["mode"] = mode;
        report["date0"] = varda::format_iso8601_utc(date0);
        report["windows"] = N;
        if (const char* j = std::getenv("SLURM_JOB_ID")) report["slurm_job_id"] = j;
        if (ref) report["reference"] = ref_path;
        if (fc_file) report["forecast"] = forecast_path;

        // ---- tolerances
        json tols;
        const std::string tol_table = !table.empty() ? table
            : opts.precision == varda::Precision::FP16 ? "fast-fp16-vs-ref" : "fast-fp32-vs-ref";
        if (!tol_path.empty()) {
            std::ifstream f(tol_path);
            if (!f) throw std::runtime_error("cannot read tolerances " + tol_path);
            tols = json::parse(f);
            if (!tols.contains(tol_table) || !tols[tol_table].contains("y"))
                throw std::runtime_error(tol_path + ": no table '" + tol_table + "' with key y");
            report["tolerances"] = {{"file", tol_path}, {"table", tol_table}};
        }

        Results R;
        R.N = N;
        R.T_out = T_out;
        R.y.assign(N, std::vector<VarMetric>(T_out));
        R.x.assign(N, std::vector<VarMetric>(2));
        const bool keep_y = !out_path.empty() || smoke;
        const bool keep_x = !dumpx_path.empty() || smoke;
        std::vector<double> predict_s;

        auto on_output = [&](int k, int t, int step_h, const DateTime& valid, const float* y) {
            if (cfg.frequency_hours == 1 && cfg.output_times.size() == size_t(T_out) &&
                (step_h != k * (cfg.input_times[1] - cfg.input_times[0]) * 1 + (cfg.output_times[t] - cfg.input_times[0]) ||
                 varda::format_iso8601_utc(valid) != varda::format_iso8601_utc(varda::add_hours(date0, step_h))))
                R.dates_ok = false;
            if (keep_y) {
                R.hourly.emplace_back(y, y + y_n);
                R.hours.push_back(step_h);
            }
            int64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static)
            for (int64_t i = 0; i < int64_t(y_n); ++i) bad += !std::isfinite(y[i]);
            R.nonfinite += bad;
            if (ref) {
                const float* r = f32(*ref, "y." + std::to_string(k), int64_t(T_out) * y_n) + size_t(t) * y_n;
                R.y[k][t] = metric_vars(y, r, G, V_out);
            }
        };

        std::unique_ptr<varda::Downscaler> ds;
        std::vector<float> x0_own;
        std::vector<std::vector<float>> y_own;
        const float* x0 = nullptr;
        int T_f = 1;
        varda::ForecasterInfo fi;
        if (!fc_file && !smoke) {
            // ---- (a) pre-assembled inputs
            ds.reset(new varda::Downscaler(*model));
            for (int k = 0; k < N; ++k) {
                const float* x = f32(*ref, "x." + std::to_string(k), int64_t(x_n));
                ds->run_window(k, date0, x, on_output);
                std::printf("window %d done, predict %.2f s\n", k, ds->window_times()[k].predict);
                std::fflush(stdout);
            }
        } else {
            // ---- (b)/(c) the full loop
            std::vector<std::string> in_names, out_names;
            if (fc_file) {
                out_names = meta_names(*fc_file, "out_vars");
                in_names = meta_names(*fc_file, "in_vars");
            }
            if (!fmodel_path.empty()) {
                fi = varda::ForecasterInfo::from_file(fmodel_path);
                if (!out_names.empty() && out_names != fi.out_names)
                    throw std::runtime_error("forecast file out_vars differ from --forecaster-model's var_output");
                if (!in_names.empty() && in_names != fi.in_names)
                    throw std::runtime_error("forecast file in_vars differ from --forecaster-model's var_input");
            } else if (smoke) {
                fi.in_names = cfg.var_input;
                fi.out_names = cfg.var_output;
                fi.n_data_nodes = G;
                fi.lat = model->latitudes();
                fi.lon = model->longitudes();
            } else {
                if (in_names.empty() || out_names.empty())
                    throw std::runtime_error("the forecast file's metadata has no in_vars / out_vars: pass --forecaster-model");
                fi.in_names = in_names;
                fi.out_names = out_names;
                fi.n_data_nodes = G;
                std::printf("note: no --forecaster-model: the forecaster grid is not checked against the downscaler's\n");
            }
            ds.reset(new varda::Downscaler(*model, fi));
            const int64_t V_fin = int64_t(fi.in_names.size()), V_fout = int64_t(fi.out_names.size());
            report["forecaster"] = {{"n_inputs", V_fin}, {"n_outputs", V_fout}, {"grid_checked_bitwise", !fi.lat.empty()}};
            std::unique_ptr<varda::SafeTensorsFile> x0_file;
            if (smoke) {
                // deterministic, distinct per (slot, point, variable)
                auto val = [](int64_t s, int64_t g, int64_t v) {
                    uint64_t h = uint64_t(s + 1) * 0x9E3779B97F4A7C15ull ^ uint64_t(g + 1) * 0xBF58476D1CE4E5B9ull ^ uint64_t(v + 1) * 0x94D049BB133111EBull;
                    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
                    return float(double(h % 2000001) / 1000000.0 - 1.0);
                };
                x0_own.resize(size_t(G) * V_fin);
                for (int64_t g = 0; g < G; ++g)
                    for (int64_t v = 0; v < V_fin; ++v) x0_own[g * V_fin + v] = val(1000, g, v);
                y_own.assign(N, std::vector<float>(size_t(G) * V_fout));
                for (int s = 0; s < N; ++s)
                    for (int64_t g = 0; g < G; ++g)
                        for (int64_t v = 0; v < V_fout; ++v) y_own[s][g * V_fout + v] = val(s, g, v);
                x0 = x0_own.data();
            } else {
                varda::SafeTensorsFile* xf = fc_file.get();
                std::string key = "x0";
                if (!x0_path.empty()) {
                    x0_file.reset(new varda::SafeTensorsFile(x0_path));
                    xf = x0_file.get();
                    if (!xf->has("x0")) key = "in.x";
                } else if (!xf->has("x0")) {
                    throw std::runtime_error(forecast_path + ": no `x0` tensor (the forecaster's input window); pass --x0 FILE");
                }
                const auto& t = xf->peek(key);
                const int64_t per_slot = G * V_fin;
                if (t.dtype != varda::DType::F32 || t.numel() % per_slot != 0 || t.numel() == 0)
                    throw std::runtime_error(xf->path() + ": '" + key + "' " + t.shape_str() +
                                             " is not (1, T, G, V_in) F32 of the forecaster");
                T_f = int(t.numel() / per_slot);
                x0 = t.as<float>();
            }
            report["forecaster_T"] = T_f;

            varda::DownscaleRequest rq;
            rq.x0 = x0;
            rq.forecaster_T = T_f;
            rq.date0 = date0;
            rq.n_windows = N;
            rq.y_at = [&](int s) -> const float* {
                if (smoke) return y_own[s].data();
                return f32(*fc_file, "y." + std::to_string(s + 1), int64_t(G) * V_fout);
            };
            rq.on_output = on_output;
            rq.on_input = [&](int k, const float* x) {
                if (keep_x) R.xkeep.emplace_back(x, x + x_n);
                if (ref) {
                    const float* rx = f32(*ref, "x." + std::to_string(k), int64_t(x_n));
                    for (int sl = 0; sl < 2; ++sl)
                        R.x[k][sl] = metric_vars(x + size_t(sl) * G * V_in, rx + size_t(sl) * G * V_in, G, V_in);
                }
            };
            t0 = app::now_s();
            ds->run(rq);
            report["run_s"] = app::now_s() - t0;
            for (int k = 0; k < N; ++k) {
                const auto& w = ds->window_times()[k];
                std::printf("window %d: assemble %.2f s (forcings %.2f s), predict %.2f s\n", k, w.assemble, w.forcings, w.predict);
            }

            if (smoke) {
                // self-checks: (1) assembly by an independent by-name implementation, (2) the pre-assembled
                // entry point gives bitwise the same y, (3) finite outputs, (4) dates.
                json sc = json::object();
                std::map<std::string, int> in_ix, out_ix, con_ix;
                for (size_t i = 0; i < fi.in_names.size(); ++i) in_ix[fi.in_names[i]] = int(i);
                for (size_t i = 0; i < fi.out_names.size(); ++i) out_ix[fi.out_names[i]] = int(i);
                std::map<std::string, bool> is_dyn, is_con;
                for (const auto& n : cfg.forcing_dynamic) is_dyn[n] = true;
                for (const auto& n : cfg.forcing_constant) is_con[n] = true;
                int64_t x_bad = 0;
                std::vector<float> fcol(static_cast<size_t>(G));
                const float* x0n = x0 + size_t(T_f - 1) * G * V_fin;
                for (int k = 0; k < N; ++k)
                    for (int sl = 0; sl < 2; ++sl) {
                        const DateTime d = varda::add_hours(date0, int64_t(k + sl) * ds->times().window_hours);
                        const float* st = sl == 1 ? y_own[k].data() : k >= 1 ? y_own[k - 1].data() : nullptr;
                        const float* xs = R.xkeep[k].data() + size_t(sl) * G * V_in;
                        for (int c = 0; c < V_in; ++c) {
                            const std::string& n = cfg.var_input[c];
                            if (is_dyn.count(n)) {
                                varda::compute_forcings(d, {n}, model->latitudes().data(), model->longitudes().data(), size_t(G), fcol.data());
                                for (int64_t g = 0; g < G; ++g) x_bad += std::memcmp(&fcol[g], &xs[g * V_in + c], 4) != 0;
                            } else if (is_con.count(n)) {
                                for (int64_t g = 0; g < G; ++g) x_bad += std::memcmp(&x0n[g * V_fin + in_ix.at(n)], &xs[g * V_in + c], 4) != 0;
                            } else if (st) {
                                for (int64_t g = 0; g < G; ++g) x_bad += std::memcmp(&st[g * V_fout + out_ix.at(n)], &xs[g * V_in + c], 4) != 0;
                            } else {
                                for (int64_t g = 0; g < G; ++g) x_bad += std::memcmp(&x0n[g * V_fin + in_ix.at(n)], &xs[g * V_in + c], 4) != 0;
                            }
                        }
                    }
                sc["assembly_mismatches"] = x_bad;
                // pre-assembled entry point
                varda::Downscaler ds2(*model);
                int64_t y_bad = 0;
                for (int k = 0; k < N; ++k) {
                    size_t idx = size_t(k) * T_out;
                    ds2.run_window(k, date0, R.xkeep[k].data(), [&](int, int t, int, const DateTime&, const float* y) {
                        y_bad += std::memcmp(y, R.hourly[idx + t].data(), y_n * 4) != 0;
                    });
                }
                sc["run_window_vs_run_y_mismatches"] = y_bad;
                sc["nonfinite_outputs"] = R.nonfinite;
                sc["dates_ok"] = R.dates_ok;
                const bool pass = x_bad == 0 && y_bad == 0 && R.nonfinite == 0 && R.dates_ok;
                sc["pass"] = pass;
                report["smoke"] = sc;
                std::printf("smoke: assembly mismatches %lld, run vs run_window y mismatches %lld, non-finite %lld, dates %s -> %s\n",
                            (long long)x_bad, (long long)y_bad, (long long)R.nonfinite, R.dates_ok ? "ok" : "BAD", pass ? "PASS" : "FAIL");
                if (!pass) ok = false;
            }
        }
        for (int k = 0; k < N; ++k) predict_s.push_back(ds->window_times()[k].predict);
        report["predict_s"] = predict_s;
        report["dates_ok"] = R.dates_ok;
        report["nonfinite_outputs"] = R.nonfinite;
        if (!R.dates_ok) ok = false;
        if (R.nonfinite) ok = false;

        // ---- input comparison (mode b with --ref)
        if (ref && fc_file) {
            json xj = json::array();
            double worst = 0;
            std::string wchan;
            int wk = -1, wsl = -1;
            std::printf("\nassembled inputs vs reference x.<k> (rel_absmax = max|a-r|/max|r| per channel; worst channel per slot)\n");
            std::printf("%6s %4s %12s %12s  %s\n", "window", "slot", "rel_absmax", "max_abs", "worst channel (rel)");
            for (int k = 0; k < N; ++k)
                for (int sl = 0; sl < 2; ++sl) {
                    const VarMetric& m = R.x[k][sl];
                    std::printf("%6d %4d %12.3e %12.3e  %s (%.3e)\n", k, sl, m.rel, m.max_abs,
                                cfg.var_input[m.worst].c_str(), m.var_rel[m.worst]);
                    json e = to_json(m, cfg.var_input, true);
                    e["window"] = k;
                    e["slot"] = sl;
                    int64_t exact = 0;
                    for (double v : m.var_max_abs) exact += v == 0;
                    e["channels_bitwise_equal"] = exact;
                    xj.push_back(e);
                    if (m.rel > worst) worst = m.rel, wk = k, wsl = sl, wchan = cfg.var_input[m.worst];
                }
            report["inputs_vs_reference"] = {{"worst_rel_absmax", num(worst)}, {"window", wk}, {"slot", wsl},
                                            {"channel", wchan}, {"per_window_slot", xj}};
        }

        // ---- output comparison and verdict
        if (ref) {
            json comp = json::array();
            std::printf("\ny vs reference y.<k> (contract metric over (G, %d); tolerance: %s)\n", V_out,
                        tols.is_null() ? "none, reported only" : (tol_path + " / " + tol_table).c_str());
            std::printf("%6s %4s %12s %12s %10s  %s\n", "window", "step", "rel_absmax", "max_abs", "x tol(y/var)", "worst variable (rel)");
            double worst_ratio = -1, worst_rel = 0;
            int wk = -1, wt = -1, nabove = 0;
            std::string wvar;
            json viol = json::array();
            for (int k = 0; k < N; ++k)
                for (int t = 0; t < T_out; ++t) {
                    const VarMetric& m = R.y[k][t];
                    const double tol = tols.is_null() ? -1 : tol_at(tols[tol_table]["y"], k, t);
                    // y_var: the worst per-variable rel_absmax, judged too when the table has it
                    const double vrel = m.var_rel[m.worst];
                    const double tolv = (tols.is_null() || !tols[tol_table].contains("y_var")) ? -1 : tol_at(tols[tol_table]["y_var"], k, t);
                    json e = to_json(m, cfg.var_output, true);
                    e["window"] = k;
                    e["output_index"] = t;
                    e["step_hours"] = ds ? ds->times().step_hours(k, t) : 0;
                    if (tol > 0) {
                        e["tolerance"] = tol;
                        e["ratio"] = num(m.rel / tol);
                        if (!(m.rel <= tol) || m.n_nan || m.n_ref_bad) {
                            ++nabove;
                            viol.push_back({{"window", k}, {"output_index", t}, {"rel_absmax", num(m.rel)}, {"limit", tol}});
                        }
                    }
                    if (tolv > 0) {
                        e["tolerance_var"] = tolv;
                        e["ratio_var"] = num(vrel / tolv);
                        if (!(vrel <= tolv)) {
                            ++nabove;
                            viol.push_back({{"window", k}, {"output_index", t}, {"variable", cfg.var_output[m.worst]},
                                            {"rel_absmax_var", num(vrel)}, {"limit_var", tolv}});
                        }
                        if (vrel / tolv > worst_ratio) worst_ratio = vrel / tolv;
                    }
                    comp.push_back(e);
                    char ratio[24] = "-";
                    if (tol > 0 && tolv > 0) std::snprintf(ratio, sizeof ratio, "%.2f/%.2f", m.rel / tol, vrel / tolv);
                    else if (tol > 0) std::snprintf(ratio, sizeof ratio, "%.2f", m.rel / tol);
                    std::printf("%6d %4d %12.3e %12.3e %10s  %s (%.3e)%s\n", k, t + 1, m.rel, m.max_abs, ratio,
                                cfg.var_output[m.worst].c_str(), m.var_rel[m.worst],
                                (m.n_nan || m.n_ref_bad) ? "  NaN!" : "");
                    if (m.rel > worst_rel) worst_rel = m.rel, wk = k, wt = t, wvar = cfg.var_output[m.worst];
                    if (tol > 0 && m.rel / tol > worst_ratio) worst_ratio = m.rel / tol;
                }
            report["y_vs_reference"] = {{"worst_rel_absmax", num(worst_rel)}, {"window", wk}, {"output_index", wt},
                                       {"variable", wvar}, {"per_window_step", comp}};
            std::printf("worst rel_absmax %.3e (window %d, output %d, worst variable %s)\n", worst_rel, wk, wt + 1, wvar.c_str());
            if (!tols.is_null()) {
                const bool pass = viol.empty();
                std::printf("worst %.2f x tolerance; %d/%d entries above it -> %s\n", worst_ratio, nabove, N * T_out, pass ? "PASS" : "FAIL");
                report["verdict"] = {{"pass", pass}, {"worst_ratio", num(worst_ratio)}, {"entries_above_tolerance", nabove},
                                     {"violations", viol}};
                if (!pass) ok = false;
            } else {
                for (int k = 0; k < N; ++k)
                    for (int t = 0; t < T_out; ++t)
                        if (R.y[k][t].n_nan || R.y[k][t].n_ref_bad) ok = false;
            }
        }

        // ---- files
        if (!out_path.empty()) {
            std::vector<varda::TensorOut> to;
            std::vector<std::string> names(R.hourly.size());
            for (size_t i = 0; i < R.hourly.size(); ++i) {
                names[i] = "y." + std::to_string(R.hours[i]);
                to.push_back({names[i], varda::DType::F32, {G, V_out}, R.hourly[i].data()});
            }
            varda::write_safetensors(out_path, to,
                {{"varda_downscale_y", "1"},
                 {"meta", json({{"out_vars", cfg.var_output}, {"date", varda::format_iso8601_utc(date0)},
                                {"checkpoint_uuid", cfg.uuid}, {"unit", "y.<hour>: state valid at date0 + hour h"}}).dump()}});
            std::printf("wrote %zu hourly states to %s\n", to.size(), out_path.c_str());
        }
        if (!dumpx_path.empty()) {
            std::vector<varda::TensorOut> to;
            std::vector<std::string> names(R.xkeep.size());
            for (size_t k = 0; k < R.xkeep.size(); ++k) {
                names[k] = "x." + std::to_string(k);
                to.push_back({names[k], varda::DType::F32, {2, G, V_in}, R.xkeep[k].data()});
            }
            varda::write_safetensors(dumpx_path, to,
                {{"varda_downscale_x", "1"},
                 {"meta", json({{"in_vars", cfg.var_input}, {"date", varda::format_iso8601_utc(date0)},
                                {"checkpoint_uuid", cfg.uuid}}).dump()}});
            std::printf("wrote %zu assembled inputs to %s\n", to.size(), dumpx_path.c_str());
        }
        report["pass"] = ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        report["error"] = e.what();
        ok = false;
        if (!json_path.empty()) std::ofstream(json_path) << report.dump(1) << "\n";
        return 1;
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << report.dump(1) << "\n";
        std::printf("report: %s\n", json_path.c_str());
    }
    return ok ? 0 : 1;
}
