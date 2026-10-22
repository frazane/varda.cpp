// Parity check of the engine against golden activations (docs/format.md §2-4).
//
//   varda-parity --model M.safetensors --check GOLDEN.safetensors=TABLE [--check ...]
//                [--report-only GOLDEN.safetensors] [--device cpu|cuda] [--precision fp32|fp16]
//                [--threads N] [--no-cuda-graph] [--tolerances FILE] [--report OUT.json]
//
// The input `in.x` comes from the first golden file (all must hold the same bytes). One
// predict_step runs with the activation hook; every contract key is compared, as it is
// produced, against each golden file: against TABLE of the tolerance file for --check, for
// information only (NaN still fails) for --report-only.
// A second predict_step runs without the hook; its output must be bitwise equal to the
// hooked run's. Exit 0 only if every key is present, NaN-free and within tolerance.
#include <cstdio>
#include <map>
#include <set>

#include "common.h"
#include "varda/varda.h"

using app::json;

namespace {

struct Check {
    std::string golden_path, table;
    bool report_only = false;
    std::unique_ptr<app::Golden> golden;
    json tol;  // the table, or null
    std::map<std::string, app::Metric> metrics;
    std::map<std::string, std::vector<int64_t>> shapes;
    std::vector<std::string> errors;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-parity --model M --check GOLDEN=TABLE [--check ...] [--report-only GOLDEN]\n"
                 "                    [--device cpu|cuda] [--precision fp32|fp16] [--threads N] [--no-cuda-graph]\n"
                 "                    [--tolerances FILE] [--report OUT.json]\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tol_path, report_path;
    std::vector<Check> checks;
    varda::Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--check") {
            Check c;
            std::string s = next();
            auto eq = s.rfind('=');
            if (eq == std::string::npos || eq + 1 == s.size()) {
                std::fprintf(stderr, "--check needs GOLDEN=TABLE (use --report-only GOLDEN for an unchecked comparison)\n");
                return 2;
            }
            c.golden_path = s.substr(0, eq);
            c.table = s.substr(eq + 1);
            checks.push_back(std::move(c));
        } else if (a == "--report-only") {
            Check c;
            c.golden_path = next();
            c.report_only = true;
            checks.push_back(std::move(c));
        } else if (a == "--device" || a == "--precision") {
            std::string v = next();
            try {
                if (a == "--device") opts.device = app::parse_device(v);
                else opts.precision = app::parse_precision(v);
            } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
        }
        else if (a == "--threads") opts.num_threads = std::stoi(next());
        else if (a == "--no-cuda-graph") opts.cuda_graph = false;
        else if (a == "--tolerances") tol_path = next();
        else if (a == "--report") report_path = next();
        else usage();
    }
    if (model_path.empty() || checks.empty()) usage();

    json report;
    int status = 0;
    try {
        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        const varda::Config& cfg = model->config();
        const auto keys = app::contract_keys(cfg.num_processor_layers);
        std::printf("model %s (uuid %s), %s %s, threads %d, load %.1f s\n", model_path.c_str(), cfg.uuid.c_str(),
                    varda::to_string(opts.device).c_str(), varda::to_string(opts.precision).c_str(),
                    omp_get_max_threads(), load_s);

        std::string used_tol;
        json tolerances;
        for (auto& c : checks) {
            c.golden = std::make_unique<app::Golden>(c.golden_path);
            if (c.golden->uuid != cfg.uuid)
                throw std::runtime_error(c.golden_path + ": golden checkpoint uuid " + c.golden->uuid +
                                         " does not match the model's " + cfg.uuid);
            if (!c.table.empty()) {
                if (tolerances.is_null()) tolerances = app::load_tolerances(tol_path, cfg.uuid, &used_tol);
                if (!tolerances.contains(c.table)) throw std::runtime_error("tolerance table '" + c.table + "' not found in " + used_tol);
                c.tol = tolerances[c.table];
            }
        }
        const varda::TensorView& x = checks[0].golden->file->peek("in.x");
        for (auto& c : checks) {
            const varda::TensorView& xi = c.golden->file->peek("in.x");
            if (xi.nbytes != x.nbytes || std::memcmp(xi.data, x.data, x.nbytes) != 0)
                throw std::runtime_error(c.golden_path + ": in.x differs from " + checks[0].golden_path);
        }
        const std::vector<int64_t> x_shape = {1, cfg.n_input_steps, cfg.n_data_nodes, cfg.n_input_vars};
        if (x.shape != x_shape) throw std::runtime_error("golden in.x has shape " + x.shape_str());

        const int64_t y_numel = cfg.n_output_steps * cfg.n_data_nodes * cfg.n_output_vars;
        std::vector<float> y_hook(y_numel), y_plain(y_numel);
        std::vector<std::string> order;
        std::map<std::string, int> seen;
        varda::ActivationHook hook = [&](const std::string& key, const float* data, const std::vector<int64_t>& shape) {
            ++seen[key];
            order.push_back(key);
            int64_t n = 1;
            for (auto s : shape) n *= s;
            for (auto& c : checks) {
                c.shapes[key] = shape;
                if (!c.golden->file->has(key)) { c.errors.push_back(key + ": not in golden file"); continue; }
                const varda::TensorView& r = c.golden->file->peek(key);
                if (r.dtype != varda::DType::F32) { c.errors.push_back(key + ": golden dtype is not F32"); continue; }
                if (r.shape != shape) {
                    c.errors.push_back(key + ": engine shape " + app::shape_str(shape) + " != golden " + r.shape_str());
                    continue;
                }
                c.metrics[key] = app::compare(data, r.as<float>(), n);
            }
        };

        t0 = app::now_s();
        model->predict_step(x.as<float>(), y_hook.data(), &hook);
        const double hook_s = app::now_s() - t0;
        t0 = app::now_s();
        model->predict_step(x.as<float>(), y_plain.data(), nullptr);
        const double plain_s = app::now_s() - t0;
        const bool hook_inert = std::memcmp(y_hook.data(), y_plain.data(), y_numel * sizeof(float)) == 0;

        // Reported keys must be exactly the contract keys, each once.
        std::vector<std::string> contract_errors;
        for (const auto& k : keys)
            if (seen[k] != 1) contract_errors.push_back(k + ": reported " + std::to_string(seen[k]) + " times");
        std::set<std::string> kset(keys.begin(), keys.end());
        for (const auto& kv : seen)
            if (!kset.count(kv.first)) contract_errors.push_back(kv.first + ": not a contract key");
        if (!hook_inert) contract_errors.push_back("output without the hook differs from the hooked output");

        report["model"] = model_path;
        report["uuid"] = cfg.uuid;
        report["device"] = varda::to_string(opts.device);
        report["precision"] = varda::to_string(opts.precision);
        report["threads"] = omp_get_max_threads();
        report["cuda_graph"] = opts.cuda_graph;
        report["load_s"] = load_s;
        report["predict_hook_s"] = hook_s;
        report["predict_plain_s"] = plain_s;
        report["hook_inert"] = hook_inert;
        report["tolerances"] = used_tol;
        report["errors"] = contract_errors;
        for (const auto& e : contract_errors) std::printf("ERROR %s\n", e.c_str());
        if (!contract_errors.empty()) status = 1;

        for (auto& c : checks) {
            std::printf("\nvs %s (tag %s)%s%s\n", c.golden_path.c_str(), c.golden->tag.c_str(),
                        c.report_only ? ", report only (unchecked)" : ", table ", c.table.c_str());
            std::printf("  %-14s %-22s %12s %12s %12s %10s %7s %s\n", "key", "shape", "absmax", "max_abs", "rel_absmax",
                        "tol", "ratio", "");
            json jc;
            jc["golden"] = c.golden_path;
            jc["tag"] = c.golden->tag;
            jc["table"] = c.table;
            jc["report_only"] = c.report_only;
            double worst_ratio = 0;
            std::string worst_key;
            for (const auto& k : keys) {
                if (!c.metrics.count(k)) continue;
                const auto& m = c.metrics[k];
                double tol = -1, ratio = 0;
                bool ok = m.n_nan == 0 && m.n_ref_bad == 0;
                if (!c.tol.is_null()) {
                    if (!c.tol.contains(k)) { c.errors.push_back(k + ": no tolerance in table"); ok = false; }
                    else {
                        tol = c.tol[k].get<double>();
                        ratio = m.rel_absmax / tol;
                        ok = ok && m.rel_absmax <= tol;
                        if (ratio > worst_ratio) { worst_ratio = ratio; worst_key = k; }
                    }
                }
                if (!ok) c.errors.push_back(k + ": rel_absmax " + std::to_string(m.rel_absmax) +
                                            (m.n_nan ? " with NaN in the engine output" : "") +
                                            (m.n_ref_bad ? " with NaN/Inf in the golden tensor" : "") +
                                            (c.report_only ? "" : " exceeds tolerance or is non-finite"));
                std::printf("  %-14s %-22s %12.4e %12.4e %12.4e %10.2e %7.3f %s\n", k.c_str(),
                            app::shape_str(c.shapes[k]).c_str(), m.absmax, m.max_abs, m.rel_absmax, tol, ratio,
                            !ok ? "FAIL" : (m.max_abs == 0 ? "bitwise" : "ok"));
                jc["keys"][k] = {{"shape", c.shapes[k]}, {"absmax", m.absmax}, {"max_abs", m.max_abs},
                                 {"rel_absmax", m.rel_absmax}, {"n_nan", m.n_nan}, {"n_ref_bad", m.n_ref_bad}, {"tol", tol}, {"ok", ok}};
            }
            jc["worst_ratio"] = worst_ratio;
            jc["worst_key"] = worst_key;
            jc["errors"] = c.errors;
            for (const auto& e : c.errors) std::printf("  ERROR %s\n", e.c_str());
            if (!c.table.empty())
                std::printf("  worst ratio to tolerance: %.3f (%s)\n", worst_ratio, worst_key.c_str());
            if (!c.errors.empty()) status = 1;
            report["checks"].push_back(jc);
        }
        std::printf("\npredict_step: %.3f s with hook, %.3f s without; hook inert: %s\n", hook_s, plain_s,
                    hook_inert ? "yes (bitwise)" : "NO");
        std::printf("%s\n", status == 0 ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        report["exception"] = e.what();
        status = 1;
    }
    report["status"] = status == 0 ? "pass" : "fail";
    if (!report_path.empty()) {
        std::ofstream f(report_path + ".tmp");
        f << report.dump(1) << "\n";
        f.close();
        std::rename((report_path + ".tmp").c_str(), report_path.c_str());
    }
    return status;
}
