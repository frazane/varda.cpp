// Builds the model input window x0 from real GRIB files (KENDA-CH1 + ECMWF open data) exactly as
// anemoi-inference does with the Hugging Face inference_opendata.yaml (docs/spec-io.md §1-3), and
// compares every stage with the reference run's dumps (docs/io-input.md).
//
//   varda-prepare --model M.safetensors --io-config varda-io.json --regrid-matrix R.safetensors
//                 --n320-template templates/ifs-levtype=sfc.grib
//                 ( --realdata DIR                    # DIR/pins.json + DIR/cache/{kenda,ecmwf/files}
//                 | --date T --kenda DIR --ifs-prev F... --ifs F... --ifs-constants F... )
//                 [--definitions D] [--allow-nans] [--threads N]
//                 [--out DIR] [--dump-stages]
//                 [--compare REF_DUMPS_DIR] [--mir-mat MIR_CACHE.mat]
//
// --out writes DIR/x0.safetensors (`x0` F32 [1, 2, G, V_in], model-input order, metadata: dates,
// in_vars, provenance) and DIR/initial_state.safetensors (`field/<name>` F32 [G], the 70 step-0
// fields at date t). --dump-stages also writes DIR/dumps/<stage>.safetensors with the reference
// dump names (source.<purpose>.<src>, cutout.<purpose>, input_tensor [2, V_in, G]) and
// regrid.safetensors (`out/<shortName>@<level>@<YYYYmmddHHMM>+<step>` F64, the decoded 16-bit
// re-encode of y = W x).
// --compare reads the reference dumps (<run>/dumps) and prints, per stage and variable, max_abs,
// the number of values that are not bitwise equal and whether the stage is bitwise (spec §5.2
// C1, C1a, C2, plus the step-0 state). It also regrids the reference's own MIR inputs
// (regrid.<k>.in.grib2) and compares with MIR's outputs. Exit status 1 if anything differs.
#include <omp.h>
#include <sys/stat.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "varda/format.h"
#include "varda/io/grib.h"
#include "varda/io/input.h"
#include "varda/io/regrid.h"

namespace {

using json = nlohmann::json;
using varda::DType;
using varda::SafeTensorsFile;
using varda::TensorOut;
using varda::TensorView;
using namespace varda::io;

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-prepare --model M --io-config J --regrid-matrix R --n320-template T\n"
                 "         (--realdata DIR | --date T --kenda DIR --ifs-prev F.. --ifs F.. --ifs-constants F..)\n"
                 "         [--definitions D] [--allow-nans] [--threads N] [--out DIR] [--dump-stages]\n"
                 "         [--compare REF_DUMPS_DIR] [--mir-mat MIR.mat]\n");
    std::exit(2);
}

std::string compact_date(const varda::DateTime& d) {  // pins.json style "2026-10-22T06:00:00"
    return varda::format_iso8601_utc(d);
}

// Fills files from a frozen real-data directory (python/reference/realdata/fetch_realdata.py).
void from_realdata(const std::string& dir, InputFiles& f) {
    std::ifstream in(dir + "/pins.json");
    if (!in) throw std::runtime_error("cannot read " + dir + "/pins.json");
    const json pins = json::parse(in);
    if (f.date.empty()) f.date = pins.at("date").get<std::string>();
    f.kenda_dir = dir + "/cache/kenda";
    const std::string files = dir + "/cache/ecmwf/files/";
    const varda::DateTime t = parse_input_date(f.date);
    auto file_of = [&](const std::string& init, long step) {
        const std::string key = init + "+" + std::to_string(step) + "h";
        return files + pins.at("ecmwf_files").at(key).at("file").get<std::string>();
    };
    for (int s = 0; s < 2; ++s) {
        const varda::DateTime d = varda::add_hours(t, -6 * (1 - s));
        const json& e = pins.at("input_dates").at(compact_date(d));
        (s == 0 ? f.ifs_files_prev : f.ifs_files_t).push_back(file_of(e.at("init"), e.at("step")));
    }
    const json& c = pins.at("constant_forcings_run");
    f.ifs_constant_files.push_back(file_of(c.at("init"), c.at("step")));
}

void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i <= p.size(); ++i) {
        if (i == p.size() || p[i] == '/') {
            if (!cur.empty()) ::mkdir(cur.c_str(), 0755);
        }
        if (i < p.size()) cur += p[i];
    }
}

json log_json(const InputState& st) {
    json j;
    j["log"] = st.log;
    j["seconds"] = st.seconds;
    return j;
}

void write_state(const std::string& path, const std::string& name, const FieldState& s, bool coords) {
    std::vector<TensorOut> t;
    for (size_t i = 0; i < s.names.size(); ++i)
        t.push_back({"field/" + s.names[i], DType::F32, {int64_t(s.n_dates), int64_t(s.n_points)}, s.values[i].data()});
    if (coords && !s.latitudes.empty()) {
        t.push_back({"latitudes", DType::F64, {int64_t(s.latitudes.size())}, s.latitudes.data()});
        t.push_back({"longitudes", DType::F64, {int64_t(s.longitudes.size())}, s.longitudes.data()});
    }
    json meta = {{"names", s.names}};
    varda::write_safetensors(path, t, {{"varda_prepare", "1"}, {"name", name}, {"meta", meta.dump()}});
}

// ----------------------------------------------------------------------------------------- compare

struct Diff {
    double max_abs = 0;
    int64_t n_differ = 0, n = 0;
    bool bitwise() const { return n_differ == 0; }
};

template <class A, class R>
Diff diff_arrays(const A* a, const R* r, int64_t n, int64_t stride_a = 1, int64_t stride_r = 1) {
    double mx = 0;
    int64_t bad = 0;
#pragma omp parallel for reduction(max : mx) reduction(+ : bad) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const A av = a[i * stride_a];
        const R rv = r[i * stride_r];
        const bool same = std::memcmp(&av, &rv, sizeof(A)) == 0 && sizeof(A) == sizeof(R);
        if (!same) {
            ++bad;
            const double d = std::fabs(double(av) - double(rv));
            if (std::isnan(d)) mx = INFINITY;
            else if (d > mx) mx = d;
        }
    }
    return {mx, bad, n};
}

struct Report {
    int64_t stages_bad = 0;
    void stage(const std::string& name, const std::vector<std::pair<std::string, Diff>>& rows, bool verbose = true) {
        int64_t bad = 0, values = 0;
        double mx = 0;
        for (const auto& [k, d] : rows) {
            bad += d.n_differ;
            values += d.n;
            mx = std::max(mx, d.max_abs);
        }
        std::printf("\n[%s] %zu arrays, %" PRId64 " values: %s (max_abs %.3g, %" PRId64 " differ)\n", name.c_str(),
                    rows.size(), values, bad ? "NOT bitwise" : "bitwise", mx, bad);
        if (verbose)
            for (const auto& [k, d] : rows)
                std::printf("  %-28s max_abs %-10.3g differ %-9" PRId64 " %s\n", k.c_str(), d.max_abs, d.n_differ,
                            d.bitwise() ? "bitwise" : "DIFFERS");
        stages_bad += bad > 0;
    }
    void fail(const std::string& name, const std::string& why) {
        std::printf("\n[%s] FAIL: %s\n", name.c_str(), why.c_str());
        ++stages_bad;
    }
};

// FNV-1a 64 of the bytes (determinism checks across thread counts).
uint64_t fnv1a(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

bool exists(const std::string& p) {
    struct stat s;
    return ::stat(p.c_str(), &s) == 0;
}

void compare_state(Report& rep, const std::string& dir, const std::string& name, const FieldState& mine,
                   bool coords) {
    const std::string path = dir + "/" + name + ".safetensors";
    if (!exists(path)) return rep.fail(name, "no reference dump " + path);
    SafeTensorsFile ref(path);
    std::vector<std::pair<std::string, Diff>> rows;
    std::set<std::string> ref_names;
    for (const auto& n : ref.names())
        if (n.rfind("field/", 0) == 0) ref_names.insert(n.substr(6));
    std::set<std::string> my_names(mine.names.begin(), mine.names.end());
    if (ref_names != my_names) {
        std::string m;
        for (const auto& n : ref_names)
            if (!my_names.count(n)) m += " -" + n;
        for (const auto& n : my_names)
            if (!ref_names.count(n)) m += " +" + n;
        rep.fail(name, "field names differ:" + m);
    }
    // the reference's names order (metadata) vs ours
    const json meta = json::parse(ref.metadata("meta"));
    if (meta.contains("names") && meta["names"].get<std::vector<std::string>>() != mine.names)
        std::printf("  note: %s field order differs from the reference's\n", name.c_str());
    for (size_t i = 0; i < mine.names.size(); ++i) {
        const std::string key = "field/" + mine.names[i];
        if (!ref.has(key)) continue;
        const TensorView& t = ref.peek(key);
        if (t.dtype != DType::F32 || t.numel() != int64_t(mine.values[i].size())) {
            rep.fail(name, key + " is " + t.shape_str() + " " + varda::dtype_name(t.dtype));
            continue;
        }
        rows.push_back({mine.names[i], diff_arrays(mine.values[i].data(), t.as<float>(), t.numel())});
    }
    if (coords && !mine.latitudes.empty() && ref.has("latitudes")) {
        const TensorView& la = ref.peek("latitudes");
        const TensorView& lo = ref.peek("longitudes");
        if (la.numel() == int64_t(mine.latitudes.size())) {
            rows.push_back({"(latitudes)", diff_arrays(mine.latitudes.data(), la.as<double>(), la.numel())});
            rows.push_back({"(longitudes)", diff_arrays(mine.longitudes.data(), lo.as<double>(), lo.numel())});
        }
    }
    rep.stage(name, rows);
}

void compare_x0(Report& rep, const std::string& dir, const InputState& st, const varda::Config& cfg,
                const std::vector<double>& lat, const std::vector<double>& lon) {
    const std::string path = dir + "/input_tensor.safetensors";
    if (!exists(path)) return rep.fail("input_tensor", "no reference dump " + path);
    SafeTensorsFile ref(path);
    const TensorView& x = ref.peek("x");
    const int64_t T = st.T, G = st.G, V = st.V_in;
    if (x.dtype != DType::F32 || x.shape != std::vector<int64_t>{T, V, G})
        return rep.fail("input_tensor", "x is " + x.shape_str());
    const json meta = json::parse(ref.metadata("meta"));
    if (meta.at("in_vars").get<std::vector<std::string>>() != cfg.var_input)
        return rep.fail("input_tensor", "in_vars differ from the model's input order");
    std::vector<std::pair<std::string, Diff>> rows;
    for (int64_t i = 0; i < V; ++i) {
        Diff d;
        for (int64_t s = 0; s < T; ++s) {
            const Diff e = diff_arrays(st.x0.data() + s * G * V + i, x.as<float>() + (s * V + i) * G, G, V, 1);
            d.max_abs = std::max(d.max_abs, e.max_abs);
            d.n_differ += e.n_differ;
            d.n += e.n;
        }
        rows.push_back({cfg.var_input[size_t(i)], d});
    }
    rep.stage("input_tensor (C2: x0 vs the reference input tensor, per column)", rows);
    // coordinates of the merged state vs grid.*
    const std::string cpath = dir + "/cutout.prognostics.safetensors";
    if (exists(cpath)) {
        SafeTensorsFile c(cpath);
        const TensorView& la = c.peek("latitudes");
        const TensorView& lo = c.peek("longitudes");
        rep.stage("coordinates (cutout state vs grid.*)",
                  {{"latitudes", diff_arrays(lat.data(), la.as<double>(), la.numel())},
                   {"longitudes", diff_arrays(lon.data(), lo.as<double>(), lo.numel())}});
    }
}

void compare_initial(Report& rep, const std::string& dir, const InputState& st) {
    const std::string path = dir + "/state0.pre.safetensors";
    if (!exists(path)) return rep.fail("state0.pre", "no reference dump " + path);
    SafeTensorsFile ref(path);
    const json meta = json::parse(ref.metadata("meta"));
    std::vector<std::pair<std::string, Diff>> rows;
    if (meta.contains("names") && meta["names"].get<std::vector<std::string>>() != st.initial_names)
        rep.fail("state0.pre", "field names/order differ from initial_state_fields");
    for (size_t i = 0; i < st.initial_names.size(); ++i) {
        const std::string key = "field/" + st.initial_names[i];
        if (!ref.has(key)) {
            rep.fail("state0.pre", "no " + key);
            continue;
        }
        const TensorView& t = ref.peek(key);
        // [G] (reduced to the last date) or [1, G]
        if (t.numel() != st.G) {
            rep.fail("state0.pre", key + " is " + t.shape_str());
            continue;
        }
        rows.push_back({st.initial_names[i], diff_arrays(st.initial_values[i].data(), t.as<float>(), st.G)});
    }
    rep.stage("state0.pre (step-0 initial state, date t)", rows, false);
}

void compare_regrid_stage(Report& rep, const std::string& dir, const InputState& st) {
    std::vector<std::pair<std::string, Diff>> rows;
    int matched = 0;
    for (int k = 0;; ++k) {
        const std::string path = dir + "/regrid." + std::to_string(k) + ".safetensors";
        if (!exists(path)) break;
        SafeTensorsFile ref(path);
        const json meta = json::parse(ref.metadata("meta"));
        const auto& in = meta.at("in");
        const auto names = meta.at("out_names").get<std::vector<std::string>>();
        for (size_t i = 0; i < names.size(); ++i) {
            char key[200];
            std::snprintf(key, sizeof key, "%s@%08ld%04ld+%ld", names[i].c_str(), in[i].at("dataDate").get<long>(),
                          in[i].at("dataTime").get<long>(), in[i].at("step").get<long>());
            auto it = st.stages.regrid.find(key);
            if (it == st.stages.regrid.end()) continue;  // a field the model does not use
            const TensorView& t = ref.peek("out/" + names[i]);
            rows.push_back({std::string("regrid.") + std::to_string(k) + " " + key,
                            diff_arrays(it->second.data(), t.as<double>(), t.numel())});
            ++matched;
        }
    }
    if (!matched) return rep.fail("regrid", "no reference regrid.<k> dumps matched");
    rep.stage("regrid (C1: decoded 16-bit N320 values, double, vs MIR's output)", rows, false);
}

// The reference's own MIR inputs through our regridder, vs MIR's output messages.
void compare_regrid_isolated(Report& rep, const std::string& dir, const Regridder& rg) {
    std::vector<std::pair<std::string, Diff>> rows;
    int64_t enc_meta_bad = 0;
    for (int k = 0;; ++k) {
        const std::string in = dir + "/regrid." + std::to_string(k) + ".in.grib2";
        const std::string out = dir + "/regrid." + std::to_string(k) + ".out.grib2";
        if (!exists(in) || !exists(out)) break;
        const auto ins = read_messages(in);
        const auto outs = read_messages(out);
        if (ins.size() != outs.size()) {
            rep.fail("regrid (isolated)", in + ": message counts differ");
            continue;
        }
        std::vector<std::pair<std::string, Diff>> local(ins.size() * 2);
        std::vector<int> meta_bad(ins.size(), 0);
        std::string err;
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < int64_t(ins.size()); ++i) {
            try {
                std::vector<float> f;
                std::vector<double> d;
                rg.regrid(ins[size_t(i)], &f, &d);
                const auto rf = outs[size_t(i)].get_float_array("values");
                const auto rd = outs[size_t(i)].get_double_array("values");
                const std::string nm = "regrid." + std::to_string(k) + " " + ins[size_t(i)].get_string("shortName") +
                                       "@" + std::to_string(ins[size_t(i)].get_long("level"));
                local[size_t(2 * i)] = {nm + " f64", diff_arrays(d.data(), rd.data(), int64_t(rd.size()))};
                local[size_t(2 * i + 1)] = {nm + " f32", diff_arrays(f.data(), rf.data(), int64_t(rf.size()))};
                std::vector<double> y(rd.size());
                // encoding parameters of our message vs MIR's
                std::vector<double> x = ins[size_t(i)].get_double_array("values");
                rg.matrix().apply(x.data(), y.data());
                const GribHandle h = rg.encode(y);
                for (const char* key : {"bitsPerValue", "binaryScaleFactor", "decimalScaleFactor"})
                    meta_bad[size_t(i)] += h.get_long(key) != outs[size_t(i)].get_long(key);
                meta_bad[size_t(i)] += h.get_double("referenceValue") != outs[size_t(i)].get_double("referenceValue");
            } catch (const std::exception& e) {
#pragma omp critical
                err = e.what();
            }
        }
        if (!err.empty()) rep.fail("regrid (isolated)", err);
        for (auto& r : local) rows.push_back(std::move(r));
        for (int b : meta_bad) enc_meta_bad += b;
    }
    if (rows.empty()) return rep.fail("regrid (isolated)", "no regrid.<k>.in/out.grib2 in " + dir);
    rep.stage("regrid isolated (MIR's input messages -> ours, vs MIR's output messages; every field of every call)",
              rows, false);
    std::printf("  encoding keys (bitsPerValue, binaryScaleFactor, decimalScaleFactor, referenceValue) differing: %" PRId64
                "\n", enc_meta_bad);
    if (enc_meta_bad) ++rep.stages_bad;
}

}  // namespace

int main(int argc, char** argv) {
    InputFiles files;
    std::string model_path, realdata, out_dir, compare_dir, mir_mat;
    bool dump_stages = false;
    int threads = 0;
    if (const char* d = std::getenv("VARDA_COSMO_DEFINITIONS")) files.cosmo_definitions = d;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--io-config") files.io_config = next();
        else if (a == "--regrid-matrix") files.regrid_matrix = next();
        else if (a == "--n320-template") files.n320_template = next();
        else if (a == "--realdata") realdata = next();
        else if (a == "--date") files.date = next();
        else if (a == "--kenda") files.kenda_dir = next();
        else if (a == "--ifs-prev") files.ifs_files_prev.push_back(next());
        else if (a == "--ifs") files.ifs_files_t.push_back(next());
        else if (a == "--ifs-constants") files.ifs_constant_files.push_back(next());
        else if (a == "--definitions") files.cosmo_definitions = next();
        else if (a == "--allow-nans") files.allow_nans = true;
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--out") out_dir = next();
        else if (a == "--dump-stages") dump_stages = true;
        else if (a == "--compare") compare_dir = next();
        else if (a == "--mir-mat") mir_mat = next();
        else usage();
    }
    if (model_path.empty() || files.io_config.empty() || files.regrid_matrix.empty() || files.n320_template.empty())
        usage();
    if (threads > 0) omp_set_num_threads(threads);
    try {
        if (!realdata.empty()) from_realdata(realdata, files);
        if (files.date.empty() || files.kenda_dir.empty() || files.ifs_files_prev.empty() || files.ifs_files_t.empty() ||
            files.ifs_constant_files.empty())
            usage();
        files.keep_stages = dump_stages || !compare_dir.empty();

        double t0 = app::now_s();
        const InputModel model = InputModel::load(model_path);
        std::printf("model %s (uuid %s): G %" PRId64 ", %d inputs, cutout", model_path.c_str(), model.config.uuid.c_str(),
                    model.config.n_data_nodes, model.config.n_input_vars);
        for (const auto& c : model.cutout) std::printf(" %s[%zu]", c.name.c_str(), c.mask.size());
        std::printf("; %d threads\n", omp_get_max_threads());

        t0 = app::now_s();
        InputState st = build_input(model, files);
        const double build_s = app::now_s() - t0;
        for (const auto& l : st.log) std::printf("  %s\n", l.c_str());
        std::printf("prepared x0 (1, %" PRId64 ", %" PRId64 ", %" PRId64 ") for %s in %.2f s:", st.T, st.G, st.V_in,
                    varda::format_iso8601_utc(st.date0).c_str(), build_s);
        for (const auto& [k, v] : st.seconds) std::printf(" %s %.2f s", k.c_str(), v);
        std::printf("\n");
        {
            uint64_t hi = 1469598103934665603ull;
            for (const auto& v : st.initial_values) hi ^= fnv1a(v.data(), v.size() * 4) + 0x9e3779b97f4a7c15ull + (hi << 6);
            std::printf("x0 fnv1a %016" PRIx64 ", initial state fnv1a %016" PRIx64 "\n",
                        fnv1a(st.x0.data(), st.x0.size() * 4), hi);
        }

        if (!out_dir.empty()) {
            mkdirs(out_dir);
            json meta = {{"date", varda::format_iso8601_utc(st.date0)},
                         {"in_vars", model.config.var_input},
                         {"checkpoint_uuid", model.config.uuid},
                         {"provenance", log_json(st)}};
            std::vector<std::string> ds;
            for (const auto& d : st.dates) ds.push_back(varda::format_iso8601_utc(d));
            meta["dates"] = ds;
            varda::write_safetensors(out_dir + "/x0.safetensors",
                                     {{"x0", DType::F32, {1, st.T, st.G, st.V_in}, st.x0.data()}},
                                     {{"varda_realdata_input", "1"}, {"meta", meta.dump()}});
            std::vector<TensorOut> init;
            for (size_t i = 0; i < st.initial_names.size(); ++i)
                init.push_back({"field/" + st.initial_names[i], DType::F32, {st.G}, st.initial_values[i].data()});
            varda::write_safetensors(out_dir + "/initial_state.safetensors", init,
                                     {{"varda_realdata_initial_state", "1"},
                                      {"meta", json({{"names", st.initial_names},
                                                     {"date", varda::format_iso8601_utc(st.date0)}}).dump()}});
            std::printf("wrote %s/x0.safetensors, %s/initial_state.safetensors\n", out_dir.c_str(), out_dir.c_str());
            if (dump_stages) {
                const std::string dd = out_dir + "/dumps";
                mkdirs(dd);
                for (const auto& [name, s] : st.stages.states) write_state(dd + "/" + name + ".safetensors", name, s, true);
                // input_tensor in the reference's numpy layout [T, V_in, G]
                std::vector<float> xt(st.x0.size());
                for (int64_t s = 0; s < st.T; ++s)
                    for (int64_t g = 0; g < st.G; ++g)
                        for (int64_t i = 0; i < st.V_in; ++i)
                            xt[size_t((s * st.V_in + i) * st.G + g)] = st.x0[size_t((s * st.G + g) * st.V_in + i)];
                varda::write_safetensors(dd + "/input_tensor.safetensors",
                                         {{"x", DType::F32, {st.T, st.V_in, st.G}, xt.data()}},
                                         {{"varda_prepare", "1"},
                                          {"meta", json({{"in_vars", model.config.var_input}}).dump()}});
                std::vector<TensorOut> rg;
                for (const auto& [k, v] : st.stages.regrid)
                    rg.push_back({"out/" + k, DType::F64, {int64_t(v.size())}, v.data()});
                varda::write_safetensors(dd + "/regrid.safetensors", rg, {{"varda_prepare", "1"}});
                std::printf("wrote %s/*.safetensors\n", dd.c_str());
            }
        }

        if (!compare_dir.empty()) {
            Report rep;
            for (const auto& [name, s] : st.stages.states) compare_state(rep, compare_dir, name, s, true);
            compare_regrid_stage(rep, compare_dir, st);
            compare_x0(rep, compare_dir, st, model.config, model.latitudes, model.longitudes);
            compare_initial(rep, compare_dir, st);
            // regrid in isolation, and the matrix itself (C1a)
            const RegridMatrix m =
                files.regrid_matrix.size() > 4 && files.regrid_matrix.substr(files.regrid_matrix.size() - 4) == ".mat"
                    ? RegridMatrix::load_mir_mat(files.regrid_matrix, RegridGeometry::ifs_open_data_0p25())
                    : RegridMatrix::load(files.regrid_matrix);
            Regridder rg(m, read_first_message(files.n320_template));
            compare_regrid_isolated(rep, compare_dir, rg);
            if (!mir_mat.empty()) {
                const RegridMatrix mm = RegridMatrix::load_mir_mat(mir_mat, m.source());
                const bool same = m.same_matrix(mm);
                std::printf("\n[C1a regrid matrix] %s vs MIR cache %s: %" PRId64 " x %" PRId64 ", nnz %" PRId64 ": %s\n",
                            files.regrid_matrix.c_str(), mir_mat.c_str(), mm.n_target(), mm.n_source(), mm.nnz(),
                            same ? "identical (indptr, stored indices, weights bitwise)" : "DIFFERENT");
                if (!same) ++rep.stages_bad;
            }
            std::printf("\ncompare: %s (%" PRId64 " stage(s) not bitwise or failed)\n",
                        rep.stages_bad ? "DIFFERENCES" : "every stage bitwise", rep.stages_bad);
            return rep.stages_bad ? 1 : 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "varda-prepare: %s\n", e.what());
        return 1;
    }
    return 0;
}
