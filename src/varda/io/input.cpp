#include "varda/io/input.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <omp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "nlohmann/json.hpp"
#include "varda/format.h"
#include "varda/io/grib.h"
#include "varda/io/io_config.h"
#include "varda/io/regrid.h"
#include "varda/io/vertical.h"

namespace varda::io {

// ------------------------------------------------------------------------------------------ FieldState

bool FieldState::has(const std::string& name) const {
    return std::find(names.begin(), names.end(), name) != names.end();
}

const std::vector<float>& FieldState::at(const std::string& name) const {
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == name) return values[i];
    throw std::runtime_error("state has no field '" + name + "'");
}

std::vector<float>& FieldState::at(const std::string& name) {
    return const_cast<std::vector<float>&>(static_cast<const FieldState&>(*this).at(name));
}

std::vector<float>& FieldState::add(const std::string& name) {
    if (has(name)) throw std::runtime_error("duplicate field '" + name + "' (anemoi: overlapping names are an error)");
    names.push_back(name);
    values.emplace_back(n_dates * n_points, std::numeric_limits<float>::quiet_NaN());
    return values.back();
}

// ------------------------------------------------------------------------------------------ InputModel

InputModel InputModel::from(const Model& m) {
    InputModel r;
    r.config = m.config();
    r.latitudes = m.latitudes();
    r.longitudes = m.longitudes();
    r.cutout = m.cutout();
    return r;
}

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Reads the header and the grid.* tensors of a converted file into the page cache with parallel
// reads, before SafeTensorsFile maps it. Best effort: any problem is left to SafeTensorsFile.
void warm_grid_tensors(const std::string& path) {
    try {
        char b[8];
        if (read_file_range(path, 0, 8, b, 1) != 8) return;
        uint64_t n = 0;
        for (int i = 7; i >= 0; --i) n = (n << 8) | uint8_t(b[i]);
        if (n == 0 || n > (uint64_t(1) << 30)) return;
        std::string h(size_t(n), '\0');
        read_file_range(path, 8, n, h.data(), 4, size_t(1) << 20);
        const nlohmann::json j = nlohmann::json::parse(h);
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto& [k, v] : j.items()) {
            if (k.rfind("grid.", 0) != 0) continue;
            const auto& o = v.at("data_offsets");
            const uint64_t a = o.at(0).get<uint64_t>(), e = o.at(1).get<uint64_t>();
            if (e > a) ranges.push_back({8 + n + a, e - a});
        }
        warm_file_cache(path, ranges, 8);
    } catch (const std::exception&) {
    }
}

}  // namespace

InputModel InputModel::load(const std::string& path) {
    const double t0 = now_s();
    warm_grid_tensors(path);
    SafeTensorsFile f(path);
    InputModel r;
    r.config = parse_config(f.metadata("config"));
    const std::vector<int64_t> shape{r.config.n_data_nodes};
    const TensorView& la = f.take("grid.latitudes", DType::F64, shape);
    const TensorView& lo = f.take("grid.longitudes", DType::F64, shape);
    r.latitudes.assign(la.as<double>(), la.as<double>() + la.numel());
    r.longitudes.assign(lo.as<double>(), lo.as<double>() + lo.numel());
    if (!f.metadata().count("cutout")) throw std::runtime_error(path + ": no cutout masks (docs/format.md §1.1)");
    for (const auto& name : nlohmann::json::parse(f.metadata("cutout"))) {
        Model::CutoutSource src{name.get<std::string>(), {}};
        const TensorView& t = f.take("grid." + src.name + ".cutout_mask");
        if (t.dtype != DType::U8 || t.shape.size() != 1)
            throw std::runtime_error(path + ": grid." + src.name + ".cutout_mask must be U8 [n]");
        src.mask.assign(t.as<uint8_t>(), t.as<uint8_t>() + t.numel());
        r.cutout.push_back(std::move(src));
    }
    r.load_seconds = now_s() - t0;
    return r;
}

// --------------------------------------------------------------------------------------------- helpers

DateTime parse_input_date(const std::string& s) {
    if (s.size() == 13 && s[10] == 'T') return parse_iso8601_utc(s + ":00");  // YYYY-MM-DDTHH
    return parse_iso8601_utc(s);
}

const std::vector<int>& kenda_interpolate_levels() {
    static const std::vector<int> v = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000};
    return v;
}
const std::vector<int>& kenda_extrapolate_levels() {
    static const std::vector<int> v = {500, 600, 700, 850, 925, 1000};
    return v;
}

namespace {

constexpr double kG = 9.80665;  // earthkit-meteo / geopotential_from_height

long yyyymmdd(const DateTime& d) { return long(d.year) * 10000 + d.month * 100 + d.day; }
long hhmm(const DateTime& d) { return long(d.hour) * 100 + d.minute; }

std::string kenda_stamp(const DateTime& d) {
    char b[32];
    std::snprintf(b, sizeof b, "%04d%02d%02dT%02d%02d%02d", d.year, d.month, d.day, d.hour, d.minute, d.second);
    return b;
}

std::string iso(const DateTime& d) { return format_iso8601_utc(d); }

bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}

void check_valid_at(const GribHandle& h, const DateTime& d, const std::string& what) {
    const long vd = h.get_long("validityDate"), vt = h.get_long("validityTime");
    if (vd != yyyymmdd(d) || vt != hhmm(d))
        throw std::runtime_error(what + ": valid at " + std::to_string(vd) + " " + std::to_string(vt) + ", expected " +
                                 iso(d));
}

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    try {
        return std::max(1, std::stoi(v));
    } catch (const std::exception&) {
        return dflt;
    }
}

// Tuning (A100 node, 32 cores, Lustre; docs: the integrator's): VARDA_INPUT_IO_THREADS reader
// threads of the GRIB prefetch pool (I/O-bound, mostly blocked on the file system), default 16;
// VARDA_INPUT_IFS_THREADS OpenMP threads of the IFS branch, default a quarter of the threads
// (4..16): it has slack, and its handle clones contend with the parser for ecCodes' global lock,
// which paces the KENDA branch. VARDA_INPUT_TRACE adds per-file and per-branch timelines to the log.
int input_io_threads() { return env_int("VARDA_INPUT_IO_THREADS", 16); }
int input_ifs_threads() {
    const int n = omp_get_max_threads();
    return env_int("VARDA_INPUT_IFS_THREADS", std::min(n, std::max(4, std::min(16, n / 4))));
}

// Runs f(i) for i in [0, n) in parallel; rethrows the first exception.
template <class F>
void parallel_for(int64_t n, F f) {
    std::string err;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < n; ++i) {
        try {
            f(i);
        } catch (const std::exception& e) {
#pragma omp critical(varda_io_input_err)
            if (err.empty()) err = e.what();
        }
    }
    if (!err.empty()) throw std::runtime_error(err);
}

// The messages of a GRIB file, from the prefetch pool when there is one (same handles as
// read_messages), with the time spent waiting for it.
struct GribReader {
    GribPrefetch* pool = nullptr;
    double wait_s = 0;
    std::shared_ptr<const std::vector<GribHandle>> operator()(const std::string& path) {
        const double t0 = now_s();
        auto r = pool ? pool->take(path) : std::make_shared<const std::vector<GribHandle>>(read_messages(path));
        wait_s += now_s() - t0;
        return r;
    }
};

// A float buffer whose pages are first touched by the parallel loop that fills it.
using FloatBuf = std::vector<float, DefaultInitAllocator<float>>;

// The model-level messages of one KENDA file, sorted by level: [n_levels][N] float32.
FloatBuf read_levels(GribReader& rd, const std::string& path, const std::string& short_name,
                               const std::string& type_of_level, long n_levels, size_t N, const DateTime* valid,
                               std::vector<std::string>& log) {
    const auto msgs = rd(path);
    std::vector<const GribHandle*> by_level(size_t(n_levels), nullptr);
    for (const auto& h : *msgs) {  // serial: ecCodes serialises key lookups internally
        const std::string sn = h.get_string("shortName");
        if (sn != short_name) throw std::runtime_error(path + ": message shortName " + sn + ", expected " + short_name);
        if (h.get_string("typeOfLevel") != type_of_level)
            throw std::runtime_error(path + ": typeOfLevel " + h.get_string("typeOfLevel") + ", expected " + type_of_level);
        const long lev = h.get_long("level");
        if (lev < 1 || lev > n_levels || by_level[size_t(lev - 1)])
            throw std::runtime_error(path + ": unexpected or duplicate level " + std::to_string(lev));
        if (valid) check_valid_at(h, *valid, path);
        if (h.get_long("bitmapPresent")) throw std::runtime_error(path + ": unexpected bitmap");
        if (h.size("values") != N) throw std::runtime_error(path + ": " + std::to_string(h.size("values")) + " values");
        by_level[size_t(lev - 1)] = &h;
    }
    for (long k = 0; k < n_levels; ++k)
        if (!by_level[size_t(k)]) throw std::runtime_error(path + ": level " + std::to_string(k + 1) + " missing");
    FloatBuf out(size_t(n_levels) * N);
    parallel_for(n_levels, [&](int64_t k) {  // decoded in place (codes_get_float_array)
        by_level[size_t(k)]->get_float_array("values", out.data() + k * int64_t(N), N);
    });
    log.push_back("read " + path + ": " + short_name + " " + type_of_level + " levels 1.." + std::to_string(n_levels));
    return out;
}

// The single message with `short_name` in a file (others ignored).
const GribHandle& one_message(const std::vector<GribHandle>& msgs, const std::string& short_name,
                              const std::string& path) {
    const GribHandle* found = nullptr;
    for (const auto& h : msgs)
        if (h.get_string("shortName") == short_name) {
            if (found) throw std::runtime_error(path + ": more than one " + short_name + " message");
            found = &h;
        }
    if (!found) throw std::runtime_error(path + ": no " + short_name + " message");
    if (found->get_long("bitmapPresent")) throw std::runtime_error(path + ": " + short_name + " has a bitmap");
    return *found;
}

std::vector<float>& slot_field(FieldState& st, const std::string& name) {
    return st.has(name) ? st.at(name) : st.add(name);
}

// Adds the fields of `names` not yet in `st` (in that order), NaN-filled as FieldState::add does,
// the fills in parallel.
void add_fields(FieldState& st, const std::vector<std::string>& names) {
    std::vector<std::string> fresh;
    for (const auto& n : names)
        if (!st.has(n) && std::find(fresh.begin(), fresh.end(), n) == fresh.end()) fresh.push_back(n);
    std::vector<std::vector<float>> v(fresh.size());
    const size_t n = st.n_dates * st.n_points;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < int64_t(fresh.size()); ++i) v[size_t(i)].assign(n, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < fresh.size(); ++i) {
        st.names.push_back(fresh[i]);
        st.values.push_back(std::move(v[i]));
    }
}

void put_slot(FieldState& st, const std::string& name, size_t slot, const std::vector<float>& v) {
    if (v.size() != st.n_points)
        throw std::runtime_error(name + ": " + std::to_string(v.size()) + " values, expected " +
                                 std::to_string(st.n_points));
    std::vector<float>& f = slot_field(st, name);
    std::copy(v.begin(), v.end(), f.begin() + int64_t(slot * st.n_points));
}

// The model's retrieved variables of one category, in variables_metadata order.
struct Retrieved {
    std::vector<const IoVariable*> prognostics, constants;
    std::vector<std::string> computed;  // computed forcings among the model inputs
};

Retrieved retrieved_variables(const IoConfig& io, const Config& cfg) {
    Retrieved r;
    std::set<std::string> inputs(cfg.var_input.begin(), cfg.var_input.end());
    std::set<std::string> prog_in;
    for (int i : cfg.residual_input) prog_in.insert(cfg.var_input.at(size_t(i)));
    for (const auto& v : io.variables) {
        if (!v.in_model || !inputs.count(v.name)) continue;
        if (v.computed_forcing) {
            if (!is_forcing_name(v.name)) throw std::runtime_error("unknown computed forcing " + v.name);
            r.computed.push_back(v.name);
        } else if (prog_in.count(v.name)) {
            r.prognostics.push_back(&v);
        } else if (v.constant_in_time) {
            r.constants.push_back(&v);
        } else {
            throw std::runtime_error("input " + v.name + " is neither prognostic, constant nor computed");
        }
    }
    if (r.prognostics.size() + r.constants.size() + r.computed.size() != cfg.var_input.size())
        throw std::runtime_error("io config variables do not cover the model inputs (" +
                                 std::to_string(r.prognostics.size()) + " + " + std::to_string(r.constants.size()) +
                                 " + " + std::to_string(r.computed.size()) + " != " +
                                 std::to_string(cfg.var_input.size()) + ")");
    return r;
}

// --------------------------------------------------------------------------------------------- KENDA

std::string kenda_horizontal_path(const std::string& dir) {
    return dir + "/constants/horizontal_constants_icon-ch1-eps.grib2";
}
std::string kenda_vertical_path(const std::string& dir) {
    return dir + "/constants/vertical_constants_icon-ch1-eps.grib2";
}
std::string kenda_raw_dir(const std::string& dir, const DateTime& d) { return dir + "/raw/" + kenda_stamp(d); }

struct KendaConstants {
    std::vector<float> hsurf;       // float decode
    std::vector<double> hsurf_d;    // double decode (geopotential_from_height)
    std::shared_ptr<const std::vector<GribHandle>> horizontal;
    std::string horizontal_path, vertical_path;

    // HHL [81][N], read on first use (its log line keeps its place, after the horizontal one).
    const FloatBuf& hhl(GribReader& rd, std::vector<std::string>& log) {
        if (!hhl_read_) {
            std::vector<std::string> l;
            hhl_ = read_levels(rd, vertical_path, "HHL", "generalVertical", 81, hsurf.size(), nullptr, l);
            log.insert(log.begin() + int64_t(hhl_log_at_), l.begin(), l.end());
            hhl_read_ = true;
        }
        return hhl_;
    }
    size_t hhl_log_at_ = 0;

private:
    FloatBuf hhl_;
    bool hhl_read_ = false;
};

KendaConstants read_kenda_constants(GribReader& rd, const std::string& dir, size_t N, std::vector<std::string>& log) {
    KendaConstants c;
    c.horizontal_path = kenda_horizontal_path(dir);
    c.vertical_path = kenda_vertical_path(dir);
    c.horizontal = rd(c.horizontal_path);
    const GribHandle& hs = one_message(*c.horizontal, "HSURF", c.horizontal_path);
    c.hsurf = hs.get_float_array("values");
    c.hsurf_d = hs.get_double_array("values");
    if (c.hsurf.size() != N) throw std::runtime_error(c.horizontal_path + ": HSURF size");
    log.push_back("read " + c.horizontal_path + ": " + std::to_string(c.horizontal->size()) + " messages");
    c.hhl_log_at_ = log.size();
    return c;
}

// What the KENDA requests ask for (spec §1.3): pressure-level params and the target levels, surface params.
struct KendaRequest {
    std::vector<std::string> pl_params, sfc_params;
    std::vector<int> targets;  // the configured levels the model needs, ascending
};

KendaRequest kenda_request(const std::vector<const IoVariable*>& progs) {
    KendaRequest r;
    std::vector<std::string>& pl_params = r.pl_params;
    std::vector<std::string>& sfc_params = r.sfc_params;
    std::vector<int> levels;
    for (const IoVariable* v : progs) {
        const std::string p = v->param();
        const Scalar* lt = v->mars_get("levtype");
        if (!lt || lt->kind != Scalar::Str) throw std::runtime_error(v->name + ": no mars levtype");
        if (lt->s == "pl") {
            const Scalar* l = v->mars_get("levelist");
            if (!l || l->kind != Scalar::Int) throw std::runtime_error(v->name + ": pl without levelist");
            if (std::find(pl_params.begin(), pl_params.end(), p) == pl_params.end()) pl_params.push_back(p);
            if (std::find(levels.begin(), levels.end(), int(l->i)) == levels.end()) levels.push_back(int(l->i));
        } else if (lt->s == "sfc") {
            sfc_params.push_back(p);
        } else {
            throw std::runtime_error(v->name + ": levtype " + lt->s);
        }
    }
    const auto& IL = kenda_interpolate_levels();
    std::vector<int>& targets = r.targets;  // the 600 hPa ones are dropped
    for (int l : IL)
        if (std::find(levels.begin(), levels.end(), l) != levels.end()) targets.push_back(l);
    if (targets.size() != levels.size())
        throw std::runtime_error("a model pressure level is not among model_to_pressure_level's levels");
    return r;
}

// The KENDA files of one valid time, in the order kenda_prognostics reads them (T_2M and PS twice).
std::vector<std::string> kenda_files(const KendaRequest& q, const std::string& dir, const DateTime& d) {
    const std::string raw = kenda_raw_dir(dir, d);
    std::vector<std::string> f = {raw + "/T_2M.grib2", raw + "/PS.grib2", raw + "/P.grib2"};
    for (const auto& sn : q.pl_params)
        if (sn != "FI") f.push_back(raw + "/" + sn + ".grib2");
    for (const auto& sn : q.sfc_params) f.push_back(raw + "/" + sn + ".grib2");
    return f;
}

// model_to_pressure_level + naming for one valid time (spec §1.3, §2.1-2.2, §1.6).
void kenda_prognostics(GribReader& rd, const IoConfig& io, const KendaRequest& q, const std::string& dir,
                       const DateTime& d, size_t slot, KendaConstants& kc, FieldState& out,
                       std::vector<std::string>& log) {
    const size_t N = out.n_points;
    const std::string raw = kenda_raw_dir(dir, d);
    const std::vector<std::string>& pl_params = q.pl_params;
    const std::vector<std::string>& sfc_params = q.sfc_params;
    const std::vector<int>& targets = q.targets;

    // Auxiliaries (VI:112-115), every one required for this valid time.
    auto read_sfc = [&](const std::string& p) {
        const std::string path = raw + "/" + p + ".grib2";
        if (!file_exists(path)) throw std::runtime_error("KENDA file missing: " + path);
        const auto msgs = rd(path);
        const GribHandle& h = one_message(*msgs, p, path);
        check_valid_at(h, d, path);
        std::vector<float> v = h.get_float_array("values");
        if (v.size() != N) throw std::runtime_error(path + ": size");
        log.push_back("read " + path + ": " + p + " level " + std::to_string(h.get_long("level")));
        return std::make_pair(v, h.get_long("level"));
    };
    const std::vector<float> t2m = read_sfc("T_2M").first, ps = read_sfc("PS").first;
    FloatBuf p = read_levels(rd, raw + "/P.grib2", "P", "generalVerticalLayer", 80, N, &d, log);
    const size_t K = 80;
    clamp_top_pressure(p.data(), N);
    check_pressure_increasing(p.data(), K, N);
    const LogPWeights w = log_p_weights(p.data(), K, N, targets);
    p.clear();
    p.shrink_to_fit();
    const auto& EL = kenda_extrapolate_levels();
    {  // the model-level fields, created up front in the order the loops below create them
        std::vector<std::string> names;
        auto add = [&](const std::string& sn) {
            for (int l : targets)
                if (const auto name = io.kenda_name(sn, l)) names.push_back(*name);
        };
        for (const std::string& sn : pl_params)
            if (sn != "FI") add(sn);
        if (std::find(pl_params.begin(), pl_params.end(), "FI") != pl_params.end()) add("FI");
        add_fields(out, names);
    }

    // The output of each target level, written in place: this slot of the named field.
    auto level_fields = [&](const std::string& short_name) {
        std::vector<float*> dst;
        for (size_t j = 0; j < targets.size(); ++j) {
            const auto name = io.kenda_name(short_name, targets[j]);
            if (!name) throw std::runtime_error("KENDA namer: no rule for " + short_name);
            dst.push_back(slot_field(out, *name).data() + slot * N);
        }
        return dst;
    };
    for (const std::string& sn : pl_params) {
        if (sn == "FI") continue;  // FI is skipped from the retrieval and computed from HHL (add_geopotential)
        const ExtrapKind kind = sn == "T" ? ExtrapKind::Temperature : ExtrapKind::LowestLevel;
        const FloatBuf x = read_levels(rd, raw + "/" + sn + ".grib2", sn, "generalVerticalLayer", 80, N, &d, log);
        interpolate_extrapolate(x.data(), K, w, t2m.data(), ps.data(), kc.hsurf.data(), kind, EL,
                                level_fields(sn).data());
    }
    if (std::find(pl_params.begin(), pl_params.end(), "FI") != pl_params.end()) {
        FloatBuf fi(K * N);
        geopotential_from_hhl(kc.hhl(rd, log).data(), K + 1, N, fi.data());
        interpolate_extrapolate(fi.data(), K, w, t2m.data(), ps.data(), kc.hsurf.data(), ExtrapKind::Geopotential, EL,
                                level_fields("FI").data());
    }
    // Surface fields pass through as decoded.
    for (const std::string& sn : sfc_params) {
        const auto [v, level] = read_sfc(sn);
        const auto name = io.kenda_name(sn, level);
        if (!name) throw std::runtime_error("KENDA namer: no rule for " + sn);
        put_slot(out, *name, slot, v);
    }
}

// Constants (spec §1.4, §2.3): HSURF*g -> FIS -> z, FR_LAND, SSO_SIGMA, SSO_STDH as decoded.
void kenda_constants(const IoConfig& io, const std::vector<const IoVariable*>& consts, const KendaConstants& kc,
                     FieldState& out, std::vector<std::string>& log) {
    const size_t N = out.n_points;
    for (const IoVariable* v : consts) {
        const std::string p = v->param();
        std::vector<float> vals;
        long level = 0;
        if (p == "FIS") {  // geopotential_from_height: double decode * 9.80665, one f32 cast
            vals.resize(N);
            for (size_t n = 0; n < N; ++n) vals[n] = float(kc.hsurf_d[n] * kG);
        } else {
            const GribHandle& h = one_message(*kc.horizontal, p, kc.horizontal_path);
            vals = h.get_float_array("values");
            level = h.get_long("level");
        }
        const auto name = io.kenda_name(p, level);
        if (!name || *name != v->name)
            throw std::runtime_error("KENDA namer gives '" + name.value_or("?") + "' for " + p + ", expected " + v->name);
        for (size_t s = 0; s < out.n_dates; ++s) put_slot(out, v->name, s, vals);
    }
    log.push_back("KENDA constants from " + kc.horizontal_path + " (HSURF * 9.80665 -> z)");
}

// ----------------------------------------------------------------------------------------------- IFS

struct IfsTask {
    const GribHandle* h = nullptr;
    std::string name, key;
    bool gh = false;  // gh -> z: double decode * 9.80665
    std::vector<size_t> slots;
};

// The ECMWF param of a variable (spec §1.5): name minus "_<level>", z on pl -> gh.
std::pair<std::string, long> ifs_param(const IoVariable& v, bool* pl) {
    const Scalar* lt = v.mars_get("levtype");
    const Scalar* l = v.mars_get("levelist");
    *pl = lt && lt->kind == Scalar::Str && lt->s == "pl";
    if (!*pl) return {v.name, 0};
    if (!l || l->kind != Scalar::Int) throw std::runtime_error(v.name + ": pl without levelist");
    const std::string suffix = "_" + std::to_string(l->i);
    if (v.name.size() <= suffix.size() || v.name.compare(v.name.size() - suffix.size(), suffix.size(), suffix) != 0)
        throw std::runtime_error(v.name + ": name does not end in " + suffix);
    std::string base = v.name.substr(0, v.name.size() - suffix.size());
    if (base == "z") base = "gh";
    return {base, long(l->i)};
}

std::string regrid_key(const GribHandle& h) {
    char b[160];
    std::snprintf(b, sizeof b, "%s@%ld@%08ld%04ld+%ld", h.get_string("shortName").c_str(), h.get_long("level"),
                  h.get_long("dataDate"), h.get_long("dataTime"), h.get_long("step"));
    return b;
}

// The selection keys of one open-data message (messages without them are never selected).
struct IfsKeys {
    const GribHandle* h = nullptr;
    std::string short_name, levtype;
    long level = 0, vdate = 0, vtime = 0, step = 0;
    size_t file = 0;
};

std::vector<IfsKeys> index_ifs(const std::vector<const std::vector<GribHandle>*>& files) {
    std::vector<IfsKeys> out;
    for (size_t f = 0; f < files.size(); ++f)
        for (const auto& h : *files[f]) {
            IfsKeys k;
            try {
                k.short_name = h.get_string("shortName");
                k.levtype = h.get_string("levtype");
                k.level = h.get_long("level");
                k.vdate = h.get_long("validityDate");
                k.vtime = h.get_long("validityTime");
                k.step = h.get_long("step");
            } catch (const std::exception&) {
                continue;
            }
            k.h = &h;
            k.file = f;
            out.push_back(std::move(k));
        }
    return out;
}

// Selects, for each variable, the one message of `files` valid at `valid` (constants: step 0).
void select_ifs(const std::vector<const IoVariable*>& vars, const std::vector<const std::vector<GribHandle>*>& files,
                const std::vector<std::string>& paths, const DateTime* valid, size_t slot_or_all, size_t n_dates,
                std::vector<IfsTask>& tasks, std::vector<std::string>& log) {
    const std::vector<IfsKeys> idx = index_ifs(files);
    for (const IoVariable* v : vars) {
        bool pl = false;
        const auto [param, level] = ifs_param(*v, &pl);
        const IfsKeys* found = nullptr;
        for (const auto& k : idx) {
            if (k.short_name != param) continue;
            if (pl ? (k.levtype != "pl" || k.level != level) : k.levtype != "sfc") continue;
            if (valid ? (k.vdate != yyyymmdd(*valid) || k.vtime != hhmm(*valid)) : k.step != 0) continue;
            if (found)
                throw std::runtime_error("IFS: more than one message for " + v->name + " (" + paths[found->file] +
                                         ", " + paths[k.file] + ")");
            found = &k;
        }
        if (!found)
            throw std::runtime_error("IFS: no message for " + v->name + " (" + param + (pl ? " pl " : " sfc ") +
                                     (valid ? "valid " + iso(*valid) : std::string("step 0")) + ")");
        IfsTask t;
        t.h = found->h;
        t.name = v->name;
        t.gh = param == "gh";
        t.key = regrid_key(*found->h);
        if (slot_or_all == size_t(-1))
            for (size_t s = 0; s < n_dates; ++s) t.slots.push_back(s);
        else
            t.slots.push_back(slot_or_all);
        log.push_back("IFS " + v->name + " <- " + t.key + " in " + paths[found->file]);
        tasks.push_back(std::move(t));
    }
}

void run_regrid(const Regridder& rg, const std::vector<IfsTask>& tasks, FieldState& out, InputStages* stages) {
    // Allocate every output field first, then regrid in parallel.
    std::vector<std::string> names;
    for (const auto& t : tasks) names.push_back(t.name);
    add_fields(out, names);
    std::vector<float*> dst(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) dst[i] = slot_field(out, tasks[i].name).data();
    std::vector<std::vector<double>> keep(tasks.size());
    parallel_for(int64_t(tasks.size()), [&](int64_t i) {
        const IfsTask& t = tasks[size_t(i)];
        std::vector<float> f;
        std::vector<double> d;
        rg.regrid(*t.h, t.gh ? nullptr : &f, &d);
        if (t.gh) {
            f.resize(d.size());
            for (size_t n = 0; n < d.size(); ++n) f[n] = float(d[n] * kG);  // orog_to_z: f64 * g, one f32 cast
        }
        for (size_t s : t.slots) std::copy(f.begin(), f.end(), dst[size_t(i)] + s * out.n_points);
        if (stages) keep[size_t(i)] = std::move(d);
    });
    if (stages)
        for (size_t i = 0; i < tasks.size(); ++i) stages->regrid[tasks[i].key] = std::move(keep[i]);
}

// ---------------------------------------------------------------------------------------------- merge

FieldState merge(const FieldState& lam, const FieldState& global, const std::vector<uint8_t>& mlam,
                 const std::vector<uint8_t>& mglob, size_t G) {
    if (mlam.size() != lam.n_points || mglob.size() != global.n_points)
        throw std::runtime_error("cutout: mask sizes differ from the source sizes");
    std::vector<size_t> idx_l, idx_g;
    for (size_t i = 0; i < mlam.size(); ++i)
        if (mlam[i]) idx_l.push_back(i);
    for (size_t i = 0; i < mglob.size(); ++i)
        if (mglob[i]) idx_g.push_back(i);
    if (idx_l.size() + idx_g.size() != G) throw std::runtime_error("cutout: masks do not keep G points");
    for (const auto& n : global.names)
        if (!lam.has(n)) throw std::runtime_error("cutout: field " + n + " in global but not in lam_0");
    FieldState out(lam.n_dates, G);
    std::vector<std::pair<const std::vector<float>*, const std::vector<float>*>> src;
    for (const auto& n : lam.names) {
        if (!global.has(n)) throw std::runtime_error("cutout: field " + n + " missing in the global source");
        src.push_back({&lam.at(n), &global.at(n)});
        out.names.push_back(n);
    }
    out.values.resize(src.size());
    // Copies only (every point is written): fields allocated, then one (field, date) per iteration.
    const int64_t n_fields = int64_t(src.size()), n_dates = int64_t(lam.n_dates);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t f = 0; f < n_fields; ++f) out.values[size_t(f)].resize(lam.n_dates * G);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t fs = 0; fs < n_fields * n_dates; ++fs) {
        const size_t f = size_t(fs / n_dates), s = size_t(fs % n_dates);
        float* dst = out.values[f].data() + s * G;
        const float* sa = src[f].first->data() + s * lam.n_points;
        const float* sb = src[f].second->data() + s * global.n_points;
        for (size_t i = 0; i < idx_l.size(); ++i) dst[i] = sa[idx_l[i]];
        for (size_t i = 0; i < idx_g.size(); ++i) dst[idx_l.size() + i] = sb[idx_g[i]];
    }
    return out;
}

}  // namespace

// ------------------------------------------------------------------------------------------- build

InputState build_input(const Model& model, const InputFiles& files) {
    return build_input(InputModel::from(model), files);
}

InputState build_input(const InputModel& model, const InputFiles& files) {
    const double t_start = now_s();
    const Config& cfg = model.config;
    InputState st;
    auto& log = st.log;
    if (!files.cosmo_definitions.empty()) set_definitions_path(files.cosmo_definitions);
    log.push_back("ecCodes " + std::to_string(eccodes_version()) + ", definitions " + definitions_path());

    const IoConfig io = IoConfig::load(files.io_config);
    const Retrieved rv = retrieved_variables(io, cfg);
    if (cfg.n_input_steps != 2) throw std::runtime_error("real-data input supports 2 input steps (t-6h, t)");
    if (cfg.timestep_hours <= 0) throw std::runtime_error("model config has no timestep");
    if (model.cutout.size() != 2 || model.cutout[0].name != "lam_0" || model.cutout[1].name != "global")
        throw std::runtime_error("model is not a lam_0 + global cutout");
    const size_t T = size_t(cfg.n_input_steps), G = size_t(cfg.n_data_nodes), V = size_t(cfg.n_input_vars);
    const size_t N_lam = model.cutout[0].mask.size(), N_glob = model.cutout[1].mask.size();
    if (model.latitudes.size() != G) throw std::runtime_error("model has no grid coordinates");

    st.T = int64_t(T);
    st.G = int64_t(G);
    st.V_in = int64_t(V);
    st.date0 = parse_input_date(files.date);
    for (size_t s = 0; s < T; ++s)
        st.dates.push_back(add_hours(st.date0, -int64_t(cfg.timestep_hours) * int64_t(T - 1 - s)));
    const std::vector<std::vector<std::string>> ifs_files = {files.ifs_files_prev, files.ifs_files_t};
    {
        std::string ds;
        for (const auto& d : st.dates) ds += " " + iso(d);
        log.push_back("dates" + ds + "; " + std::to_string(rv.prognostics.size()) + " prognostics, " +
                      std::to_string(rv.constants.size()) + " constants, " + std::to_string(rv.computed.size()) +
                      " computed forcings");
    }

    // ---- KENDA (lam_0) and IFS (global) run concurrently. Every GRIB file is read ahead of its use
    // by a pool of parallel readers, in the order the branches need them: KENDA's first files
    // (constants, T_2M, PS, P), the IFS files, then the rest of KENDA in the order its branch reads them.
    const KendaRequest kq = kenda_request(rv.prognostics);
    std::vector<std::string> ahead;
    ahead.push_back(kenda_horizontal_path(files.kenda_dir));
    for (size_t s = 0; s < T; ++s) {
        std::vector<std::string> f = kenda_files(kq, files.kenda_dir, st.dates[s]);
        if (s == 0) {  // HHL (FI) after the first date's model-level files
            size_t n_ml = 3;
            for (const auto& sn : kq.pl_params) n_ml += sn != "FI";
            f.insert(f.begin() + int64_t(n_ml), kenda_vertical_path(files.kenda_dir));
        }
        ahead.insert(ahead.end(), f.begin(), f.end());
        if (s == 0) {  // the IFS files after T_2M, PS and P of the first date
            std::vector<std::string> ifs;
            for (size_t t = 0; t < T; ++t) ifs.insert(ifs.end(), ifs_files[t].begin(), ifs_files[t].end());
            ifs.insert(ifs.end(), files.ifs_constant_files.begin(), files.ifs_constant_files.end());
            ahead.insert(ahead.begin() + 4, ifs.begin(), ifs.end());
        }
    }
    GribPrefetch::Options po;
    po.threads = input_io_threads();
    GribPrefetch pool(ahead, po);
    GribReader rd_kenda{&pool}, rd_ifs{&pool};

    double t0 = now_s();
    const int ifs_threads = input_ifs_threads();

    // ---- the x0 buffer and the computed forcings (dates and grid only), on their own thread
    std::vector<std::string> fnames;  // the computed forcings among the inputs, in model-input order
    for (const auto& n : cfg.var_input)
        if (std::find(rv.computed.begin(), rv.computed.end(), n) != rv.computed.end()) fnames.push_back(n);
    std::vector<std::vector<float>> forc(T);
    std::exception_ptr x0_err;
    std::thread x0_thread([&] {
        try {
            omp_set_num_threads(std::max(1, std::min(8, ifs_threads)));
            st.x0.assign(T * G * V, std::numeric_limits<float>::quiet_NaN());
            for (size_t s = 0; s < T; ++s) {
                forc[s].resize(G * fnames.size());
                if (!fnames.empty())
                    compute_forcings(st.dates[s], fnames, model.latitudes.data(), model.longitudes.data(), G,
                                     forc[s].data());
            }
        } catch (...) {
            x0_err = std::current_exception();
        }
    });
    struct JoinX0 {
        std::thread& t;
        ~JoinX0() {
            if (t.joinable()) t.join();
        }
    } join_x0{x0_thread};
    std::vector<std::string> trace_ifs, trace_kenda;  // VARDA_INPUT_TRACE: branch timelines
    auto mark = [&](std::vector<std::string>& tr, const std::string& what) {
        char b[48];
        std::snprintf(b, sizeof b, "%7.3f s ", now_s() - t0);
        tr.push_back(b + what);
    };

    // ---- global: IFS open data, regridded to N320 (own thread, own log; appended after KENDA's)
    FieldState prog_glob(T, N_glob), const_glob(T, N_glob);
    std::vector<std::string> log_ifs;
    double ifs_s = 0;
    std::exception_ptr ifs_err;
    std::thread ifs_thread([&] {
        try {
            omp_set_num_threads(ifs_threads);
            auto& log = log_ifs;
            // ecCodes loads its ECMWF concept tables on the first key lookups, under its global
            // lock: do it now, from the N320 template (also ECMWF GRIB2), while the IFS files are
            // still being read, rather than when the parser needs the lock. Best effort.
            try {
                const GribHandle h = read_first_message(files.n320_template);
                h.get_string("shortName");
                h.get_string("levtype");
                h.get_long("level");
                h.get_long("validityDate");
                h.get_long("validityTime");
                h.get_long("step");
            } catch (const std::exception&) {
            }
            mark(trace_ifs, "ecCodes concepts loaded");
            const RegridMatrix m =
                files.regrid_matrix.size() > 4 &&
                        files.regrid_matrix.compare(files.regrid_matrix.size() - 4, 4, ".mat") == 0
                    ? RegridMatrix::load_mir_mat(files.regrid_matrix, RegridGeometry::ifs_open_data_0p25())
                    : RegridMatrix::load(files.regrid_matrix);
            mark(trace_ifs, "matrix loaded");
            if (m.n_target() != int64_t(N_glob))
                throw std::runtime_error("regrid matrix has " + std::to_string(m.n_target()) + " rows, global source " +
                                         std::to_string(N_glob) + " points");
            log.push_back("regrid matrix " + files.regrid_matrix + ": " + std::to_string(m.n_target()) + " x " +
                          std::to_string(m.n_source()) + ", nnz " + std::to_string(m.nnz()));
            Regridder rg(m, read_first_message(files.n320_template));
            log.push_back("N320 encoding template " + files.n320_template);
            rg.target_coordinates(&prog_glob.latitudes, &prog_glob.longitudes);
            const_glob.latitudes = prog_glob.latitudes;
            const_glob.longitudes = prog_glob.longitudes;
            mark(trace_ifs, "template and coordinates");

            std::vector<std::shared_ptr<const std::vector<GribHandle>>> loaded;
            std::vector<std::string> paths;
            std::vector<std::vector<size_t>> per_slot(T);
            for (size_t s = 0; s < T; ++s)
                for (const auto& p : ifs_files[s]) {
                    per_slot[s].push_back(loaded.size());
                    loaded.push_back(rd_ifs(p));
                    paths.push_back(p);
                    log.push_back("read " + p + ": " + std::to_string(loaded.back()->size()) + " messages");
                }
            std::vector<IfsTask> tasks_prog, tasks_const;
            for (size_t s = 0; s < T; ++s) {
                std::vector<const std::vector<GribHandle>*> fs;
                std::vector<std::string> ps;
                for (size_t i : per_slot[s]) {
                    fs.push_back(loaded[i].get());
                    ps.push_back(paths[i]);
                }
                select_ifs(rv.prognostics, fs, ps, &st.dates[s], s, T, tasks_prog, log);
            }
            std::vector<std::shared_ptr<const std::vector<GribHandle>>> cfiles;
            for (const auto& p : files.ifs_constant_files) {
                cfiles.push_back(rd_ifs(p));
                log.push_back("read " + p + ": " + std::to_string(cfiles.back()->size()) + " messages (constants run)");
            }
            std::vector<const std::vector<GribHandle>*> cfs;
            for (const auto& f : cfiles) cfs.push_back(f.get());
            select_ifs(rv.constants, cfs, files.ifs_constant_files, nullptr, size_t(-1), T, tasks_const, log);
            InputStages* sp = files.keep_stages ? &st.stages : nullptr;
            mark(trace_ifs, "files taken, messages selected");
            run_regrid(rg, tasks_prog, prog_glob, sp);
            run_regrid(rg, tasks_const, const_glob, sp);
            mark(trace_ifs, "regridded");
            ifs_s = now_s() - t0;
        } catch (...) {
            ifs_err = std::current_exception();
        }
    });
    struct Joiner {
        std::thread& t;
        ~Joiner() {
            if (t.joinable()) t.join();
        }
    } joiner{ifs_thread};

    // ---- lam_0: KENDA (this thread)
    FieldState prog_lam(T, N_lam), const_lam(T, N_lam);
    std::exception_ptr kenda_err;
    try {
        KendaConstants kc = read_kenda_constants(rd_kenda, files.kenda_dir, N_lam, log);
        mark(trace_kenda, "constants");
        for (size_t s = 0; s < T; ++s) {
            kenda_prognostics(rd_kenda, io, kq, files.kenda_dir, st.dates[s], s, kc, prog_lam, log);
            mark(trace_kenda, "date " + iso(st.dates[s]));
        }
        kc.hhl(rd_kenda, log);  // read (and checked) even when no FI is requested
        kenda_constants(io, rv.constants, kc, const_lam, log);
    } catch (...) {
        kenda_err = std::current_exception();
    }
    st.seconds["kenda"] = now_s() - t0;
    st.seconds["kenda_read_wait"] = rd_kenda.wait_s;
    ifs_thread.join();
    if (kenda_err) std::rethrow_exception(kenda_err);  // KENDA's error first, as in the sequential order
    if (ifs_err) std::rethrow_exception(ifs_err);
    st.seconds["ifs"] = ifs_s;
    st.seconds["ifs_read_wait"] = rd_ifs.wait_s;
    st.seconds["kenda_ifs"] = now_s() - t0;
    log.insert(log.end(), log_ifs.begin(), log_ifs.end());
    if (std::getenv("VARDA_INPUT_TRACE")) {
        for (const auto& l : pool.trace()) log.push_back("trace prefetch " + l);
        for (const auto& l : trace_kenda) log.push_back("trace kenda " + l);
        for (const auto& l : trace_ifs) log.push_back("trace ifs " + l);
    }

    // Every requested variable present in both sources, nothing else.
    for (const auto* v : rv.prognostics)
        if (!prog_lam.has(v->name) || !prog_glob.has(v->name))
            throw std::runtime_error("prognostic " + v->name + " not produced by both sources");
    if (prog_lam.names.size() != rv.prognostics.size() || prog_glob.names.size() != rv.prognostics.size())
        throw std::runtime_error("sources produced extra prognostic fields");

    // ---- cutout merge (spec §2.6)
    t0 = now_s();
    const auto& mlam = model.cutout[0].mask;
    const auto& mglob = model.cutout[1].mask;
    // Order the source states like the reference (variables_metadata order).
    auto reorder = [](FieldState& s, const std::vector<const IoVariable*>& vars) {
        FieldState o(s.n_dates, s.n_points);
        o.latitudes = std::move(s.latitudes);
        o.longitudes = std::move(s.longitudes);
        for (const auto* v : vars) {
            o.names.push_back(v->name);
            o.values.push_back(std::move(s.at(v->name)));
        }
        s = std::move(o);
    };
    reorder(prog_lam, rv.prognostics);
    reorder(prog_glob, rv.prognostics);
    reorder(const_lam, rv.constants);
    reorder(const_glob, rv.constants);
    FieldState prog = merge(prog_lam, prog_glob, mlam, mglob, G);
    FieldState cons = merge(const_lam, const_glob, mlam, mglob, G);
    // Coordinates: the global part of the data grid is the N320 geometry under the mask (§1.7).
    {
        size_t g = 0, bad = 0;
        for (size_t i = 0; i < mlam.size(); ++i) g += mlam[i];
        for (size_t i = 0; i < mglob.size(); ++i)
            if (mglob[i]) {
                bad += prog_glob.latitudes[i] != model.latitudes[g] || prog_glob.longitudes[i] != model.longitudes[g];
                ++g;
            }
        if (bad) throw std::runtime_error("N320 template coordinates differ from grid.* at " + std::to_string(bad) +
                                          " global points");
        log.push_back("global coordinates: N320 template geometry under the mask == grid.latitudes/longitudes (bitwise)");
    }
    st.seconds["merge"] = now_s() - t0;

    // ---- x0 (spec §3)
    t0 = now_s();
    std::vector<int> fcol(V, -1);
    for (size_t i = 0, k = 0; i < V; ++i) {
        const std::string& n = cfg.var_input[i];
        const bool in_p = prog.has(n), in_c = cons.has(n), comp = std::find(rv.computed.begin(), rv.computed.end(), n) !=
                                                                    rv.computed.end();
        if (int(in_p) + int(in_c) + int(comp) != 1) throw std::runtime_error("input " + n + " is not set exactly once");
        if (comp) fcol[i] = int(k++);
    }
    x0_thread.join();
    if (x0_err) std::rethrow_exception(x0_err);
    for (size_t s = 0; s < T; ++s) {
        // Row by row (contiguous writes), each column gathered from its field or forcing.
        std::vector<const float*> col(V);
        std::vector<size_t> stride(V, 1);
        for (size_t i = 0; i < V; ++i) {
            const std::string& n = cfg.var_input[i];
            if (fcol[i] >= 0) {
                col[i] = forc[s].data() + size_t(fcol[i]);
                stride[i] = fnames.size();
            } else {
                col[i] = (prog.has(n) ? prog.at(n) : cons.at(n)).data() + s * G;
            }
        }
        float* xs = st.x0.data() + s * G * V;
#pragma omp parallel for schedule(static)
        for (int64_t g = 0; g < int64_t(G); ++g) {
            float* row = xs + size_t(g) * V;
            for (size_t i = 0; i < V; ++i) row[i] = col[i][size_t(g) * stride[i]];
        }
    }
    // NaN rule (spec §3); integer counts, so the sum does not depend on the threads
    {
        std::vector<int64_t> nan(V, 0);
        const float* x = st.x0.data();
#pragma omp parallel
        {
            std::vector<int64_t> mine(V, 0);
#pragma omp for schedule(static)
            for (int64_t r = 0; r < int64_t(T * G); ++r)
                for (size_t i = 0; i < V; ++i) mine[i] += std::isnan(x[size_t(r) * V + i]);
#pragma omp critical(varda_io_input_nan)
            for (size_t i = 0; i < V; ++i) nan[i] += mine[i];
        }
        std::string msg;
        for (size_t i = 0; i < V; ++i)
            if (nan[i]) msg += " " + cfg.var_input[i] + ":" + std::to_string(nan[i]);
        if (!msg.empty()) {
            log.push_back("NaN in x0:" + msg);
            if (!files.allow_nans) throw std::runtime_error("NaN in the model input (variable:count):" + msg);
        } else {
            log.push_back("x0 is NaN-free");
        }
    }
    st.seconds["x0"] = now_s() - t0;

    // ---- step-0 output fields (spec §4.1)
    {
        std::vector<const std::vector<float>*> src;
        for (const auto& n : io.initial_state_fields) {
            src.push_back(&(prog.has(n) ? prog : cons).at(n));
            st.initial_names.push_back(n);
        }
        st.initial_values.resize(src.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < int64_t(src.size()); ++i)
            st.initial_values[size_t(i)].assign(src[size_t(i)]->begin() + int64_t((T - 1) * G),
                                                src[size_t(i)]->begin() + int64_t(T * G));
    }

    if (files.keep_stages) {
        st.stages.states["source.prognostics.lam_0"] = std::move(prog_lam);
        st.stages.states["source.prognostics.global"] = std::move(prog_glob);
        st.stages.states["source.constant_forcings.lam_0"] = std::move(const_lam);
        st.stages.states["source.constant_forcings.global"] = std::move(const_glob);
        st.stages.states["cutout.prognostics"] = std::move(prog);
        st.stages.states["cutout.constant_forcings"] = std::move(cons);
    }
    st.seconds["total"] = now_s() - t_start;
    if (model.load_seconds > 0) st.seconds["model_grid"] = model.load_seconds;
    return st;
}

void files_from_realdata(const std::string& dir, InputFiles& f) {
    std::ifstream in(dir + "/pins.json");
    if (!in) throw std::runtime_error("cannot read " + dir + "/pins.json");
    const nlohmann::json pins = nlohmann::json::parse(in);
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
        const nlohmann::json& e = pins.at("input_dates").at(varda::format_iso8601_utc(d));
        (s == 0 ? f.ifs_files_prev : f.ifs_files_t).push_back(file_of(e.at("init"), e.at("step")));
    }
    const nlohmann::json& c = pins.at("constant_forcings_run");
    f.ifs_constant_files.push_back(file_of(c.at("init"), c.at("step")));
}


}  // namespace varda::io
