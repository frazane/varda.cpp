#include "varda/io/output.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>

namespace varda::io {

namespace fs = std::filesystem;

// ---- Python-compatible helpers -------------------------------------------------------------------

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

Scalar* find_mut(ScalarMap& m, const std::string& key) {
    for (auto& kv : m)
        if (kv.first == key) return &kv.second;
    return nullptr;
}
// dict[key] = value: replace in place, or append.
void put(ScalarMap& m, const std::string& key, Scalar v) {
    if (Scalar* s = find_mut(m, key)) *s = std::move(v);
    else m.emplace_back(key, std::move(v));
}
// dict.setdefault(key, value)
void put_default(ScalarMap& m, const std::string& key, Scalar v) {
    if (!find_mut(m, key)) m.emplace_back(key, std::move(v));
}
void erase(ScalarMap& m, const std::string& key) {
    m.erase(std::remove_if(m.begin(), m.end(), [&](const auto& kv) { return kv.first == key; }), m.end());
}

}  // namespace

bool py_parses_as_int(const std::string& str) {
    std::string s = trim(str);
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
    if (i >= s.size()) return false;
    bool prev_digit = false;
    for (; i < s.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(s[i]))) prev_digit = true;
        else if (s[i] == '_' && prev_digit && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1]))) prev_digit = false;
        else return false;
    }
    return true;
}

bool py_parses_as_float(const std::string& str) {
    std::string s = trim(str);
    if (s.empty()) return false;
    for (char c : s)
        if (c == 'x' || c == 'X' || c == '(' || c == ')') return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}

std::string param_key_name(const std::string& param) {
    if (py_parses_as_int(param)) return "paramId";
    if (py_parses_as_float(param)) return "param";
    return "shortName";
}

// ---- keys -----------------------------------------------------------------------------------------

ScalarMap build_keys(const IoVariable& v, const ScalarMap& encoding, int64_t edition, const DateTime& base,
                     int step_hours, std::optional<int> accum_start_hours) {
    ScalarMap r = encoding;
    put(r, "edition", Scalar::integer(edition));
    if (edition != 2) throw std::runtime_error("build_keys: only GRIB edition 2 is implemented");

    const Scalar* pm = v.mars_get("param");
    const std::string param = (pm && pm->kind != Scalar::Null) ? pm->str() : v.name;
    put_default(r, param_key_name(param), Scalar::string(param));

    put_default(r, "type", Scalar::string("fc"));
    {
        const Scalar* t = find(r, "type");
        if (t->kind == Scalar::Str && (t->s == "an" || t->s == "fc")) {
            Scalar tv = *t;
            erase(r, "type");
            put_default(r, "dataType", tv);
        }
    }
    put(r, "date", Scalar::integer(int64_t(base.year) * 10000 + base.month * 100 + base.day));
    put(r, "time", Scalar::integer(int64_t(base.hour) * 100));

    // encode_time_processing (non-ensemble)
    if (v.process.empty()) {
        put(r, "step", Scalar::integer(step_hours));
        put(r, "stepType", Scalar::string("instant"));
    } else if (v.process == "accumulation") {
        int start = step_hours - v.period_hours;
        if (start < 0)
            throw std::runtime_error("build_keys: negative start step for " + v.name + " at step " +
                                     std::to_string(step_hours) +
                                     " (anemoi shifts the base time and then refuses it; not implemented)");
        if (accum_start_hours) start = *accum_start_hours;
        put(r, "startStep", Scalar::integer(start));
        put(r, "endStep", Scalar::integer(step_hours));
        put(r, "stepType", Scalar::string("accum"));
        put(r, "productDefinitionTemplateNumber", Scalar::integer(8));
    } else {
        throw std::runtime_error("build_keys: process " + v.process + " not implemented");
    }

    static const std::set<std::string> skip = {"domain", "type", "expver", "class", "param", "number", "step",
                                               "date", "hdate", "time", "timespan", "valid_datetime", "variable"};
    for (const auto& [k0, val] : v.mars) {
        if (skip.count(k0)) continue;
        std::string k = k0;
        Scalar x = val;
        if (k == "stream") throw std::runtime_error("build_keys: mars key 'stream' not implemented");
        if (k == "levtype") {
            static const std::map<std::string, std::string> lev = {
                {"pl", "isobaricInhPa"}, {"ml", "hybrid"}, {"pt", "theta"}, {"pv", "potentialVorticity"}};
            auto it = lev.find(x.str());
            if (x.kind != Scalar::Str || it == lev.end()) continue;
            x = Scalar::string(it->second);
            k = "typeOfLevel";
        }
        if (k == "levelist") k = "level";
        put_default(r, k, x);
    }

    // stable sort by ORDERING, then drop None
    auto rank = [](const std::string& k) {
        static const std::vector<std::string> o = {"edition", "typeOfLevel", "stepType",
                                                   "productDefinitionTemplateNumber", "eps", "number"};
        for (size_t i = 0; i < o.size(); ++i)
            if (o[i] == k) return int(i);
        return 999;
    };
    std::stable_sort(r.begin(), r.end(), [&](const auto& a, const auto& b) { return rank(a.first) < rank(b.first); });
    r.erase(std::remove_if(r.begin(), r.end(), [](const auto& kv) { return kv.second.kind == Scalar::Null; }), r.end());
    return r;
}

ScalarMap apply_patches(const ModifierDef& m, const std::string& name, ScalarMap keys) {
    const auto& rules = m.patches.at(size_t(m.effective));
    for (const auto& rule : rules) {
        std::regex re(rule.regex, std::regex::ECMAScript);
        std::smatch sm;
        if (!std::regex_search(name, sm, re, std::regex_constants::match_continuous)) continue;
        for (const auto& [k, v] : rule.keys) {
            if (v.kind == Scalar::Str && !v.s.empty() && v.s[0] == '\\')
                throw std::runtime_error("apply_patches: back-references not implemented");
            put(keys, k, v);
        }
        break;  // Matching.patch returns the first match
    }
    return keys;
}

const TemplateIndexEntry* match_template(const OutputDef& out, const IoVariable& v) {
    auto lookup = [&](const std::string& k) -> Scalar {
        if (const Scalar* e = find(out.encoding, k)) return *e;
        if (k == "name") return Scalar::string(v.name);
        if (k == "time_processing") return v.process.empty() ? Scalar::null() : Scalar::string(v.process);
        if (k == "step" || k == "date" || k == "time" || k == "hdate") return Scalar::null();
        if (const Scalar* s = v.mars_get(k)) return *s;
        return Scalar::null();
    };
    for (const auto& e : out.templates) {
        bool ok = true;
        for (const auto& [k, vals] : e.match) {
            const Scalar have = lookup(k);
            if (std::find(vals.begin(), vals.end(), have) == vals.end()) {
                ok = false;
                break;
            }
        }
        if (ok) return &e;
    }
    return nullptr;
}

std::string render_path(const std::string& pattern, const std::string& date_time, int64_t step) {
    std::string out;
    size_t i = 0;
    while (i < pattern.size()) {
        if (pattern[i] != '{') {
            out += pattern[i++];
            continue;
        }
        size_t e = pattern.find('}', i);
        if (e == std::string::npos) throw std::runtime_error("render_path: unbalanced '{' in " + pattern);
        const std::string expr = pattern.substr(i + 1, e - i - 1);
        if (expr.find(':') != std::string::npos)
            throw std::runtime_error("render_path: format specs are not implemented: {" + expr + "}");
        if (expr == "dateTime") out += date_time;
        else if (expr == "step") out += std::to_string(step);
        else throw std::runtime_error("render_path: unknown key {" + expr + "}");
        i = e + 1;
    }
    return out;
}

// ---- post-processors ----------------------------------------------------------------------------

void accumulate_into(std::vector<float>& acc, const std::vector<float>& x, bool allow_negative) {
    if (acc.empty()) acc.assign(x.size(), 0.0f);
    if (acc.size() != x.size()) throw std::runtime_error("accumulate: size changed");
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        if (!allow_negative) {
            // np.maximum(0, v): (a >= b || isnan(a)) ? a : b, a = 0
            v = (0.0f >= v) ? 0.0f : v;
        }
        acc[i] = acc[i] + v;
    }
}

std::vector<float> rescale(const std::vector<float>& x, double scale, double offset) {
    std::vector<float> out(x.size());
    const float s = float(scale), o = float(offset);
    for (size_t i = 0; i < x.size(); ++i) {
        const float p = x[i] * s;
        out[i] = p + o;
    }
    return out;
}

std::vector<float> assign_mask(const std::vector<float>& x, const std::vector<uint8_t>& mask) {
    std::vector<float> out(mask.size(), std::nanf(""));
    size_t k = 0;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i]) {
            if (k >= x.size()) throw std::runtime_error("assign_mask: fewer values than true mask entries");
            out[i] = x[k++];
        }
    if (k != x.size()) throw std::runtime_error("assign_mask: more values than true mask entries");
    return out;
}

bool nan_to_missing(std::vector<float>& x, float missing_value) {
    bool any = false;
    for (float v : x)
        if (std::isnan(v)) {
            any = true;
            break;
        }
    if (!any) return false;
    for (float& v : x) {
        if (std::isnan(v)) v = missing_value;
        else if (std::isinf(v)) v = v > 0 ? FLT_MAX : -FLT_MAX;
    }
    return true;
}

// ---- the writer ---------------------------------------------------------------------------------

namespace {

int64_t count_true(const std::vector<uint8_t>& m) {
    int64_t n = 0;
    for (uint8_t b : m) n += b != 0;
    return n;
}

// check_encoding (encoding.py:349-412): read every key back.
std::vector<std::string> mismatches(const GribHandle& h, const ScalarMap& keys) {
    std::vector<std::string> out;
    for (const auto& [k0, v0] : keys) {
        std::string k = k0;
        Scalar v = v0;
        if (k == "param") k = param_key_name(v.str());
        if (k == "date") {
            std::string t = v.str();
            t.erase(std::remove(t.begin(), t.end(), '-'), t.end());
            v = Scalar::integer(std::stoll(t));
        }
        if (k == "time") {
            int64_t t = v.kind == Scalar::Int ? v.i : std::stoll(v.str());
            if (t < 100) t *= 100;
            v = Scalar::integer(t);
        }
        try {
            if (v.kind == Scalar::Int) {
                const long w = h.get_long(k);
                if (w != v.i) out.push_back(k + ": expected " + v.str() + " got " + std::to_string(w));
            } else {
                const std::string w = h.get_string(k);
                if (w != v.str()) out.push_back(k + ": expected " + v.str() + " got " + w);
            }
        } catch (const std::exception& e) {
            out.push_back(k + ": " + e.what());
        }
    }
    return out;
}

struct OutStream {
    const OutputDef* def = nullptr;
    std::map<std::string, GribHandle> templates;                    // by file
    std::map<std::string, const TemplateIndexEntry*> entry_of;      // by variable name
    std::map<std::string, std::unique_ptr<GribFileWriter>> writers; // by final path
    std::vector<std::string> created;
};

}  // namespace

struct GribOutput::Impl {
    IoConfig cfg;
    std::vector<GridSource> sources;
    std::vector<int64_t> offset, count;
    std::vector<std::string> out_names;
    std::string templates_dir, out_dir;
    DateTime base;
    int64_t G = 0;
    std::vector<OutStream> streams;
    std::map<std::string, std::vector<float>> acc;  // accumulators by variable name
    std::vector<float> transposed;                  // write_step's (V, G) copy of y

    const GridSource& source(const std::string& name, size_t* idx = nullptr) const {
        for (size_t i = 0; i < sources.size(); ++i)
            if (sources[i].name == name) {
                if (idx) *idx = i;
                return sources[i];
            }
        throw std::runtime_error("GribOutput: no cutout source '" + name + "'");
    }
    const std::vector<uint8_t>& mask_named(const std::string& m) const {
        const std::string suffix = "/cutout_mask";
        if (m.size() <= suffix.size() || m.compare(m.size() - suffix.size(), suffix.size(), suffix) != 0)
            throw std::runtime_error("GribOutput: mask '" + m + "' is not <source>/cutout_mask");
        return source(m.substr(0, m.size() - suffix.size())).mask;
    }

    // Top-level post-processors over a whole state; returns the fields they changed and the start steps.
    struct Top {
        std::map<std::string, std::vector<float>> changed;
        std::map<std::string, int> start_steps;
    };
    template <class Col>
    Top run_top(const std::vector<std::string>& names, Col&& column) {
        Top t;
        auto current = [&](size_t i) -> std::vector<float> {
            auto it = t.changed.find(names[i]);
            return it != t.changed.end() ? it->second : column(i);
        };
        for (const auto& pp : cfg.post_processors) {
            if (pp.type == PostProcessorDef::Accumulate) {
                for (const auto& a : pp.accumulations) {
                    auto it = std::find(names.begin(), names.end(), a);
                    if (it == names.end()) continue;
                    const size_t i = size_t(it - names.begin());
                    accumulate_into(acc[a], current(i), pp.allow_negative);
                    t.changed[a] = acc[a];
                    t.start_steps[a] = 0;
                }
            } else if (pp.type == PostProcessorDef::Rescale) {
                for (size_t i = 0; i < names.size(); ++i) {
                    const IoVariable* v = cfg.find_variable(names[i]);
                    if (v && v->param() == pp.param) t.changed[names[i]] = rescale(current(i), pp.scale, pp.offset);
                }
            } else {
                throw std::runtime_error("GribOutput: top-level post-processor type not supported");
            }
        }
        return t;
    }

    std::vector<float> run_output_pps(const OutputDef& d, std::vector<float> a) const {
        for (const auto& pp : d.post_processors) {
            switch (pp.type) {
                case PostProcessorDef::ExtractFromState: {
                    size_t i = 0;
                    source(pp.source, &i);
                    a = std::vector<float>(a.begin() + offset[i], a.begin() + offset[i] + count[i]);
                    break;
                }
                case PostProcessorDef::ExtractMask: {
                    if (!pp.as_slice) throw std::runtime_error("GribOutput: extract_mask without as_slice not implemented");
                    const int64_t n = count_true(mask_named(pp.mask));
                    a = pp.inverse ? std::vector<float>(a.begin() + n, a.end()) : std::vector<float>(a.begin(), a.begin() + n);
                    break;
                }
                case PostProcessorDef::AssignMask:
                    a = assign_mask(a, mask_named(pp.mask));
                    break;
                default:
                    throw std::runtime_error("GribOutput: output post-processor type not supported");
            }
        }
        return a;
    }

    const GribHandle& template_for(OutStream& s, const IoVariable& v) {
        auto it = s.entry_of.find(v.name);
        const TemplateIndexEntry* e;
        if (it != s.entry_of.end()) e = it->second;
        else {
            e = match_template(*s.def, v);
            if (!e) throw std::runtime_error("GribOutput: no template for '" + v.name + "' in output " + s.def->name);
            s.entry_of[v.name] = e;
        }
        auto t = s.templates.find(e->file);
        if (t == s.templates.end())
            t = s.templates.emplace(e->file, read_first_message(templates_dir + "/" + e->file)).first;
        return t->second;
    }

    // The template of a variable whose template write_state has already looked up (thread-safe).
    const GribHandle& template_of(const OutStream& s, const IoVariable& v) const {
        return s.templates.at(s.entry_of.at(v.name)->file);
    }

    struct Encoded {
        GribHandle h;
        std::string path, error;
    };

    // One message, encoded as anemoi-inference's encode_message does; no shared state is modified, so
    // messages of a state are encoded in parallel and written afterwards in order.
    Encoded encode_field(const OutStream& s, const IoVariable& v, std::vector<float> values, int step_hours,
                         const Top& top, double* cpu) const {
        using clock = std::chrono::steady_clock;
        auto secs = [](clock::time_point a) { return std::chrono::duration<double>(clock::now() - a).count(); };
        auto t0 = clock::now();
        const OutputDef& d = *s.def;
        const GribHandle& tmpl = template_of(s, v);
        auto sit = top.start_steps.find(v.name);
        ScalarMap keys = build_keys(v, d.encoding, tmpl.get_long("edition"), base, step_hours,
                                    sit == top.start_steps.end() ? std::nullopt : std::optional<int>(sit->second));
        for (const auto& m : d.modifiers) keys = apply_patches(m, v.name, keys);

        // encode_message
        ScalarMap meta = keys;
        if (nan_to_missing(values, float(d.missing_value))) {
            put(meta, "missingValue", Scalar::integer(int64_t(d.missing_value)));
            put(meta, "bitmapPresent", Scalar::integer(1));
        }
        put_default(meta, "generatingProcessIdentifier", Scalar::integer(255));
        Encoded e;
        {
            std::lock_guard<std::mutex> lock(clone_mutex);  // reads the shared template handle
            e.h = tmpl.clone();
        }
        GribHandle& h = e.h;
        KeyVals kv;
        for (const auto& [k, x] : meta) kv.emplace_back(k, x.str());
        try {
            h.set_key_vals(kv);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "Failed to set metadata at once: %s\n", ex.what());
            for (const auto& [k, x] : meta) {
                if (x.kind == Scalar::Int) h.set_long(k, long(x.i));
                else h.set_string(k, x.s);
            }
        }
        std::vector<double> dv(values.begin(), values.end());
        cpu[1] += secs(t0);
        t0 = clock::now();
        h.set_double_array("values", dv);
        cpu[2] += secs(t0);
        t0 = clock::now();

        // target file
        const std::string dt = h.get_string("dateTime");
        e.path = out_dir + "/" + render_path(d.path_effective, dt, h.get_long("step"));

        if (d.check_encoding) {
            auto mm = mismatches(h, keys);
            if (!mm.empty()) {
                GribHandle c = h.clone();
                mm = mismatches(c, keys);
            }
            if (!mm.empty()) {
                std::string msg = "GRIB field could not be encoded (" + v.name + " in " + d.name + "):";
                for (const auto& m : mm) msg += " " + m + ";";
                throw std::runtime_error(msg);
            }
        }
        cpu[3] += secs(t0);
        return e;
    }

    void write_encoded(OutStream& s, const Encoded& e) {
        auto w = s.writers.find(e.path);
        if (w == s.writers.end()) {
            fs::create_directories(fs::path(e.path).parent_path());
            w = s.writers.emplace(e.path, std::make_unique<GribFileWriter>(e.path)).first;
            s.created.push_back(e.path);
        }
        w->second->write(e.h);
    }

    GribOutput::Stats stats;

    template <class Col>
    void write_state(const std::vector<std::string>& names, Col&& column, int step_hours) {
        using clock = std::chrono::steady_clock;
        auto secs = [](clock::time_point a) { return std::chrono::duration<double>(clock::now() - a).count(); };
        auto t0 = clock::now();
        Top top = run_top(names, column);
        stats.post += secs(t0);
        for (auto& s : streams) {
            // write_initial_state first checks that every field has a template
            std::vector<size_t> fields;
            for (size_t i = 0; i < names.size(); ++i) {
                const IoVariable& v = cfg.variable(names[i]);
                if (v.computed_forcing) continue;
                template_for(s, v);
                fields.push_back(i);
            }
            std::vector<Encoded> enc(fields.size());
            t0 = clock::now();
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t f = 0; f < fields.size(); ++f) {
                const size_t i = fields[f];
                double cpu[4] = {0, 0, 0, 0};
                try {
                    const auto tg = clock::now();
                    const IoVariable& v = cfg.variable(names[i]);
                    auto it = top.changed.find(names[i]);
                    std::vector<float> a = it != top.changed.end() ? it->second : column(i);
                    a = run_output_pps(*s.def, std::move(a));
                    cpu[0] += secs(tg);
                    enc[f] = encode_field(s, v, std::move(a), step_hours, top, cpu);
                } catch (const std::exception& ex) {
                    enc[f].error = ex.what();
                }
                std::lock_guard<std::mutex> lock(stats_mutex);
                stats.cpu_gather += cpu[0];
                stats.cpu_keys += cpu[1];
                stats.cpu_pack += cpu[2];
                stats.cpu_check += cpu[3];
            }
            stats.encode += secs(t0);
            for (const Encoded& e : enc)
                if (!e.error.empty()) {
                    wait_pending();
                    throw std::runtime_error(e.error);
                }
            // Write this output's messages (in field order, as anemoi writes them) while the next
            // output is encoded; outputs write to different files.
            wait_pending();
            pending = std::async(std::launch::async, [this, &s, secs, enc = std::move(enc)]() {
                const auto tw = clock::now();
                for (const Encoded& e : enc) {
                    write_encoded(s, e);
                    stats.bytes += int64_t(e.h.message().second);
                }
                stats.messages += int64_t(enc.size());
                stats.write += secs(tw);
            });
        }
        wait_pending();
    }

    std::future<void> pending;  // the previous output's file writes
    void wait_pending() {
        if (pending.valid()) pending.get();  // rethrows a write error
    }

    mutable std::mutex clone_mutex;
    std::mutex stats_mutex;
};

namespace {
std::vector<GridSource> sources_of(const Model& m) {
    std::vector<GridSource> s;
    for (const auto& c : m.cutout()) s.push_back({c.name, c.mask});
    if (s.empty()) throw std::runtime_error("GribOutput: the model has no cutout masks (converted file too old)");
    return s;
}
}  // namespace

GribOutput::GribOutput(const Model& model, const IoConfig& cfg, const std::string& templates_dir,
                       const std::string& out_dir, const DateTime& t)
    : GribOutput(sources_of(model), model.config().var_output, cfg, templates_dir, out_dir, t) {}

GribOutput::GribOutput(std::vector<GridSource> sources, std::vector<std::string> output_names, const IoConfig& cfg,
                       const std::string& templates_dir, const std::string& out_dir, const DateTime& t)
    : p_(new Impl) {
    if (const char* d = std::getenv("VARDA_COSMO_DEFINITIONS")) set_definitions_path(d);
    p_->cfg = cfg;
    p_->sources = std::move(sources);
    p_->out_names = std::move(output_names);
    p_->templates_dir = templates_dir;
    p_->out_dir = out_dir;
    p_->base = t;
    if (p_->out_names != cfg.output_variables)
        throw std::runtime_error("GribOutput: the model's output variables differ from the io config's");
    int64_t off = 0;
    for (const auto& s : p_->sources) {
        p_->offset.push_back(off);
        p_->count.push_back(count_true(s.mask));
        off += p_->count.back();
    }
    p_->G = off;
    G_ = off;
    if (G_ != cfg.number_of_grid_points)
        throw std::runtime_error("GribOutput: cutout masks give " + std::to_string(G_) + " points, io config says " +
                                 std::to_string(cfg.number_of_grid_points));
    p_->streams.resize(p_->cfg.outputs.size());
    for (size_t i = 0; i < p_->streams.size(); ++i) p_->streams[i].def = &p_->cfg.outputs[i];
}

GribOutput::~GribOutput() = default;

void GribOutput::write_initial(const std::vector<std::string>& names, const std::vector<std::vector<float>>& values) {
    if (!p_->cfg.write_initial_state) return;
    if (names.size() != values.size()) throw std::runtime_error("write_initial: names and values differ in length");
    for (const auto& v : values)
        if (int64_t(v.size()) != p_->G) throw std::runtime_error("write_initial: a field does not have G values");
    p_->write_state(names, [&](size_t i) { return values[i]; }, 0);
}

void GribOutput::write_step(int step_hours, const float* y) {
    const size_t V = p_->out_names.size();
    const size_t G = size_t(p_->G);
    // y is (G, V): transpose it once, in cache-sized blocks, so that every message reads its variable
    // contiguously (a strided gather per message streams all of y through memory V times).
    std::vector<float>& yt = p_->transposed;
    yt.resize(G * V);
    constexpr size_t B = 256;
#pragma omp parallel for schedule(static)
    for (size_t g0 = 0; g0 < G; g0 += B) {
        const size_t g1 = std::min(G, g0 + B);
        for (size_t j = 0; j < V; ++j)
            for (size_t g = g0; g < g1; ++g) yt[j * G + g] = y[g * V + j];
    }
    p_->write_state(
        p_->out_names, [&](size_t j) { return std::vector<float>(yt.begin() + j * G, yt.begin() + (j + 1) * G); },
        step_hours);
}

void GribOutput::close() {
    for (auto& s : p_->streams)
        for (auto& [path, w] : s.writers) w->close();
}

GribOutput::Stats GribOutput::stats() const { return p_->stats; }

std::vector<std::string> GribOutput::files() const {
    std::vector<std::string> out;
    for (const auto& s : p_->streams) out.insert(out.end(), s.created.begin(), s.created.end());
    return out;
}

}  // namespace varda::io
