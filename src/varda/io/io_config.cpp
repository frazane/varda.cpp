#include "varda/io/io_config.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace varda::io {

using ojson = nlohmann::ordered_json;

std::string Scalar::str() const {
    switch (kind) {
        case Null: return "None";
        case Int: return std::to_string(i);
        case Str: return s;
    }
    return {};
}

const Scalar* find(const ScalarMap& m, const std::string& key) {
    for (const auto& kv : m)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

std::string IoVariable::param() const {
    const Scalar* p = mars_get("param");
    return p && p->kind == Scalar::Str ? p->s : std::string();
}

namespace {

[[noreturn]] void bad(const std::string& where, const std::string& what) {
    throw std::runtime_error("io config: " + where + ": " + what);
}

void only_keys(const ojson& j, std::initializer_list<const char*> allowed, const std::string& where) {
    if (!j.is_object()) bad(where, "expected an object");
    std::set<std::string> ok(allowed.begin(), allowed.end());
    for (auto it = j.begin(); it != j.end(); ++it)
        if (!ok.count(it.key())) bad(where, "unknown key '" + it.key() + "'");
    for (const auto& k : ok)
        if (!j.contains(k)) bad(where, "missing key '" + k + "'");
}

const ojson& need(const ojson& j, const char* key, const std::string& where) {
    if (!j.is_object() || !j.contains(key)) bad(where, std::string("missing key '") + key + "'");
    return j.at(key);
}

std::string get_str(const ojson& j, const char* key, const std::string& where) {
    const ojson& v = need(j, key, where);
    if (!v.is_string()) bad(where + "." + key, "expected a string");
    return v.get<std::string>();
}
bool get_bool(const ojson& j, const char* key, const std::string& where) {
    const ojson& v = need(j, key, where);
    if (!v.is_boolean()) bad(where + "." + key, "expected a bool");
    return v.get<bool>();
}
int64_t get_int(const ojson& j, const char* key, const std::string& where) {
    const ojson& v = need(j, key, where);
    if (!v.is_number_integer()) bad(where + "." + key, "expected an integer");
    return v.get<int64_t>();
}
double get_num(const ojson& j, const char* key, const std::string& where) {
    const ojson& v = need(j, key, where);
    if (!v.is_number()) bad(where + "." + key, "expected a number");
    return v.get<double>();
}
std::vector<std::string> str_list(const ojson& v, const std::string& where) {
    if (!v.is_array()) bad(where, "expected a list");
    std::vector<std::string> out;
    for (const auto& e : v) {
        if (!e.is_string()) bad(where, "expected strings");
        out.push_back(e.get<std::string>());
    }
    return out;
}

Scalar to_scalar(const ojson& v, const std::string& where) {
    if (v.is_null()) return Scalar::null();
    if (v.is_number_integer()) return Scalar::integer(v.get<int64_t>());
    if (v.is_string()) return Scalar::string(v.get<std::string>());
    bad(where, "only null, integer and string values are supported");
}

ScalarMap to_scalar_map(const ojson& v, const std::string& where) {
    if (!v.is_object()) bad(where, "expected an object");
    ScalarMap m;
    for (auto it = v.begin(); it != v.end(); ++it) m.emplace_back(it.key(), to_scalar(it.value(), where + "." + it.key()));
    return m;
}

PostProcessorDef parse_pp(const ojson& j, const std::string& where) {
    PostProcessorDef p;
    const std::string t = get_str(j, "type", where);
    if (t == "accumulate_from_start_of_forecast") {
        only_keys(j, {"type", "accumulations", "allow_negative"}, where);
        p.type = PostProcessorDef::Accumulate;
        p.accumulations = str_list(j.at("accumulations"), where + ".accumulations");
        p.allow_negative = get_bool(j, "allow_negative", where);
    } else if (t == "rescale") {
        only_keys(j, {"type", "scale", "offset", "param"}, where);
        p.type = PostProcessorDef::Rescale;
        p.scale = get_num(j, "scale", where);
        p.offset = get_num(j, "offset", where);
        p.param = get_str(j, "param", where);
    } else if (t == "extract_from_state") {
        only_keys(j, {"type", "source"}, where);
        p.type = PostProcessorDef::ExtractFromState;
        p.source = get_str(j, "source", where);
    } else if (t == "extract_mask") {
        only_keys(j, {"type", "mask", "as_slice", "inverse"}, where);
        p.type = PostProcessorDef::ExtractMask;
        p.mask = get_str(j, "mask", where);
        p.as_slice = get_bool(j, "as_slice", where);
        p.inverse = get_bool(j, "inverse", where);
    } else if (t == "assign_mask") {
        only_keys(j, {"type", "mask", "fill_value"}, where);
        p.type = PostProcessorDef::AssignMask;
        p.mask = get_str(j, "mask", where);
        if (get_str(j, "fill_value", where) != "nan") bad(where, "only fill_value 'nan' is supported");
    } else {
        bad(where, "unknown post-processor type '" + t + "'");
    }
    return p;
}

}  // namespace

IoConfig IoConfig::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("io config: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

IoConfig IoConfig::parse(const std::string& text) {
    ojson j;
    try {
        j = ojson::parse(text);
    } catch (const std::exception& e) {
        bad("json", e.what());
    }
    only_keys(j,
              {"schema", "source", "grid", "area", "number_of_grid_points", "timestep_hours", "lead_time_hours",
               "write_initial_state", "allow_nans", "cutout_sources", "variables", "model_variables",
               "output_variables", "initial_state_fields", "kenda_namer_rules", "post_processors", "outputs"},
              "top level");
    IoConfig c;
    c.schema = get_str(j, "schema", "top level");
    if (c.schema != "varda-io/1") bad("schema", "unsupported '" + c.schema + "' (expected varda-io/1)");
    c.grid = get_str(j, "grid", "top level");
    for (const auto& a : j.at("area")) {
        if (!a.is_number()) bad("area", "expected numbers");
        c.area.push_back(a.get<double>());
    }
    c.number_of_grid_points = get_int(j, "number_of_grid_points", "top level");
    c.timestep_hours = int(get_int(j, "timestep_hours", "top level"));
    c.lead_time_hours = int(get_int(j, "lead_time_hours", "top level"));
    c.write_initial_state = get_bool(j, "write_initial_state", "top level");
    c.allow_nans = get_bool(j, "allow_nans", "top level");
    c.cutout_sources = str_list(j.at("cutout_sources"), "cutout_sources");
    c.model_variables = str_list(j.at("model_variables"), "model_variables");
    c.output_variables = str_list(j.at("output_variables"), "output_variables");
    c.initial_state_fields = str_list(j.at("initial_state_fields"), "initial_state_fields");

    for (const auto& v : j.at("variables")) {
        const std::string w = "variables[" + std::to_string(c.variables.size()) + "]";
        only_keys(v, {"name", "in_model", "mars", "constant_in_time", "computed_forcing", "process", "period_hours",
                      "categories"}, w);
        IoVariable x;
        x.name = get_str(v, "name", w);
        x.in_model = get_bool(v, "in_model", w);
        x.mars = to_scalar_map(v.at("mars"), w + ".mars");
        x.constant_in_time = get_bool(v, "constant_in_time", w);
        x.computed_forcing = get_bool(v, "computed_forcing", w);
        if (!v.at("process").is_null()) x.process = get_str(v, "process", w);
        if (!x.process.empty() && x.process != "accumulation") bad(w, "process '" + x.process + "' not supported");
        if (!v.at("period_hours").is_null()) x.period_hours = int(get_int(v, "period_hours", w));
        if (!v.at("categories").is_null()) x.categories = str_list(v.at("categories"), w + ".categories");
        c.var_index_.emplace_back(x.name, c.variables.size());
        c.variables.push_back(std::move(x));
    }
    std::sort(c.var_index_.begin(), c.var_index_.end());
    for (size_t i = 1; i < c.var_index_.size(); ++i)
        if (c.var_index_[i].first == c.var_index_[i - 1].first) bad("variables", "duplicate " + c.var_index_[i].first);

    for (const auto& r : j.at("kenda_namer_rules")) {
        only_keys(r, {"match", "name"}, "kenda_namer_rules");
        only_keys(r.at("match"), {"shortName"}, "kenda_namer_rules.match");
        c.kenda_namer_rules.push_back({get_str(r.at("match"), "shortName", "kenda_namer_rules"),
                                       get_str(r, "name", "kenda_namer_rules")});
    }
    for (const auto& p : j.at("post_processors"))
        c.post_processors.push_back(parse_pp(p, "post_processors"));

    for (const auto& o : j.at("outputs")) {
        const std::string w = "outputs[" + std::to_string(c.outputs.size()) + "]";
        only_keys(o, {"name", "kind", "path", "path_effective", "encoding", "template_index", "post_processors",
                      "modifiers", "check_encoding", "missing_value"}, w);
        OutputDef d;
        d.name = get_str(o, "name", w);
        if (get_str(o, "kind", w) != "grib") bad(w, "only grib outputs are supported");
        d.path = get_str(o, "path", w);
        d.path_effective = get_str(o, "path_effective", w);
        d.encoding = to_scalar_map(o.at("encoding"), w + ".encoding");
        d.check_encoding = get_bool(o, "check_encoding", w);
        d.missing_value = get_num(o, "missing_value", w);
        const ojson& ti = o.at("template_index");
        only_keys(ti, {"source", "entries"}, w + ".template_index");
        d.template_index_source = get_str(ti, "source", w);
        for (const auto& e : ti.at("entries")) {
            only_keys(e, {"match", "file"}, w + ".template_index.entries");
            TemplateIndexEntry t;
            t.file = get_str(e, "file", w);
            for (auto it = e.at("match").begin(); it != e.at("match").end(); ++it) {
                std::vector<Scalar> vals;
                if (it.value().is_array())
                    for (const auto& x : it.value()) vals.push_back(to_scalar(x, w + ".match." + it.key()));
                else
                    vals.push_back(to_scalar(it.value(), w + ".match." + it.key()));
                t.match.emplace_back(it.key(), std::move(vals));
            }
            d.templates.push_back(std::move(t));
        }
        for (const auto& p : o.at("post_processors")) d.post_processors.push_back(parse_pp(p, w));
        for (const auto& m : o.at("modifiers")) {
            only_keys(m, {"type", "patches", "effective_patch"}, w + ".modifiers");
            if (get_str(m, "type", w) != "patches") bad(w, "only the patches modifier is supported");
            ModifierDef md;
            md.effective = int(get_int(m, "effective_patch", w));
            for (const auto& patch : m.at("patches")) {
                only_keys(patch, {"variable"}, w + ".patches");
                std::vector<PatchRule> rules;
                for (const auto& r : patch.at("variable")) {
                    only_keys(r, {"regex", "keys"}, w + ".patches.variable");
                    rules.push_back({get_str(r, "regex", w), to_scalar_map(r.at("keys"), w + ".patches.keys")});
                }
                md.patches.push_back(std::move(rules));
            }
            if (md.effective < 0 || md.effective >= int(md.patches.size())) bad(w, "effective_patch out of range");
            d.modifiers.push_back(std::move(md));
        }
        c.outputs.push_back(std::move(d));
    }
    return c;
}

const IoVariable* IoConfig::find_variable(const std::string& name) const {
    auto it = std::lower_bound(var_index_.begin(), var_index_.end(), std::make_pair(name, size_t(0)));
    if (it == var_index_.end() || it->first != name) return nullptr;
    return &variables[it->second];
}

const IoVariable& IoConfig::variable(const std::string& name) const {
    const IoVariable* v = find_variable(name);
    if (!v) throw std::runtime_error("io config: unknown variable '" + name + "'");
    return *v;
}

const OutputDef& IoConfig::output(const std::string& name) const {
    for (const auto& o : outputs)
        if (o.name == name) return o;
    throw std::runtime_error("io config: unknown output '" + name + "'");
}

std::optional<std::string> IoConfig::kenda_name(const std::string& short_name, long level) const {
    for (const auto& r : kenda_namer_rules) {
        if (r.short_name != short_name) continue;
        std::string n = r.name;
        const std::string tag = "{level}";
        for (size_t p = n.find(tag); p != std::string::npos; p = n.find(tag, p))
            n.replace(p, tag.size(), std::to_string(level));
        return n;
    }
    return std::nullopt;
}

}  // namespace varda::io
