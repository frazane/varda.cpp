// The real-data I/O configuration (docs/io-output.md): everything the GRIB layer takes from the
// checkpoint metadata, the Hugging Face patch and the anemoi-inference config, as one JSON file
// written by python/export_io_config.py. The engine never reads YAML.
// Shared by the input side (namer rules, mars entries) and the output side (src/varda/io/output.h).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace varda::io {

// A JSON scalar as the config uses it: null, integer or string (the types a mars entry or a GRIB
// key can have here; floats are rejected by the loader).
struct Scalar {
    enum Kind { Null, Int, Str } kind = Null;
    int64_t i = 0;
    std::string s;
    static Scalar null() { return {}; }
    static Scalar integer(int64_t v) { Scalar x; x.kind = Int; x.i = v; return x; }
    static Scalar string(std::string v) { Scalar x; x.kind = Str; x.s = std::move(v); return x; }
    // Python's str(): "None", "500", "T".
    std::string str() const;
    bool operator==(const Scalar& o) const { return kind == o.kind && i == o.i && s == o.s; }
    bool operator!=(const Scalar& o) const { return !(*this == o); }
};

using ScalarMap = std::vector<std::pair<std::string, Scalar>>;  // insertion-ordered dict
const Scalar* find(const ScalarMap& m, const std::string& key);

// One entry of the patched `variables_metadata`.
struct IoVariable {
    std::string name;
    bool in_model = false;
    ScalarMap mars;  // as patched: date, levelist (may be null), levtype, param (COSMO name), step, time
    bool constant_in_time = false, computed_forcing = false;
    std::string process;      // "" (instantaneous) or "accumulation"
    int period_hours = 0;     // accumulation only
    std::vector<std::string> categories;  // model variables only

    // mars["param"] as a string; empty if absent.
    std::string param() const;
    const Scalar* mars_get(const std::string& key) const { return find(mars, key); }
};

// First matching rule wins (anemoi/inference/inputs/ekd.py RulesNamer). "{level}" in `name` is
// replaced by the field's level.
struct NamerRule {
    std::string short_name, name;
};

struct TemplateIndexEntry {
    // key -> list of accepted values; ALL keys must match (anemoi's SamplesTemplates.template).
    std::vector<std::pair<std::string, std::vector<Scalar>>> match;
    std::string file;  // below the templates directory
};

struct PostProcessorDef {
    enum Type { Accumulate, Rescale, ExtractFromState, ExtractMask, AssignMask } type = Accumulate;
    std::vector<std::string> accumulations;  // Accumulate
    bool allow_negative = false;             // Accumulate
    double scale = 1, offset = 0;            // Rescale
    std::string param;                       // Rescale: selects the fields whose mars param equals it
    std::string source;                      // ExtractFromState
    std::string mask;                        // ExtractMask / AssignMask: "<source>/cutout_mask"
    bool as_slice = false, inverse = false;  // ExtractMask
};

struct PatchRule {
    std::string regex;  // Python re.match semantics: anchored at the start only
    ScalarMap keys;     // update()d into the key dict, in this order
};

struct ModifierDef {
    // The `patches` modifier. anemoi keeps only the LAST patch (upstream bug): `effective`.
    std::vector<std::vector<PatchRule>> patches;
    int effective = 0;
};

struct OutputDef {
    std::string name;            // "lam" | "global"
    std::string path;            // as configured: outputs/{dateTime}_{step:03}.grib
    std::string path_effective;  // after format_dataset_name: outputs/{dateTime}_{step}.grib
    ScalarMap encoding;          // typeOfGeneratingProcess=2
    std::string template_index_source;
    std::vector<TemplateIndexEntry> templates;
    std::vector<PostProcessorDef> post_processors;
    std::vector<ModifierDef> modifiers;
    bool check_encoding = true;
    double missing_value = -9999;
};

struct IoConfig {
    std::string schema, grid;
    std::vector<double> area;
    int64_t number_of_grid_points = 0;
    int timestep_hours = 0, lead_time_hours = 0;
    bool write_initial_state = false, allow_nans = false;
    std::vector<std::string> cutout_sources;

    std::vector<IoVariable> variables;          // variables_metadata order (98)
    std::vector<std::string> model_variables;   // dataset variable order (80)
    std::vector<std::string> output_variables;  // model output order (67)
    std::vector<std::string> initial_state_fields;  // 66 prognostics then z lsm slor sdor
    std::vector<NamerRule> kenda_namer_rules;
    std::vector<PostProcessorDef> post_processors;  // top level, config order
    std::vector<OutputDef> outputs;                 // lam, global

    // Throws std::runtime_error naming the problem; strict about unknown keys and types.
    static IoConfig load(const std::string& path);
    static IoConfig parse(const std::string& json_text);

    const IoVariable& variable(const std::string& name) const;  // throws if unknown
    const IoVariable* find_variable(const std::string& name) const;
    const OutputDef& output(const std::string& name) const;

    // KENDA namer (docs/spec-io.md §1.6): the anemoi name for a GRIB shortName at `level`
    // (hPa for pressure levels), or nullopt if no rule matches (default namer applies).
    std::optional<std::string> kenda_name(const std::string& short_name, long level) const;

private:
    std::vector<std::pair<std::string, size_t>> var_index_;
};

}  // namespace varda::io
