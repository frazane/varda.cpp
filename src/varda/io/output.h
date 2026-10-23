// GRIB output of the Varda forecaster, as anemoi-inference 9505dd0 writes it with the Hugging Face
// config (docs/spec-io.md §4, docs/io-output.md): top-level post-processors (tp accumulated from the
// start of the forecast in float32, rescaled), then two grib outputs (the `tee`): the LAM points
// through the ICON templates, and the global points scattered back onto N320 through the IFS
// templates with NaN -> -9999 and a bitmap. Keys, key order and the Python str() spelling of each
// value follow anemoi/inference/grib/encoding.py and outputs/grib.py exactly, so that the messages
// are the same bytes when fed the same values.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "varda/forcings.h"
#include "varda/io/grib.h"
#include "varda/io/io_config.h"
#include "varda/varda.h"

namespace varda::io {

// ---- pieces, public for tests and for the compare tool ---------------------------------------------

// Python's int() / float() on a string (anemoi's _param): true if the whole string parses.
bool py_parses_as_int(const std::string& s);
bool py_parses_as_float(const std::string& s);
// "paramId" | "param" | "shortName": the key anemoi sets a mars param under (encoding.py:_param).
std::string param_key_name(const std::string& param);

// encoding.py:grib_keys (edition 2, not an ensemble) for one variable: the keys in the order they are
// set, with None dropped. `base` is the forecast base time (date/time keys: YYYYMMDD and hour*100,
// minutes dropped), `step_hours` the output step, `accum_start_hours` the start step a post-processor
// registered (accumulate: 0), or nullopt. `encoding` is the output's `encoding` dict.
ScalarMap build_keys(const IoVariable& v, const ScalarMap& encoding, int64_t edition, const DateTime& base,
                     int step_hours, std::optional<int> accum_start_hours);

// The `patches` modifier (outputs/grib.py PatchOutput): the first rule whose regex matches the START of
// the anemoi name (Python re.match) updates the keys (existing keys keep their place, new ones are
// appended). Only the effective patch is used.
ScalarMap apply_patches(const ModifierDef& m, const std::string& variable_name, ScalarMap keys);

// The template index lookup (grib/templates/__init__.py): first entry for which every match key's
// lookup value is in the accepted list. Lookup keys: name, time_processing, the variable's mars keys
// (levtype, param, levelist) and the output's encoding keys. nullptr if nothing matches.
const TemplateIndexEntry* match_template(const OutputDef& out, const IoVariable& v);

// utils/templating.py:render_template for `{key}` (no format spec) expressions; `dateTime` is
// YYYYMMDDHHMM, `step` the end step in hours.
std::string render_path(const std::string& pattern, const std::string& date_time, int64_t step);

// post_processors/accumulate.py with allow_negative=false: acc = f32(acc + max(0, x)) (np.maximum, NaN kept).
void accumulate_into(std::vector<float>& acc, const std::vector<float>& x, bool allow_negative);
// rescale: f32(x * scale + offset)
std::vector<float> rescale(const std::vector<float>& x, double scale, double offset);
// assign_mask: float array of mask.size() filled with NaN, `x` written at the true positions in order.
std::vector<float> assign_mask(const std::vector<float>& x, const std::vector<uint8_t>& mask);
// encoding.py:encode_message NaN handling: NaN -> missing_value, +-inf -> +-FLT_MAX. Returns true if any NaN.
bool nan_to_missing(std::vector<float>& x, float missing_value);

// ---- the writer ------------------------------------------------------------------------------------

struct GridSource {
    std::string name;           // "lam_0", "global"
    std::vector<uint8_t> mask;  // over the source's own points; the data grid is the kept points in order
};

class GribOutput {
public:
    // `templates_dir`: the directory holding the files named by the template indices (the HF `templates/`).
    // `out_dir`: where the configured relative paths (`outputs/...`) are rooted; created as needed.
    // `t`: forecast base time (date/time keys). Needs the COSMO definitions path set (set_definitions_path,
    // or VARDA_COSMO_DEFINITIONS in the environment, read here) before the first template is loaded.
    GribOutput(const Model& model, const IoConfig& cfg, const std::string& templates_dir, const std::string& out_dir,
               const DateTime& t);
    // Explicit geometry (tests): output_names in model output order.
    GribOutput(std::vector<GridSource> sources, std::vector<std::string> output_names, const IoConfig& cfg,
               const std::string& templates_dir, const std::string& out_dir, const DateTime& t);
    ~GribOutput();
    GribOutput(const GribOutput&) = delete;
    GribOutput& operator=(const GribOutput&) = delete;

    // Step 0 (write_initial_state): the fields as read, each of G values on the merged grid, in the given
    // order (the input state: cfg.initial_state_fields). Top-level post-processors run first.
    void write_initial(const std::vector<std::string>& names, const std::vector<std::vector<float>>& values);
    // One step: y is (G, V_out), physical, bounded, in model output order (Config::var_output).
    void write_step(int step_hours, const float* y);
    // Closes the files (a file appears under its final name only here; written as <name>.part before).
    void close();

    // Files created so far (final names), in creation order, per output.
    std::vector<std::string> files() const;
    int64_t n_points() const { return G_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int64_t G_ = 0;
};

}  // namespace varda::io
