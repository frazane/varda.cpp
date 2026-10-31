// GRIB output of the Varda forecaster, as anemoi-inference 9505dd0 writes it with the Hugging Face
// config (docs/spec-io.md §4, docs/io-output.md): top-level post-processors (tp accumulated from the
// start of the forecast in float32, rescaled), then two grib outputs (the `tee`): the LAM points
// through the ICON templates, and the global points scattered back onto N320 through the IFS
// templates with NaN -> -9999 and a bitmap. Keys, key order and the Python str() spelling of each
// value follow anemoi/inference/grib/encoding.py and outputs/grib.py exactly, so that the messages
// are the same bytes when fed the same values.
#pragma once

#include <cstdint>
#include <functional>
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
    // `forecaster_y`: for the temporal downscaler's output (an output with a CopyFromForecaster
    // post-processor), the forecaster's own (G, V_out) state of this step, same variable order; it must
    // be given at every step the post-processor applies to (a positive multiple of its
    // common_leadtime_hours) and is ignored otherwise. nullptr when no output copies.
    void write_step(int step_hours, const float* y, const float* forecaster_y = nullptr);
    // The same, handing y back: `release` is called exactly once, as soon as y and forecaster_y are no
    // longer read (sequential mode: before this call returns, also when it throws; concurrent mode: on a
    // worker thread once the state is gathered, before its messages are encoded, or here if the call
    // throws). Both must stay valid and unchanged until then.
    void write_step(int step_hours, const float* y, const float* forecaster_y, std::function<void()> release);

    // Concurrent encoding of whole states (default: off, one state at a time). With states > 1,
    // write_initial / write_step run the top-level post-processors on the calling thread, in call order
    // (the tp accumulation is the only state carried from one call to the next), and return as soon as
    // the state is queued (at most `states` encoding and one waiting; the call blocks while that is
    // full); `states` worker threads encode whole states at the same time, each with an OpenMP team of
    // `threads` (0: omp_get_max_threads() of the calling thread / states, at least 1). Files are created
    // in call order and every file receives its messages in call order, so the files, their bytes and
    // files() are those of the sequential mode. write_initial copies its values. Call before the first
    // write; calls must come from one thread at a time. An error of a state is rethrown by the next
    // write_initial / write_step, by close(), stats() and files(); after it, queued states are dropped.
    void set_concurrency(int states, int threads = 0);
    int concurrency() const;

    // Closes the files (a file appears under its final name only here; written as <name>.part before).
    // The file writes of the last state are left pending by write_initial/write_step so that they overlap
    // the next state's encoding (concurrent mode: every queued state); close(), stats() and files() wait
    // for them (and rethrow an encoding or write error).
    void close();

    // Files created so far (final names), in creation order, per output.
    std::vector<std::string> files() const;

    // Cumulative seconds since construction: top-level post-processors (calling thread), message
    // encoding (wall time of each state's parallel encoding, summed over states: in concurrent mode the
    // states overlap, so it exceeds the elapsed time), and file writes (summed likewise).
    struct Stats {
        double post = 0, encode = 0, write = 0;
        // CPU seconds summed over threads, per part of a message's encoding
        double cpu_gather = 0, cpu_keys = 0, cpu_pack = 0, cpu_check = 0;
        int64_t messages = 0, bytes = 0;
        // concurrent mode: seconds the writes of a state waited for an earlier state's writes to the same
        // file (summed), and the most states encoding at once
        double write_wait = 0;
        int max_states_in_flight = 0;
    };
    Stats stats() const;
    int64_t n_points() const { return G_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int64_t G_ = 0;
};

}  // namespace varda::io
