// Temporal downscaling loop (docs/downscaler.md): the host-level driver of the 1-hourly
// downscaler (an anemoi interpolator, config model.kind "enc_proc_dec_interpolator") over a
// forecaster run. The counterpart of rollout.h for the forecaster, with the semantics of
// anemoi-inference's `temporal_downscaler` runner:
//
// A forecaster run gives date0 (the analysis date = the newest input slot of the forecaster), its
// input window x0 (1, T_f, G, V_fin) in the FORECASTER's input order, and, per forecaster step
// s = 0..N-1, the output y_s (G, V_fout) valid at date0 + (s + 1) * W (W = the forecaster time step
// = the downscaler's window = (input_times[1] - input_times[0]) * frequency_hours, 6 h).
//
// The downscaler processes the windows k = 0..N-1 covering [date0 + k*W, date0 + (k+1)*W]:
//   input slot 0 = the state at the window start: k = 0 the forecaster's newest input slot
//                  x0[T_f - 1]; k >= 1 the forecast y_{k-1};
//   input slot 1 = the state at the window end:   y_k;
// and every downscaler input channel is filled BY VARIABLE NAME (never by position):
//   prognostic channel      <- the state's variable of the same name (y: forecaster output names;
//                              x0: forecaster input names);
//   constant forcing        <- x0's newest slot, by name, in both slots (they do not change in time);
//   dynamic forcing         <- compute_forcings at the slot's date (window start, window end), on the
//                              downscaler's own grid (same code as the rollout, bitwise anemoi's).
// One predict_step gives T_out states valid at window start + (output_times[t] - input_times[0]) *
// frequency_hours (hours t+1 .. t+6 for Varda). The `tp` output stays the raw per-hour value.
//
// The inputs come straight from memory here; in anemoi-inference they are read back from the
// forecaster's GRIB files (packed), so the two differ by the GRIB packing (docs/downscaler.md §5).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "varda/config.h"
#include "varda/forcings.h"
#include "varda/varda.h"

namespace varda {

// What the downscaler needs to know about the forecaster that feeds it: variable orders, time step,
// grid. `lat`/`lon` (degrees) may be empty, then only the node count is checked.
struct ForecasterInfo {
    std::vector<std::string> in_names, out_names;  // var_input / var_output
    int64_t n_data_nodes = 0;
    int timestep_hours = 0;  // 0 = unknown (taken as the downscaler's window)
    std::vector<double> lat, lon;

    static ForecasterInfo from_config(const Config& c, std::vector<double> lat = {}, std::vector<double> lon = {});
    // Reads only the config and the grid coordinates of a converted forecaster file (no weights).
    static ForecasterInfo from_file(const std::string& converted_path);
};

// Per input channel of the downscaler, where its values come from. Built from the downscaler config
// and the forecaster's names; every input channel must be covered exactly once, otherwise
// construction throws a std::runtime_error naming the channel.
struct DownscaleAssembly {
    enum Kind : int8_t { Prognostic = 0, Constant = 1, Forcing = 2 };

    int64_t G = 0;
    int V_in = 0;      // downscaler input channels
    int V_fin = 0;     // forecaster input variables
    int V_fout = 0;    // forecaster output variables
    int F = 0;         // dynamic forcings
    std::vector<std::string> names;  // [V_in] downscaler input names
    std::vector<int8_t> kind;        // [V_in]
    std::vector<int> from_out;       // [V_in] Prognostic: index in the forecaster output names; else -1
    std::vector<int> from_in;        // [V_in] Prognostic: index in the forecaster input names (the
                                     //        k = 0 start state), Constant: index in them; else -1
    std::vector<int> forcing_col;    // [V_in] Forcing: column in the dynamic-forcing array; else -1
    std::vector<std::string> forcing_names;  // [F] compute_forcings names, in column order

    // From the downscaler config (var_input, forcing_dynamic/_index, forcing_constant/_index).
    // `fc` supplies the forecaster names and (if given) is checked against the downscaler's grid.
    static DownscaleAssembly from_config(const Config& ds, const ForecasterInfo& fc);
    // Explicit lists (tests): channel names, dynamic/constant names with their channel indices.
    static DownscaleAssembly make(int64_t G, const std::vector<std::string>& var_input,
                                  const std::vector<std::string>& dyn_names, const std::vector<int>& dyn_idx,
                                  const std::vector<std::string>& const_names, const std::vector<int>& const_idx,
                                  const std::vector<std::string>& fc_in_names,
                                  const std::vector<std::string>& fc_out_names);

    // Fills one slot (G, V_in) of the downscaler input.
    //   state_out: the state as a forecaster OUTPUT (G, V_fout), or null when the slot is the start
    //   of window 0, then state_in: the forecaster input slot (G, V_fin) is used for the prognostic
    //   channels. x0_newest: the forecaster's newest input slot (G, V_fin), for the constants.
    //   forcing: (G, F) at the slot's date (null when F == 0).
    void fill_slot(float* slot, const float* state_out, const float* state_in, const float* x0_newest,
                   const float* forcing) const;
};

// Dates of window k: start = date0 + k * W, end = start + W; output t valid at
// start + out_offset_hours(t); step hours from date0 = k * W + out_offset_hours(t).
struct DownscaleTimes {
    int window_hours = 0;                  // W
    std::vector<int> out_offset_hours;     // [T_out] (output_times[t] - input_times[0]) * frequency_hours
    static DownscaleTimes from_config(const Config& ds);
    DateTime window_start(const DateTime& date0, int k) const { return add_hours(date0, int64_t(k) * window_hours); }
    DateTime window_end(const DateTime& date0, int k) const { return add_hours(date0, int64_t(k + 1) * window_hours); }
    int step_hours(int k, int t) const { return k * window_hours + out_offset_hours.at(t); }
    DateTime valid(const DateTime& date0, int k, int t) const { return add_hours(date0, step_hours(k, t)); }
};

// window k, output index t (0-based), step hours from date0, valid date, y_t (G, V_out) physical,
// host. `y` is valid only during the call.
using OutputCallback =
    std::function<void(int window, int out_index, int step_hours, const DateTime& valid, const float* y)>;
// The assembled input (2, G, V_in) physical that window k sees, before its predict_step.
using InputCallback = std::function<void(int window, const float* x)>;

struct DownscaleRequest {
    const float* x0 = nullptr;  // forecaster input window (1, T_f, G, V_fin) physical, host; newest = last slot
    int forecaster_T = 0;       // T_f (>= 1)
    DateTime date0;             // date of the newest forecaster input slot
    int n_windows = 1;          // = number of forecaster steps
    // The forecaster's output of step s (G, V_fout) valid at date0 + (s + 1) * W. Called with
    // s = 0, 1, ..., n_windows - 1, in order, once each. Window k uses y_{k-1} (its start slot) and
    // y_k (its end slot); the loop fills the start slot BEFORE it calls y_at(k), so the pointer of
    // step s is read only between the return of y_at(s) and the call of y_at(s + 1) (the last one
    // until run returns); keeping it valid until the call for s + 1 has returned is enough.
    // y_at may block until step s exists (a forecaster running concurrently): the window's forcings
    // and its start slot are done before the wait.
    std::function<const float*(int)> y_at;
    OutputCallback on_output;
    InputCallback on_input;
    // Optional: the host buffer (T_out, G, V_out) that window k's predict_step writes into, asked for
    // once per window (k = 0, 1, ..) right before its predict_step; on_output's `y` then points into
    // it (state t at offset t * G * V_out). The loop never touches the buffer again after the
    // window's last on_output has returned, so the caller may hand the states on (e.g. to a writer
    // thread) instead of copying them, and recycle the buffer later. Unset: an internal buffer, `y`
    // valid only during the call.
    std::function<float*(int)> out_buffer;
};

class Downscaler {
public:
    // Pre-assembled inputs only (run_window): the model must be an interpolator.
    explicit Downscaler(Model& model);
    // Full loop (run) as well. Throws if the model is not an interpolator, the config lacks a
    // `forcings` section or grid coordinates, a channel is not covered exactly once, the grids or
    // time steps differ.
    Downscaler(Model& model, const ForecasterInfo& forecaster);

    // The assembled loop over a forecaster run.
    void run(const DownscaleRequest& req);

    // One window from an already assembled input x (2, G, V_in) physical: predict_step, then
    // on_output for each of the T_out states. `window_start` is the date of input slot 0.
    void run_window(int k, const DateTime& date0, const float* x, const OutputCallback& on_output);
    // The same into the caller's buffer y_out (T_out, G, V_out) (null: the internal one).
    void run_window(int k, const DateTime& date0, const float* x, const OutputCallback& on_output, float* y_out);

    const Config& config() const { return m_.config(); }
    Model& model() { return m_; }
    const DownscaleTimes& times() const { return times_; }
    bool has_assembly() const { return has_asm_; }
    const DownscaleAssembly& assembly() const { return asm_; }
    const std::vector<float>& input() const { return x_; }  // last assembled input
    // Seconds per window: assembly (state copy), forcings, predict_step; waiting in y_at for the
    // window's end state (wait_input) and in out_buffer (wait_output). The device loop
    // (cuda/downscale_cuda.h) reports the same fields with: assemble = GPU time of the slot copy and
    // the scatter, predict = GPU time of the forward pass (no copies), wait_input = waiting for the
    // forecaster's state, plus d2h (device-to-host copy of the window's outputs through the staging
    // ring, from the first chunk to the last), host_copy (summed copies out of the staging ring into
    // the output buffer) and window (wall time from the window's first enqueue to its last on_output).
    struct WindowTimes {
        double assemble = 0, forcings = 0, predict = 0, wait_input = 0, wait_output = 0;
        double d2h = 0, host_copy = 0, window = 0;
    };
    const std::vector<WindowTimes>& window_times() const { return wt_; }

private:
    Model& m_;
    DownscaleTimes times_;
    DownscaleAssembly asm_;
    bool has_asm_ = false;
    std::vector<float> x_, y_, f_start_, f_end_;
    std::vector<WindowTimes> wt_;
};

}  // namespace varda
