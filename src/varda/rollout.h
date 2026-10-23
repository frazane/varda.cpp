// Autoregressive rollout with anemoi-inference's semantics (docs/rollout-engine.md;
// $IE_ROOT/docs/rollout.md §1): the host-level loop, usable with any backend (CPU, or CUDA
// through Model::predict_step's host copies). This is the readable oracle of the rollout; the
// device-resident CUDA loop (src/varda/cuda/rollout_cuda.h) is bitwise equal to it on the same
// backend.
//
// The window (1, T, G, V_in) holds PHYSICAL fp32 values in model-input order, oldest slot first.
// Step s (0-based):
//   1. y = predict_step(window)                  (re-normalises the whole window every step)
//   2. unless s is the last step, update the window in place (anemoi-inference 9505dd0, the
//      version the Hugging Face release pins: metadata.advance_map, one output step m = 1):
//        inin:  slot t <- slot t+1 for t = 0 .. T-2, in ascending order (a shift down by one);
//        outin: newest slot, prognostic inputs <- y's prognostic outputs (indices.residual_input /
//                                                 residual_output; physical, bounded);
//               newest slot, dynamic forcings  <- forcings at the valid date of this step's output;
//        newest slot, constant inputs: never written, so they keep the NEWEST slot's own values
//        (ConstantSource::Newest, the default). ConstantSource::Oldest reproduces anemoi-inference
//        eaeb36fa (the Python project's references), whose roll(-1) brought the OLDEST slot's values
//        round. The two agree whenever the constants are equal in all slots (real inputs).
//   3. valid date of output s = date0 + (s + 1) * timestep_hours, date0 = the newest input slot's date.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "varda/config.h"
#include "varda/forcings.h"
#include "varda/varda.h"

namespace varda {

// What the update writes into each input channel of the newest slot. Built from the config:
// every input channel must be exactly one of prognostic, dynamic forcing or constant (as
// anemoi-inference's "check" requires), otherwise construction throws.
// Where the newest slot's constant-in-time inputs come from in the update.
enum class ConstantSource : int8_t {
    Newest,  // kept in place: anemoi-inference 9505dd0 (advance_map), the default
    Oldest,  // inherited through roll(-1): anemoi-inference eaeb36fa
};
const char* to_string(ConstantSource c);

struct StateUpdate {
    enum Kind : int8_t { Kept = 0, Prognostic = 1, Forcing = 2 };

    int T = 0, V_in = 0, V_out = 0, F = 0;
    int64_t G = 0;
    std::vector<int8_t> kind;  // [V_in]
    std::vector<int> src;      // [V_in]: Prognostic -> output index, Forcing -> forcing column, Kept -> unused
    ConstantSource constants = ConstantSource::Newest;

    // Throws std::runtime_error naming the problem if the config has no `forcings` section
    // or the index sets do not partition the inputs.
    static StateUpdate from_config(const Config& cfg, ConstantSource cs = ConstantSource::Newest);
    // Explicit index lists (tests): prog_in[i] <- y[prog_out[i]], dyn_in[k] <- forcing column k,
    // const_in constant in time (source: `cs`).
    static StateUpdate make(int T, int64_t G, int V_in, int V_out, const std::vector<int>& prog_in,
                            const std::vector<int>& prog_out, const std::vector<int>& dyn_in,
                            const std::vector<int>& const_in, ConstantSource cs = ConstantSource::Newest);

    // window: (T, G, V_in) in place; y: (G, V_out); forcing: (G, F) (may be null when F == 0).
    void apply(float* window, const float* y, const float* forcing) const;
};

// Valid date of the output of step s: date0 + (s + 1) * timestep_hours.
DateTime valid_date(const DateTime& date0, int step, int timestep_hours);

// Per-step callback: step index, valid date of the output, y (1, 1, 1, G, V_out) physical, host.
// `y` is valid only during the call.
using StepCallback = std::function<void(int step, const DateTime& valid, const float* y)>;
// Optional: the window (1, T, G, V_in) physical that step s sees (before its predict_step).
using WindowCallback = std::function<void(int step, const float* window)>;

struct RolloutRequest {
    const float* x0 = nullptr;  // (1, T, G, V_in) physical, host
    DateTime date0;             // date of the newest input slot
    int n_steps = 1;
    // Forcings: computed with compute_forcings (default), or replayed from a host array
    // [n_steps - 1, G, n_dynamic] in config `forcings.dynamic` order (forcing s is written after
    // step s, so it belongs to the valid date of output s).
    const float* replay_forcings = nullptr;
    StepCallback on_step;
    WindowCallback on_window;
};

// Host-level rollout: owns the window and output buffers, calls Model::predict_step.
class Rollout {
public:
    // Throws if the model's config has no `forcings` section, no timestep, or the file has no
    // grid coordinates (needed for computed forcings).
    explicit Rollout(Model& model, ConstantSource cs = ConstantSource::Newest);

    void run(const RolloutRequest& req);

    const StateUpdate& update() const { return upd_; }
    const std::vector<float>& window() const { return win_; }  // after a run: the last step's input
    // Seconds spent per step: predict_step, forcings (computed only), state update.
    struct StepTimes { double predict = 0, forcings = 0, update = 0; };
    const std::vector<StepTimes>& times() const { return times_; }

private:
    Model& m_;
    StateUpdate upd_;
    std::vector<float> win_, y_, f_;
    std::vector<StepTimes> times_;
};

// Checks shared by both rollout implementations; throws std::runtime_error with a clear message.
void check_rollout_model(const Model& m);

}  // namespace varda
