#include "varda/rollout.h"

#include <omp.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace varda {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("rollout: " + what);
}

}  // namespace

const char* to_string(ConstantSource c) {
    return c == ConstantSource::Newest ? "newest (in place, anemoi-inference 9505dd0)"
                                       : "oldest (roll, anemoi-inference eaeb36fa)";
}

StateUpdate StateUpdate::make(int T, int64_t G, int V_in, int V_out, const std::vector<int>& prog_in,
                              const std::vector<int>& prog_out, const std::vector<int>& dyn_in,
                              const std::vector<int>& const_in, ConstantSource cs) {
    require(T >= 1 && G >= 0 && V_in > 0 && V_out > 0, "bad sizes");
    require(prog_in.size() == prog_out.size(), "residual_input and residual_output differ in length");
    StateUpdate u;
    u.T = T;
    u.G = G;
    u.V_in = V_in;
    u.V_out = V_out;
    u.F = static_cast<int>(dyn_in.size());
    u.constants = cs;
    u.kind.assign(V_in, int8_t(-1));
    u.src.assign(V_in, -1);
    auto claim = [&](int c, Kind k, int s, const char* what) {
        require(c >= 0 && c < V_in, std::string(what) + " index " + std::to_string(c) + " out of range");
        require(u.kind[c] < 0, "input channel " + std::to_string(c) + " is listed twice (" + what + ")");
        u.kind[c] = k;
        u.src[c] = s;
    };
    for (size_t i = 0; i < prog_in.size(); ++i) {
        require(prog_out[i] >= 0 && prog_out[i] < V_out, "residual_output index out of range");
        claim(prog_in[i], Prognostic, prog_out[i], "prognostic");
    }
    for (size_t k = 0; k < dyn_in.size(); ++k) claim(dyn_in[k], Forcing, static_cast<int>(k), "dynamic forcing");
    for (int c : const_in) claim(c, Kept, -1, "constant forcing");
    for (int c = 0; c < V_in; ++c)
        require(u.kind[c] >= 0, "input channel " + std::to_string(c) +
                                    " is neither prognostic, dynamic forcing nor constant: anemoi-inference would"
                                    " refuse to roll it out");
    return u;
}

StateUpdate StateUpdate::from_config(const Config& cfg, ConstantSource cs) {
    require(cfg.has_forcings,
            "the model config has no `forcings` section (dynamic/constant forcing names and input indices,"
            " timestep_hours); convert the checkpoint with a converter that writes it (docs/format.md §1.2)");
    return make(cfg.n_input_steps, cfg.n_data_nodes, cfg.n_input_vars, cfg.n_output_vars, cfg.residual_input,
                cfg.residual_output, cfg.forcing_dynamic_index, cfg.forcing_constant_index, cs);
}

void StateUpdate::apply(float* w, const float* y, const float* f) const {
    require(F == 0 || f != nullptr, "forcings missing");
    const int64_t slot = G * V_in;
    float* newest = w + int64_t(T - 1) * slot;
    const bool from_oldest = constants == ConstantSource::Oldest;
#pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; ++g) {
        float* col0 = w + g * V_in;  // oldest slot, grid point g
        for (int c = 0; c < V_in; ++c) {
            const float oldest = col0[c];
            // advance_map "inin": slot t <- slot t+1, ascending (each source is read before it is
            // overwritten); the newest slot is not touched by it.
            for (int t = 0; t + 1 < T; ++t) col0[t * slot + c] = col0[(t + 1) * slot + c];
            switch (kind[c]) {
                case Prognostic: newest[g * V_in + c] = y[g * V_out + src[c]]; break;  // "outin"
                case Forcing: newest[g * V_in + c] = f[g * F + src[c]]; break;
                default:  // constant in time: in place keeps the newest slot's own value
                    if (from_oldest) newest[g * V_in + c] = oldest;  // eaeb36fa: what roll(-1) brings round
                    break;
            }
        }
    }
}

DateTime valid_date(const DateTime& date0, int step, int timestep_hours) {
    return add_hours(date0, int64_t(step + 1) * timestep_hours);
}

void check_rollout_model(const Model& m) {
    const Config& c = m.config();
    StateUpdate::from_config(c);  // throws with a clear message
    require(c.timestep_hours > 0, "the config's forcings section has no timestep_hours");
    require(c.n_output_steps == 1, "only models with one output step are supported");
    require(int64_t(m.latitudes().size()) == c.n_data_nodes && int64_t(m.longitudes().size()) == c.n_data_nodes,
            "the model file has no grid.latitudes / grid.longitudes (needed for computed forcings); use a file"
            " from converted-new/");
}

Rollout::Rollout(Model& model, ConstantSource cs) : m_(model) {
    check_rollout_model(model);
    upd_ = StateUpdate::from_config(model.config(), cs);
    win_.resize(size_t(upd_.T) * upd_.G * upd_.V_in);
    y_.resize(size_t(upd_.G) * upd_.V_out);
    f_.resize(size_t(upd_.G) * upd_.F);
}

void Rollout::run(const RolloutRequest& req) {
    const Config& cfg = m_.config();
    require(req.x0 != nullptr && req.n_steps >= 1, "need x0 and n_steps >= 1");
    std::memcpy(win_.data(), req.x0, win_.size() * sizeof(float));
    times_.assign(req.n_steps, {});
    const size_t fstep = size_t(upd_.G) * upd_.F;
    for (int s = 0; s < req.n_steps; ++s) {
        if (req.on_window) req.on_window(s, win_.data());
        double t0 = now_s();
        m_.predict_step(win_.data(), y_.data(), nullptr);
        times_[s].predict = now_s() - t0;
        const DateTime valid = valid_date(req.date0, s, cfg.timestep_hours);
        if (s + 1 < req.n_steps) {  // anemoi-inference: no update after the last step
            const float* f = nullptr;
            t0 = now_s();
            if (req.replay_forcings) {
                f = req.replay_forcings + size_t(s) * fstep;
            } else {
                compute_forcings(valid, cfg.forcing_dynamic, m_.latitudes().data(), m_.longitudes().data(),
                                 size_t(upd_.G), f_.data());
                f = f_.data();
                times_[s].forcings = now_s() - t0;
            }
            t0 = now_s();
            upd_.apply(win_.data(), y_.data(), f);
            times_[s].update = now_s() - t0;
        }
        if (req.on_step) req.on_step(s, valid, y_.data());
    }
}

}  // namespace varda
