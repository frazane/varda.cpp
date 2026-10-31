#include "varda/downscale.h"

#include <omp.h>

#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "varda/format.h"

namespace varda {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("downscale: " + what);
}

std::map<std::string, int> index_of(const std::vector<std::string>& v, const char* what) {
    std::map<std::string, int> m;
    for (size_t i = 0; i < v.size(); ++i)
        require(m.emplace(v[i], int(i)).second, std::string("duplicate ") + what + " name '" + v[i] + "'");
    return m;
}

}  // namespace

ForecasterInfo ForecasterInfo::from_config(const Config& c, std::vector<double> lat, std::vector<double> lon) {
    ForecasterInfo f;
    f.in_names = c.var_input;
    f.out_names = c.var_output;
    f.n_data_nodes = c.n_data_nodes;
    f.timestep_hours = c.timestep_hours;
    f.lat = std::move(lat);
    f.lon = std::move(lon);
    return f;
}

ForecasterInfo ForecasterInfo::from_file(const std::string& path) {
    SafeTensorsFile file(path);
    require(file.metadata("varda_format") == "1", path + ": unsupported varda_format");
    const Config c = parse_config(file.metadata("config"));
    std::vector<double> lat, lon;
    if (file.has("grid.latitudes") && file.has("grid.longitudes")) {
        const std::vector<int64_t> shape{c.n_data_nodes};
        const TensorView& la = file.peek("grid.latitudes");
        const TensorView& lo = file.peek("grid.longitudes");
        require(la.dtype == DType::F64 && lo.dtype == DType::F64 && la.shape == shape && lo.shape == shape,
                path + ": grid.latitudes / grid.longitudes must be F64 [G]");
        lat.assign(la.as<double>(), la.as<double>() + la.numel());
        lon.assign(lo.as<double>(), lo.as<double>() + lo.numel());
    }
    return from_config(c, std::move(lat), std::move(lon));
}

DownscaleAssembly DownscaleAssembly::make(int64_t G, const std::vector<std::string>& var_input,
                                          const std::vector<std::string>& dyn_names, const std::vector<int>& dyn_idx,
                                          const std::vector<std::string>& const_names,
                                          const std::vector<int>& const_idx,
                                          const std::vector<std::string>& fc_in_names,
                                          const std::vector<std::string>& fc_out_names) {
    require(G >= 0 && !var_input.empty(), "bad sizes");
    require(dyn_names.size() == dyn_idx.size(), "forcing_dynamic and forcing_dynamic_index differ in length");
    require(const_names.size() == const_idx.size(), "forcing_constant and forcing_constant_index differ in length");
    DownscaleAssembly a;
    a.G = G;
    a.V_in = int(var_input.size());
    a.V_fin = int(fc_in_names.size());
    a.V_fout = int(fc_out_names.size());
    a.F = int(dyn_names.size());
    a.names = var_input;
    a.kind.assign(a.V_in, int8_t(-1));
    a.from_out.assign(a.V_in, -1);
    a.from_in.assign(a.V_in, -1);
    a.forcing_col.assign(a.V_in, -1);
    a.forcing_names = dyn_names;
    index_of(var_input, "downscaler input");
    const auto in_ix = index_of(fc_in_names, "forecaster input");
    const auto out_ix = index_of(fc_out_names, "forecaster output");
    auto claim = [&](int c, Kind k, const std::string& name, const char* what) {
        require(c >= 0 && c < a.V_in, std::string(what) + " index " + std::to_string(c) + " out of range");
        require(var_input[c] == name, std::string(what) + " '" + name + "' is listed at input channel " +
                                          std::to_string(c) + ", which is '" + var_input[c] + "'");
        require(a.kind[c] < 0, "input channel " + std::to_string(c) + " ('" + name + "') is listed twice (" + what + ")");
        a.kind[c] = k;
    };
    for (size_t k = 0; k < dyn_names.size(); ++k) {
        require(is_forcing_name(dyn_names[k]), "dynamic forcing '" + dyn_names[k] + "' is not a computable forcing");
        claim(dyn_idx[k], Forcing, dyn_names[k], "dynamic forcing");
        a.forcing_col[dyn_idx[k]] = int(k);
    }
    for (size_t k = 0; k < const_names.size(); ++k) {
        claim(const_idx[k], Constant, const_names[k], "constant forcing");
        const auto it = in_ix.find(const_names[k]);
        require(it != in_ix.end(), "constant forcing '" + const_names[k] + "' is not a variable of the forecaster input");
        a.from_in[const_idx[k]] = it->second;
    }
    // Everything else is prognostic: it must come from the forecaster, by name.
    for (int c = 0; c < a.V_in; ++c) {
        if (a.kind[c] >= 0) continue;
        a.kind[c] = Prognostic;
        const auto o = out_ix.find(var_input[c]);
        require(o != out_ix.end(), "input channel " + std::to_string(c) + " ('" + var_input[c] +
                                       "') is neither a dynamic forcing, a constant forcing nor a forecaster output");
        const auto i = in_ix.find(var_input[c]);
        require(i != in_ix.end(), "prognostic input '" + var_input[c] +
                                      "' is not a variable of the forecaster input (window 0 starts from it)");
        a.from_out[c] = o->second;
        a.from_in[c] = i->second;
    }
    return a;
}

DownscaleAssembly DownscaleAssembly::from_config(const Config& ds, const ForecasterInfo& fc) {
    require(ds.is_interpolator(), "the model is not a temporal downscaler (model.kind " + ds.model_kind + ")");
    require(ds.has_forcings, "the downscaler config has no `forcings` section (dynamic/constant forcing names and"
                             " input indices); convert the checkpoint with a converter that writes it");
    require(ds.n_data_nodes == fc.n_data_nodes,
            "grid mismatch: the downscaler has " + std::to_string(ds.n_data_nodes) + " grid points, the forecaster " +
                std::to_string(fc.n_data_nodes));
    return make(ds.n_data_nodes, ds.var_input, ds.forcing_dynamic, ds.forcing_dynamic_index, ds.forcing_constant,
                ds.forcing_constant_index, fc.in_names, fc.out_names);
}

void DownscaleAssembly::fill_slot(float* slot, const float* state_out, const float* state_in, const float* x0_newest,
                                  const float* forcing) const {
    require(F == 0 || forcing != nullptr, "forcings missing");
    require(state_out != nullptr || state_in != nullptr, "no state for the slot");
    require(x0_newest != nullptr, "x0 missing");
#pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; ++g) {
        float* dst = slot + g * V_in;
        const float* so = state_out ? state_out + g * V_fout : nullptr;
        const float* si = state_in ? state_in + g * V_fin : nullptr;
        const float* xc = x0_newest + g * V_fin;
        const float* f = forcing ? forcing + g * F : nullptr;
        for (int c = 0; c < V_in; ++c) {
            switch (kind[c]) {
                case Prognostic: dst[c] = so ? so[from_out[c]] : si[from_in[c]]; break;
                case Constant: dst[c] = xc[from_in[c]]; break;
                default: dst[c] = f[forcing_col[c]]; break;
            }
        }
    }
}

DownscaleTimes DownscaleTimes::from_config(const Config& ds) {
    require(ds.is_interpolator(), "the model is not a temporal downscaler");
    require(ds.n_input_steps == 2 && ds.input_times.size() == 2,
            "only downscalers with two input states (window start and end) are supported");
    DownscaleTimes t;
    t.window_hours = (ds.input_times[1] - ds.input_times[0]) * ds.frequency_hours;
    require(t.window_hours > 0, "empty window");
    for (int ot : ds.output_times) {
        require(ot > ds.input_times[0] && ot <= ds.input_times[1], "output time outside the window");
        t.out_offset_hours.push_back((ot - ds.input_times[0]) * ds.frequency_hours);
    }
    require(int(t.out_offset_hours.size()) == ds.n_output_steps, "output_times size != n_output_steps");
    return t;
}

Downscaler::Downscaler(Model& model) : m_(model) {
    const Config& c = model.config();
    require(c.is_interpolator(), "the model is not a temporal downscaler (model.kind " + c.model_kind + ")");
    times_ = DownscaleTimes::from_config(c);
    // x_ (run) and y_ (run_window without a caller buffer) are allocated on first use: several GB
    // for Varda, whose first touch should not land on the constructing thread.
}

Downscaler::Downscaler(Model& model, const ForecasterInfo& fc) : Downscaler(model) {
    const Config& c = model.config();
    asm_ = DownscaleAssembly::from_config(c, fc);
    has_asm_ = true;
    require(fc.timestep_hours == 0 || fc.timestep_hours == times_.window_hours,
            "the forecaster time step (" + std::to_string(fc.timestep_hours) + " h) differs from the downscaler window (" +
                std::to_string(times_.window_hours) + " h)");
    const size_t G = size_t(c.n_data_nodes);
    require(model.latitudes().size() == G && model.longitudes().size() == G,
            "the downscaler file has no grid.latitudes / grid.longitudes (needed for computed forcings)");
    if (!fc.lat.empty() || !fc.lon.empty()) {
        require(fc.lat.size() == G && fc.lon.size() == G, "forecaster grid coordinates have the wrong size");
        const bool same = std::memcmp(fc.lat.data(), model.latitudes().data(), G * sizeof(double)) == 0 &&
                          std::memcmp(fc.lon.data(), model.longitudes().data(), G * sizeof(double)) == 0;
        require(same, "grid mismatch: the downscaler's latitudes/longitudes differ (bitwise) from the forecaster's");
    }
    f_start_.resize(G * asm_.F);
    f_end_.resize(G * asm_.F);
}

void Downscaler::run_window(int k, const DateTime& date0, const float* x, const OutputCallback& on_output) {
    run_window(k, date0, x, on_output, nullptr);
}

void Downscaler::run_window(int k, const DateTime& date0, const float* x, const OutputCallback& on_output,
                            float* y_out) {
    const Config& c = m_.config();
    if (!y_out) y_.resize(size_t(c.n_output_steps) * c.n_data_nodes * c.n_output_vars);
    float* y = y_out ? y_out : y_.data();
    const double t0 = now_s();
    m_.predict_step(x, y, nullptr);
    if (int(wt_.size()) <= k) wt_.resize(k + 1);
    wt_[k].predict = now_s() - t0;
    const size_t out_n = size_t(c.n_data_nodes) * c.n_output_vars;
    if (!on_output) return;
    for (int t = 0; t < c.n_output_steps; ++t)
        on_output(k, t, times_.step_hours(k, t), times_.valid(date0, k, t), y + size_t(t) * out_n);
}

void Downscaler::run(const DownscaleRequest& req) {
    require(has_asm_, "this Downscaler was built without forecaster information (use run_window)");
    require(req.x0 != nullptr && req.forecaster_T >= 1 && req.n_windows >= 1 && req.y_at, "need x0, forecaster_T, n_windows, y_at");
    const Config& c = m_.config();
    const size_t G = size_t(c.n_data_nodes);
    const size_t slot = G * size_t(c.n_input_vars);
    const float* x0_newest = req.x0 + size_t(req.forecaster_T - 1) * G * asm_.V_fin;
    x_.resize(size_t(c.n_input_steps) * slot);
    wt_.assign(req.n_windows, {});
    const float* y_prev = nullptr;
    // Per window: the forcings and the start slot first (they need nothing new from the forecaster),
    // then y_k (y_at may wait for it), the end slot, predict_step.
    // The dynamic forcings of the window start are those of the previous window's end: same date,
    // same function, so f_start_ <- f_end_ is bitwise what recomputing would give.
    for (int k = 0; k < req.n_windows; ++k) {
        const DateTime start = times_.window_start(req.date0, k), end = times_.window_end(req.date0, k);
        double t0 = now_s();
        if (asm_.F > 0) {
            if (k == 0)
                compute_forcings(start, asm_.forcing_names, m_.latitudes().data(), m_.longitudes().data(), G, f_start_.data());
            else
                f_start_.swap(f_end_);
            compute_forcings(end, asm_.forcing_names, m_.latitudes().data(), m_.longitudes().data(), G, f_end_.data());
        }
        wt_[k].forcings = now_s() - t0;
        const float* ff0 = asm_.F ? f_start_.data() : nullptr;
        const float* ff1 = asm_.F ? f_end_.data() : nullptr;
        t0 = now_s();
        if (k == 0)
            asm_.fill_slot(x_.data(), nullptr, x0_newest, x0_newest, ff0);
        else
            asm_.fill_slot(x_.data(), y_prev, nullptr, x0_newest, ff0);
        double assemble = now_s() - t0;
        t0 = now_s();
        const float* y_cur = req.y_at(k);  // y_prev stays valid until this call has returned
        wt_[k].wait_input = now_s() - t0;
        require(y_cur != nullptr, "y_at(" + std::to_string(k) + ") returned null");
        t0 = now_s();
        asm_.fill_slot(x_.data() + slot, y_cur, nullptr, x0_newest, ff1);
        wt_[k].assemble = assemble + (now_s() - t0);
        y_prev = y_cur;
        if (req.on_input) req.on_input(k, x_.data());
        float* y_out = nullptr;
        if (req.out_buffer) {
            t0 = now_s();
            y_out = req.out_buffer(k);
            wt_[k].wait_output = now_s() - t0;
            require(y_out != nullptr, "out_buffer(" + std::to_string(k) + ") returned null");
        }
        run_window(k, req.date0, x_.data(), req.on_output, y_out);
    }
}

}  // namespace varda
