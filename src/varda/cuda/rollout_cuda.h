// Device-resident autoregressive rollout on the CUDA backend, engineered like a decode loop
// (docs/rollout-engine.md). Same semantics and same results as the host-level varda::Rollout
// (src/varda/rollout.h): bitwise equal on the same backend, since both run the backend's
// normalise -> forward_normalised -> denormalise kernels and the update only copies values.
//
//   auto model = varda::Model::load(path, opts);          // opts.device = Device::CUDA
//   varda::cuda::DeviceRollout r(*model);                 // buffers, pinned memory, CUDA graphs
//   varda::RolloutRequest q;  q.x0 = x0; q.date0 = d0; q.n_steps = 24; q.on_step = cb;
//   r.run(q);
//
// Per step s (parity p = s % 2), all enqueued without a host synchronisation:
//   main stream:  wait forcing s uploaded; wait the D2H copy of step s-2 out of y[p];
//                 replay graph[p] = normalise(win) -> forward_normalised -> denormalise -> y[p]
//                                   -> state update(win, y[p], f[p])   (last step: no update)
//   D2H stream:   y[p] -> pinned host y[p] (after the graph)
//   host:         computes the forcings of step s+1 into pinned f[1-p] while the GPU runs step s,
//   H2D stream:   uploads them to f[1-p] (after step s-1, the last reader of f[1-p], is done);
//   host:         waits for the copy of step s-1 and calls on_step(s-1) while step s runs.
// The only host waits in the loop are the ones the callback needs (and, with on_window, the
// window copy each step: verification only).
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "varda/rollout.h"
#include "varda/varda.h"

namespace varda::cuda {

class CudaBackend;

// The state update on the device: window (T, G, V_in) in place, y (G, V_out), forcing (G, F)
// (may be null if no channel is a forcing). kind / src: device copies of StateUpdate::kind / src;
// constants: StateUpdate::constants. Same per-element semantics as StateUpdate::apply (exposed for
// the unit test).
void rollout_update(float* window, const float* y, const float* forcing, const int8_t* kind, const int* src, int T,
                    int64_t G, int V_in, int V_out, int F, ConstantSource constants, cudaStream_t s);

class DeviceRollout {
public:
    // cuda_graph: one CUDA graph per step (default); false runs the same calls eagerly.
    // cs: source of the newest slot's constant inputs (rollout.h; default in place, 9505dd0).
    explicit DeviceRollout(Model& model, bool cuda_graph = true, ConstantSource cs = ConstantSource::Newest);
    ~DeviceRollout();
    DeviceRollout(const DeviceRollout&) = delete;
    DeviceRollout& operator=(const DeviceRollout&) = delete;

    void run(const RolloutRequest& req);

    // Optional consumer of the device-resident outputs (e.g. the temporal downscaler's device loop,
    // docs/downscaler.md §2), called on run()'s thread right after step s is enqueued, with d_y = step
    // s's output (G, V_out) on this rollout's device, complete once `ready` has completed (an event on
    // the rollout's stream). The hook may enqueue reads of d_y on its own streams (after waiting for
    // `ready`) and returns an event recorded after them, or nullptr: step s + 2, the next writer of
    // d_y, waits for that event on the GPU (no host synchronisation). The hook may block the host
    // (e.g. while its own buffers are full); it must not enqueue work on the rollout's streams. The
    // returned event must not be re-recorded before step s + 2 has been enqueued, i.e. before the
    // hook's call for step s + 2. Set it before run(); the numerics are unchanged.
    using StepHook = std::function<cudaEvent_t(int step, const float* d_y, cudaEvent_t ready)>;
    void set_step_hook(StepHook hook) { hook_ = std::move(hook); }

    // GPU time of each step of the last run (ms; CUDA events around each step's graph, read
    // after the run), and the host wall time of the whole loop (s, upload of x0 excluded).
    const std::vector<float>& step_ms() const { return step_ms_; }
    double wall_s() const { return wall_s_; }
    // Host seconds spent computing (or copying, when replayed) the forcings per step.
    const std::vector<double>& forcing_host_s() const { return forcing_s_; }

    // Times `iters` replays of the forward-only graph (normalise -> forward -> denormalise, the
    // last step's graph) on the current window: the like-for-like one-step time. Returns ms each.
    std::vector<float> time_forward(int iters);

    const float* device_window() const { return win_; }  // (T, G, V_in) after the last run
    size_t device_bytes() const { return dev_bytes_; }  // buffers owned by the rollout
    size_t pinned_bytes() const { return pin_bytes_; }

private:
    void enqueue_step(int parity, bool update);  // the body of one step on main_
    void capture();
    float* dalloc(size_t n);
    float* halloc(size_t n);

    Model& m_;
    CudaBackend* be_ = nullptr;
    StateUpdate upd_;
    bool use_graph_ = true;
    int64_t in_n_ = 0, out_n_ = 0, f_n_ = 0;

    float *win_ = nullptr, *xn_ = nullptr, *yn_ = nullptr, *y_[2] = {nullptr, nullptr}, *f_[2] = {nullptr, nullptr};
    int8_t* kind_ = nullptr;
    int* src_ = nullptr;
    float *hy_[2] = {nullptr, nullptr}, *hf_[2] = {nullptr, nullptr}, *hwin_ = nullptr;
    std::vector<void*> dev_allocs_, host_allocs_;
    size_t dev_bytes_ = 0, pin_bytes_ = 0;

    cudaStream_t main_ = nullptr, d2h_ = nullptr, h2d_ = nullptr;
    cudaEvent_t ev_up_[2] = {nullptr, nullptr}, ev_d2h_[2] = {nullptr, nullptr};
    StepHook hook_;
    cudaEvent_t ev_hook_[2] = {nullptr, nullptr};  // per parity: the hook's event for the step that last wrote y_[p]
    std::vector<cudaEvent_t> ev_beg_, ev_end_;
    // graph_[p][u]: parity p (which y / f buffers), u = 1 with the state update, 0 without (last step)
    cudaGraphExec_t graph_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};

    std::vector<float> step_ms_;
    std::vector<double> forcing_s_;
    double wall_s_ = 0;
};

}  // namespace varda::cuda
