// Device-resident loop of the temporal downscaler on the CUDA backend (docs/downscaler.md §2): the
// semantics and the results of varda::Downscaler::run (src/varda/downscale.h), with the window inputs
// assembled on the downscaler's GPU from the forecaster's device-resident outputs and the outputs
// copied to the host through a small page-locked staging ring.
//
//   varda::Downscaler ds(*td_model, forecaster_info);                  // checks, assembly table
//   varda::cuda::DeviceDownscaler dl(ds, forecaster_device);          // buffers, streams, events
//   rollout.set_step_hook([&](int s, const float* d_y, cudaEvent_t ready) { return dl.push_device(s, d_y, ready); });
//   std::thread t([&] { dl.run(request); });                           // DownscaleRequest as for ds.run
//   rollout.run(...);  t.join();
//
// Per window k, on run()'s thread, all on the downscaler backend's stream (no host synchronisation):
//   host:   the dynamic forcings at the window end (compute_forcings, as the host loop) into pinned
//           memory, uploaded (the device rollout computes its forcings the same way);
//   slot 0: k = 0: assembled on the host from x0 (DownscaleAssembly::fill_slot) and uploaded once
//           before the first window; k >= 1: a device copy of the previous window's slot 1 (same state
//           y_{k-1}, same constants, forcings of the same date: bitwise what assembling it gives);
//   slot 1: gather_columns from y_k (in the loop's device ring), the constants (resident, taken from
//           x0's newest slot once) and the forcings, per DownscaleAssembly's table;
//   predict: CudaBackend::enqueue_predict (the backend's graph), output in the backend's buffer, then
//           (when memory allows) copied on the device to a hold buffer so the next window can start.
// A drain thread copies each window's (T_out, G, V_out) outputs device-to-host in chunks through
// `staging_slots` pinned slots of `staging_bytes`, then into req.out_buffer(k) with an OpenMP copy of
// `copy_threads` threads (a transposing copy with vmajor_output: (T_out, V_out, G) window buffers), and
// calls on_output(k, t, ..) as soon as state t is complete (on the drain thread, in order). Nothing is
// page-locked beyond the ring (the window buffers stay pageable).
//
// Forecaster states: with a producer device (>= 0), push_device(s, d_y, ready) copies step s's
// output (on the producer's device, complete after `ready`) into a ring of `ring_states` device
// slots on the downscaler's device (cudaMemcpyAsync on one device; cudaMemcpyPeerAsync across
// devices, direct when peer access can be enabled, else staged by the driver), and returns the event
// after that copy, which the producer waits for (GPU side) before overwriting d_y. push_device blocks
// the producer's host thread only while its ring slot still holds a state no window has consumed.
// Without a producer, run() takes the states from req.y_at (host pointers) and uploads them.
#pragma once

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "varda/downscale.h"

namespace varda::cuda {

class CudaBackend;

// out (T, V, G) <- the elements [off, off + len) of a (T, G, V) array held at src[0, len) (any range, also
// across states and rows); the drain's copy with vmajor_output. The calling thread's OpenMP team.
void transposed_copy(float* out, const float* src, size_t off, size_t len, size_t G, size_t V);

struct DeviceLoopOptions {
    int ring_states = 2;                          // forecaster states in flight on the device (>= 1)
    size_t staging_bytes = size_t(64) << 20;      // one pinned staging slot
    int staging_slots = 4;                        // pinned staging slots (>= 1)
    int copy_threads = 8;                         // OpenMP threads of the host copy out of the staging ring
    int hold_output = -1;                         // -1: when the device has room, 0: never, 1: always
    // true: the host copy out of the staging ring transposes each state, so that the window buffers are
    // variable-major, (T_out, V_out, G), and on_output's state is (V_out, G) (GribOutput::write_step_vmajor
    // reads it in place); false: (T_out, G, V_out), as Downscaler::run gives them. Same values.
    bool vmajor_output = false;
};

class DeviceDownscaler {
public:
    // `ds` must have its assembly (the forecaster-info constructor) and a model on CUDA.
    // producer_device: CUDA ordinal of the device the forecaster's outputs live on (push_device),
    // or -1 (states from DownscaleRequest::y_at). Allocates everything here (device buffers, pinned
    // staging, streams, events): construct it before other threads use the GPU(s).
    DeviceDownscaler(Downscaler& ds, int producer_device, const DeviceLoopOptions& o = {});
    ~DeviceDownscaler();
    DeviceDownscaler(const DeviceDownscaler&) = delete;
    DeviceDownscaler& operator=(const DeviceDownscaler&) = delete;

    // Producer side (thread-safe w.r.t. run()): see the header comment. Returns nullptr after cancel().
    // Steps are pushed in order s = 0, 1, .., once each, for the current or the next run().
    cudaEvent_t push_device(int s, const float* d_y, cudaEvent_t ready);
    // Unblocks both sides (an error elsewhere): push_device returns nullptr, run() throws.
    void cancel();

    // The loop; same request as Downscaler::run (y_at only without a producer). on_input, if set,
    // receives the device-assembled (2, G, V_in) copied to the host (verification; synchronises).
    // on_output and out_buffer are called on the drain thread, in window and output order.
    void run(const DownscaleRequest& req);

    const std::vector<Downscaler::WindowTimes>& window_times() const { return wt_; }
    double setup_seconds() const { return setup_s_; }  // constants and window 0's slot 0 (host + upload)
    std::string feed_path() const;   // "same-device", "peer", "peer-staged-by-driver" or "host"
    bool output_held() const { return hold_ != nullptr; }
    const DeviceLoopOptions& options() const { return o_; }
    size_t device_bytes() const { return dev_bytes_; }
    size_t pinned_bytes() const { return pin_bytes_; }
    int device() const { return dev_; }
    size_t free_bytes_at_setup() const { return free_at_setup_; }

private:
    void drain(const DownscaleRequest& req, int n);
    float* dalloc(size_t n);
    float* halloc(size_t n);

    Downscaler& ds_;
    CudaBackend* be_ = nullptr;
    DeviceLoopOptions o_;
    int dev_ = 0, prod_ = -1;
    bool peer_ = false;
    int64_t G_ = 0;
    int V_in_ = 0, V_fin_ = 0, V_fout_ = 0, F_ = 0, Vc_ = 0, T_out_ = 0;
    size_t fc_n_ = 0, out_n_ = 0, slot_n_ = 0;  // G * V_fout, G * V_out, G * V_in
    std::vector<int> const_cols_;               // forecaster input columns of the constants, compacted

    float *ring_ = nullptr, *cst_ = nullptr, *f_ = nullptr, *hold_ = nullptr;
    int8_t* sel_ = nullptr;
    int* col_ = nullptr;
    float* hf_[2] = {nullptr, nullptr};
    std::vector<float*> stg_;
    std::vector<void*> dev_allocs_, host_allocs_;
    size_t dev_bytes_ = 0, pin_bytes_ = 0, free_at_setup_ = 0;

    cudaStream_t copy_ = nullptr, d2h_ = nullptr;
    std::vector<cudaEvent_t> ev_copied_, ev_consumed_, ev_stg_;
    cudaEvent_t ev_fup_[2] = {nullptr, nullptr}, ev_out_ready_[2] = {nullptr, nullptr}, ev_drain_done_ = nullptr;
    std::vector<cudaEvent_t> ev_asm_b_, ev_asm_e_, ev_pred_b_, ev_pred_e_, ev_d2h_b_, ev_d2h_e_;

    std::mutex mu_;
    std::condition_variable cv_;
    int produced_ = 0, consumed_ = 0, handed_ = 0, drained_ = 0;
    bool cancelled_ = false;
    std::exception_ptr drain_err_;

    std::vector<Downscaler::WindowTimes> wt_;
    std::vector<double> win_t0_;
    std::vector<float> y_host_;  // without out_buffer
    double setup_s_ = 0;
};

}  // namespace varda::cuda
