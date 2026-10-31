// Device-resident temporal-downscaler loop (downscale_cuda.h, docs/downscaler.md §2). Host code only
// (C++ with OpenMP for the host copies); the kernel it launches is in kernels.cu.
#include "varda/cuda/downscale_cuda.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "varda/cuda/common.h"
#include "varda/cuda/cuda_backend.h"
#include "varda/cuda/kernels.h"

namespace varda::cuda {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("device downscaler: " + what);
}

// dst[0, n) = src[0, n) with the calling thread's OpenMP team, one contiguous piece per thread.
void parallel_copy(float* dst, const float* src, size_t n) {
#pragma omp parallel
    {
        const size_t t = size_t(omp_get_thread_num()), nt = size_t(omp_get_num_threads());
        const size_t lo = n * t / nt, hi = n * (t + 1) / nt;
        if (hi > lo) std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(float));
    }
}

cudaEvent_t make_event(bool timing) {
    cudaEvent_t e = nullptr;
    VARDA_CUDA_CHECK(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming));
    return e;
}

double elapsed_s(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    VARDA_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return 1e-3 * ms;
}

}  // namespace

float* DeviceDownscaler::dalloc(size_t n) {
    void* p = nullptr;
    VARDA_CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float)));
    dev_allocs_.push_back(p);
    dev_bytes_ += std::max<size_t>(n, 1) * sizeof(float);
    return static_cast<float*>(p);
}

float* DeviceDownscaler::halloc(size_t n) {
    void* p = nullptr;
    VARDA_CUDA_CHECK(cudaHostAlloc(&p, std::max<size_t>(n, 1) * sizeof(float), cudaHostAllocDefault));
    host_allocs_.push_back(p);
    pin_bytes_ += std::max<size_t>(n, 1) * sizeof(float);
    return static_cast<float*>(p);
}

DeviceDownscaler::DeviceDownscaler(Downscaler& ds, int producer_device, const DeviceLoopOptions& o)
    : ds_(ds), o_(o), prod_(producer_device) {
    require(ds.has_assembly(), "the Downscaler was built without forecaster information");
    be_ = as_cuda(ds.model());
    require(be_ != nullptr, "the downscaler model is not on a CUDA device");
    require(o_.ring_states >= 1 && o_.staging_slots >= 1 && o_.staging_bytes >= 4096 && o_.copy_threads >= 1,
            "bad options");
    const DownscaleAssembly& a = ds.assembly();
    const Config& c = ds.config();
    dev_ = be_->device();
    G_ = a.G;
    V_in_ = a.V_in;
    V_fin_ = a.V_fin;
    V_fout_ = a.V_fout;
    F_ = a.F;
    T_out_ = c.n_output_steps;
    fc_n_ = size_t(G_) * size_t(V_fout_);
    out_n_ = size_t(G_) * size_t(c.n_output_vars);
    slot_n_ = size_t(G_) * size_t(V_in_);
    require(c.n_input_steps == 2 && be_->input_numel() == int64_t(2 * slot_n_) &&
                be_->output_numel() == int64_t(size_t(T_out_) * out_n_),
            "backend tensor sizes do not match the config");

    // Per input channel: source (0 the state, 1 the compacted constants, 2 the forcings) and column.
    std::vector<int8_t> sel(static_cast<size_t>(V_in_));
    std::vector<int> col(static_cast<size_t>(V_in_));
    for (int ch = 0; ch < V_in_; ++ch) {
        switch (a.kind[size_t(ch)]) {
            case DownscaleAssembly::Prognostic:
                sel[size_t(ch)] = 0;
                col[size_t(ch)] = a.from_out[size_t(ch)];
                break;
            case DownscaleAssembly::Constant:
                sel[size_t(ch)] = 1;
                col[size_t(ch)] = int(const_cols_.size());
                const_cols_.push_back(a.from_in[size_t(ch)]);
                break;
            default:
                sel[size_t(ch)] = 2;
                col[size_t(ch)] = a.forcing_col[size_t(ch)];
                break;
        }
    }
    Vc_ = int(const_cols_.size());

    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    ring_ = dalloc(size_t(o_.ring_states) * fc_n_);
    cst_ = dalloc(size_t(G_) * size_t(std::max(Vc_, 1)));
    f_ = dalloc(size_t(G_) * size_t(std::max(F_, 1)));
    VARDA_CUDA_CHECK(cudaMalloc(&sel_, size_t(V_in_)));
    dev_allocs_.push_back(sel_);
    VARDA_CUDA_CHECK(cudaMalloc(&col_, size_t(V_in_) * sizeof(int)));
    dev_allocs_.push_back(col_);
    VARDA_CUDA_CHECK(cudaMemcpy(sel_, sel.data(), size_t(V_in_), cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(col_, col.data(), size_t(V_in_) * sizeof(int), cudaMemcpyHostToDevice));
    // The hold buffer lets window k+1's forward pass overwrite the backend's output while window k is
    // still being copied out; without it the next forward waits for that copy.
    const size_t hold_bytes = size_t(T_out_) * out_n_ * sizeof(float);
    size_t free_b = 0, total_b = 0;
    VARDA_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    free_at_setup_ = free_b;
    const size_t margin = size_t(2) << 30;
    if (o_.hold_output == 1 || (o_.hold_output < 0 && free_b > hold_bytes + margin))
        hold_ = dalloc(size_t(T_out_) * out_n_);
    for (int i = 0; i < 2; ++i) {
        hf_[i] = halloc(size_t(G_) * size_t(std::max(F_, 1)));
        ev_fup_[i] = make_event(false);
        ev_out_ready_[i] = make_event(false);
    }
    const size_t chunk = o_.staging_bytes / sizeof(float);
    for (int i = 0; i < o_.staging_slots; ++i) {
        stg_.push_back(halloc(chunk));
        ev_stg_.push_back(make_event(false));
    }
    ev_drain_done_ = make_event(false);
    VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_, cudaStreamNonBlocking));
    for (int i = 0; i < o_.ring_states; ++i) ev_consumed_.push_back(make_event(false));
    // One launch of the gather kernel now (one grid point into cst_, overwritten by run()): with lazy
    // module loading its first launch loads it, which waited ~0.65 s for the rollout's kernels when it
    // happened in the first window.
    gather_columns(cst_, 1, 1, sel_, col_, ring_, V_fout_, cst_, std::max(Vc_, 1), f_, std::max(F_, 1), d2h_);
    VARDA_CUDA_CHECK(cudaStreamSynchronize(d2h_));

    if (prod_ >= 0) {
        // the copy stream and its events live on the producer's device
        if (prod_ != dev_) {
            int can = 0;
            VARDA_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, prod_, dev_));
            if (can) {
                VARDA_CUDA_CHECK(cudaSetDevice(prod_));
                const cudaError_t e = cudaDeviceEnablePeerAccess(dev_, 0);
                if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) peer_ = true;
                (void)cudaGetLastError();
            }
        }
        VARDA_CUDA_CHECK(cudaSetDevice(prod_));
        VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
        for (int i = 0; i < o_.ring_states; ++i) ev_copied_.push_back(make_event(false));
        VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    }
}

DeviceDownscaler::~DeviceDownscaler() {
    cudaSetDevice(dev_);
    if (d2h_) cudaStreamSynchronize(d2h_);
    if (be_) cudaStreamSynchronize(be_->stream());
    if (copy_) {
        cudaSetDevice(prod_);
        cudaStreamSynchronize(copy_);
        for (cudaEvent_t e : ev_copied_) cudaEventDestroy(e);
        cudaStreamDestroy(copy_);
        cudaSetDevice(dev_);
    }
    for (auto* v : {&ev_consumed_, &ev_stg_, &ev_asm_b_, &ev_asm_e_, &ev_pred_b_, &ev_pred_e_, &ev_d2h_b_, &ev_d2h_e_})
        for (cudaEvent_t e : *v) cudaEventDestroy(e);
    for (cudaEvent_t e : {ev_fup_[0], ev_fup_[1], ev_out_ready_[0], ev_out_ready_[1], ev_drain_done_})
        if (e) cudaEventDestroy(e);
    if (d2h_) cudaStreamDestroy(d2h_);
    for (void* p : dev_allocs_) cudaFree(p);
    for (void* p : host_allocs_) cudaFreeHost(p);
}

std::string DeviceDownscaler::feed_path() const {
    if (prod_ < 0) return "host";
    if (prod_ == dev_) return "same-device";
    return peer_ ? "peer" : "peer-staged-by-driver";
}

void DeviceDownscaler::cancel() {
    std::lock_guard<std::mutex> lk(mu_);
    cancelled_ = true;
    cv_.notify_all();
}

cudaEvent_t DeviceDownscaler::push_device(int s, const float* d_y, cudaEvent_t ready) {
    require(prod_ >= 0, "push_device without a producer device");
    const int R = o_.ring_states;
    {
        // ring slot s % R last held state s - R: window s - R must have enqueued its read of it
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return consumed_ >= s - R + 1 || cancelled_; });
        if (cancelled_) return nullptr;
    }
    const int slot = s % R;
    VARDA_CUDA_CHECK(cudaSetDevice(prod_));
    if (s >= R) VARDA_CUDA_CHECK(cudaStreamWaitEvent(copy_, ev_consumed_[size_t(slot)], 0));
    if (ready) VARDA_CUDA_CHECK(cudaStreamWaitEvent(copy_, ready, 0));
    float* dst = ring_ + size_t(slot) * fc_n_;
    if (prod_ == dev_)
        VARDA_CUDA_CHECK(cudaMemcpyAsync(dst, d_y, fc_n_ * sizeof(float), cudaMemcpyDeviceToDevice, copy_));
    else
        VARDA_CUDA_CHECK(cudaMemcpyPeerAsync(dst, dev_, d_y, prod_, fc_n_ * sizeof(float), copy_));
    VARDA_CUDA_CHECK(cudaEventRecord(ev_copied_[size_t(slot)], copy_));
    {
        std::lock_guard<std::mutex> lk(mu_);
        produced_ = std::max(produced_, s + 1);
        cv_.notify_all();
    }
    return ev_copied_[size_t(slot)];
}

void DeviceDownscaler::drain(const DownscaleRequest& req, int n) {
    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    omp_set_num_threads(o_.copy_threads);
    const DownscaleTimes& tm = ds_.times();
    const size_t total = size_t(T_out_) * out_n_, chunk = o_.staging_bytes / sizeof(float);
    const size_t n_chunks = (total + chunk - 1) / chunk;
    const size_t S = stg_.size();
    for (int k = 0; k < n; ++k) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return handed_ > k || cancelled_; });
            if (cancelled_) return;
        }
        Downscaler::WindowTimes& w = wt_[size_t(k)];
        double t0 = now_s();
        float* out = nullptr;
        if (req.out_buffer) {
            out = req.out_buffer(k);
            require(out != nullptr, "out_buffer(" + std::to_string(k) + ") returned null");
        } else {
            if (y_host_.size() < total) y_host_.resize(total);
            out = y_host_.data();
        }
        w.wait_output = now_s() - t0;
        const float* src = hold_ ? hold_ : be_->device_output();
        VARDA_CUDA_CHECK(cudaStreamWaitEvent(d2h_, ev_out_ready_[k & 1], 0));
        VARDA_CUDA_CHECK(cudaEventRecord(ev_d2h_b_[size_t(k)], d2h_));
        auto enqueue = [&](size_t i) {
            const size_t off = i * chunk, len = std::min(chunk, total - off);
            VARDA_CUDA_CHECK(cudaMemcpyAsync(stg_[i % S], src + off, len * sizeof(float), cudaMemcpyDeviceToHost, d2h_));
            VARDA_CUDA_CHECK(cudaEventRecord(ev_stg_[i % S], d2h_));
            if (i + 1 == n_chunks) {
                // the last read of the window's device output: it may be overwritten after this
                VARDA_CUDA_CHECK(cudaEventRecord(ev_d2h_e_[size_t(k)], d2h_));
                VARDA_CUDA_CHECK(cudaEventRecord(ev_drain_done_, d2h_));
                std::lock_guard<std::mutex> lk(mu_);
                drained_ = k + 1;
                cv_.notify_all();
            }
        };
        for (size_t i = 0; i < std::min(S, n_chunks); ++i) enqueue(i);
        int next = 0;  // next output state to hand on
        for (size_t i = 0; i < n_chunks; ++i) {
            VARDA_CUDA_CHECK(cudaEventSynchronize(ev_stg_[i % S]));
            const size_t off = i * chunk, len = std::min(chunk, total - off);
            t0 = now_s();
            parallel_copy(out + off, stg_[i % S], len);
            w.host_copy += now_s() - t0;
            if (i + S < n_chunks) enqueue(i + S);
            while (next < T_out_ && size_t(next + 1) * out_n_ <= off + len) {
                if (req.on_output)
                    req.on_output(k, next, tm.step_hours(k, next), tm.valid(req.date0, k, next),
                                  out + size_t(next) * out_n_);
                ++next;
            }
        }
        w.window = now_s() - win_t0_[size_t(k)];
    }
}

void DeviceDownscaler::run(const DownscaleRequest& req) {
    require(req.x0 != nullptr && req.forecaster_T >= 1 && req.n_windows >= 1, "need x0, forecaster_T, n_windows");
    require(prod_ >= 0 || req.y_at, "need y_at (no producer device)");
    const int n = req.n_windows;
    const DownscaleAssembly& a = ds_.assembly();
    const DownscaleTimes& tm = ds_.times();
    Model& m = ds_.model();
    const int R = o_.ring_states;
    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    cudaStream_t st = be_->stream();
    float* x = be_->device_input();
    wt_.assign(size_t(n), {});
    win_t0_.assign(size_t(n), 0.0);
    for (auto* v : {&ev_asm_b_, &ev_asm_e_, &ev_pred_b_, &ev_pred_e_, &ev_d2h_b_, &ev_d2h_e_})
        while (int(v->size()) < n) v->push_back(make_event(true));
    {
        std::lock_guard<std::mutex> lk(mu_);
        handed_ = drained_ = 0;
        drain_err_ = nullptr;
    }

    // Constants (compacted) and window 0's slot 0, on the host from x0's newest slot, uploaded once.
    double t0 = now_s();
    const float* x0n = req.x0 + size_t(req.forecaster_T - 1) * size_t(G_) * size_t(V_fin_);
    {
        std::vector<float> h(std::max(slot_n_, size_t(G_) * size_t(Vc_)));
        if (Vc_ > 0) {
            const int64_t G = G_;
            const int Vc = Vc_, Vf = V_fin_;
            const int* cc = const_cols_.data();
            float* hp = h.data();
#pragma omp parallel for schedule(static)
            for (int64_t g = 0; g < G; ++g)
                for (int j = 0; j < Vc; ++j) hp[g * Vc + j] = x0n[g * Vf + cc[j]];
            VARDA_CUDA_CHECK(cudaMemcpyAsync(cst_, h.data(), size_t(G_) * size_t(Vc_) * sizeof(float),
                                             cudaMemcpyHostToDevice, st));  // pageable: h reusable on return
        }
        std::vector<float> f0;
        if (F_ > 0) {
            f0.resize(size_t(G_) * size_t(F_));
            compute_forcings(tm.window_start(req.date0, 0), a.forcing_names, m.latitudes().data(),
                             m.longitudes().data(), size_t(G_), f0.data());
        }
        a.fill_slot(h.data(), nullptr, x0n, x0n, F_ ? f0.data() : nullptr);
        VARDA_CUDA_CHECK(cudaMemcpyAsync(x, h.data(), slot_n_ * sizeof(float), cudaMemcpyHostToDevice, st));
    }
    setup_s_ = now_s() - t0;

    std::thread drainer([this, &req, n] {
        try {
            drain(req, n);
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu_);
            drain_err_ = std::current_exception();
            cancelled_ = true;
            cv_.notify_all();
        }
    });
    auto wait_for = [&](auto pred) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return pred() || cancelled_; });
        if (cancelled_) throw std::runtime_error("device downscaler: cancelled");
    };
    std::vector<float> xin_host;
    try {
        for (int k = 0; k < n; ++k) {
            Downscaler::WindowTimes& w = wt_[size_t(k)];
            win_t0_[size_t(k)] = now_s();
            // the forcings of the window end (those of its start are the previous window's end)
            t0 = now_s();
            if (F_ > 0) {
                if (k >= 2) VARDA_CUDA_CHECK(cudaEventSynchronize(ev_fup_[k & 1]));  // window k-2's upload
                compute_forcings(tm.window_end(req.date0, k), a.forcing_names, m.latitudes().data(),
                                 m.longitudes().data(), size_t(G_), hf_[k & 1]);
            }
            w.forcings = now_s() - t0;
            // the forecaster's state y_k
            t0 = now_s();
            const float* yk = nullptr;
            const int slot = k % R;
            if (prod_ >= 0) {
                wait_for([&] { return produced_ > k; });
                VARDA_CUDA_CHECK(cudaStreamWaitEvent(st, ev_copied_[size_t(slot)], 0));
                yk = ring_ + size_t(slot) * fc_n_;
            } else {
                const float* h = req.y_at(k);
                require(h != nullptr, "y_at(" + std::to_string(k) + ") returned null");
                VARDA_CUDA_CHECK(cudaMemcpyAsync(ring_, h, fc_n_ * sizeof(float), cudaMemcpyHostToDevice, st));
                yk = ring_;
            }
            w.wait_input = now_s() - t0;
            // assembly: forcings up, slot 0 <- the previous slot 1 (k >= 1), slot 1 <- y_k
            VARDA_CUDA_CHECK(cudaEventRecord(ev_asm_b_[size_t(k)], st));
            if (F_ > 0) {
                VARDA_CUDA_CHECK(cudaMemcpyAsync(f_, hf_[k & 1], size_t(G_) * size_t(F_) * sizeof(float),
                                                 cudaMemcpyHostToDevice, st));
                VARDA_CUDA_CHECK(cudaEventRecord(ev_fup_[k & 1], st));
            }
            if (k >= 1)
                VARDA_CUDA_CHECK(cudaMemcpyAsync(x, x + slot_n_, slot_n_ * sizeof(float), cudaMemcpyDeviceToDevice, st));
            gather_columns(x + slot_n_, V_in_, G_, sel_, col_, yk, V_fout_, cst_, std::max(Vc_, 1), f_,
                           std::max(F_, 1), st);
            VARDA_CUDA_CHECK(cudaEventRecord(ev_asm_e_[size_t(k)], st));
            if (prod_ >= 0) {
                VARDA_CUDA_CHECK(cudaEventRecord(ev_consumed_[size_t(slot)], st));
                std::lock_guard<std::mutex> lk(mu_);
                consumed_ = k + 1;
                cv_.notify_all();
            }
            if (req.on_input) {  // verification: the assembled input, on the host
                xin_host.resize(2 * slot_n_);
                VARDA_CUDA_CHECK(cudaMemcpyAsync(xin_host.data(), x, 2 * slot_n_ * sizeof(float),
                                                 cudaMemcpyDeviceToHost, st));
                VARDA_CUDA_CHECK(cudaStreamSynchronize(st));
                req.on_input(k, xin_host.data());
            }
            // the forward pass; its output (the backend's buffer) must have been copied out of the way
            if (k >= 1 && !hold_) {
                wait_for([&] { return drained_ >= k; });
                VARDA_CUDA_CHECK(cudaStreamWaitEvent(st, ev_drain_done_, 0));
            }
            VARDA_CUDA_CHECK(cudaEventRecord(ev_pred_b_[size_t(k)], st));
            be_->enqueue_predict();
            VARDA_CUDA_CHECK(cudaEventRecord(ev_pred_e_[size_t(k)], st));
            if (hold_) {
                if (k >= 1) {
                    wait_for([&] { return drained_ >= k; });
                    VARDA_CUDA_CHECK(cudaStreamWaitEvent(st, ev_drain_done_, 0));
                }
                VARDA_CUDA_CHECK(cudaMemcpyAsync(hold_, be_->device_output(), size_t(T_out_) * out_n_ * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, st));
            }
            VARDA_CUDA_CHECK(cudaEventRecord(ev_out_ready_[k & 1], st));
            std::lock_guard<std::mutex> lk(mu_);
            handed_ = k + 1;
            cv_.notify_all();
        }
    } catch (...) {
        cancel();
        drainer.join();
        std::lock_guard<std::mutex> lk(mu_);
        if (drain_err_) std::rethrow_exception(drain_err_);
        throw;
    }
    drainer.join();
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (drain_err_) std::rethrow_exception(drain_err_);
        if (cancelled_) throw std::runtime_error("device downscaler: cancelled");
        produced_ = consumed_ = 0;  // ready for another run
    }
    VARDA_CUDA_CHECK(cudaStreamSynchronize(st));
    VARDA_CUDA_CHECK(cudaStreamSynchronize(d2h_));
    for (int k = 0; k < n; ++k) {
        wt_[size_t(k)].assemble = elapsed_s(ev_asm_b_[size_t(k)], ev_asm_e_[size_t(k)]);
        wt_[size_t(k)].predict = elapsed_s(ev_pred_b_[size_t(k)], ev_pred_e_[size_t(k)]);
        wt_[size_t(k)].d2h = elapsed_s(ev_d2h_b_[size_t(k)], ev_d2h_e_[size_t(k)]);
    }
}

}  // namespace varda::cuda
