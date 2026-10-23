// Device-resident rollout (rollout_cuda.h, docs/rollout-engine.md).
#include "varda/cuda/rollout_cuda.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "varda/cuda/common.h"
#include "varda/cuda/cuda_backend.h"

namespace varda::cuda {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// One thread per (grid point g, input channel c): shifts the column's T slots down by one
// (advance_map "inin") and writes the newest slot from its source ("outin"); a constant input
// keeps the newest slot's own value, or with from_oldest (eaeb36fa's roll) takes the old oldest
// one. Each thread owns its column, so the in-place update is race free. Pure copies: no
// arithmetic, bitwise the host StateUpdate::apply.
__global__ void rollout_update_kernel(float* __restrict__ w, const float* __restrict__ y,
                                      const float* __restrict__ f, const int8_t* __restrict__ kind,
                                      const int* __restrict__ src, int T, int64_t G, int V_in, int V_out, int F,
                                      bool from_oldest) {
    const int64_t n = G * V_in;
    const int64_t slot = n;
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t g = i / V_in;
        const int c = static_cast<int>(i - g * V_in);
        const float oldest = w[i];
        for (int t = 0; t + 1 < T; ++t) w[t * slot + i] = w[(t + 1) * slot + i];
        const int k = kind[c];
        float* nw = w + int64_t(T - 1) * slot + i;
        if (k == StateUpdate::Prognostic) *nw = y[g * V_out + src[c]];
        else if (k == StateUpdate::Forcing) *nw = f[g * F + src[c]];
        else if (from_oldest) *nw = oldest;  // otherwise: the newest slot keeps its own value
    }
}

}  // namespace

void rollout_update(float* window, const float* y, const float* forcing, const int8_t* kind, const int* src, int T,
                    int64_t G, int V_in, int V_out, int F, ConstantSource constants, cudaStream_t s) {
    const int64_t n = G * V_in;
    if (n == 0) return;
    const int threads = 256;
    const int64_t blocks = std::min<int64_t>((n + threads - 1) / threads, 108 * 32);
    rollout_update_kernel<<<static_cast<unsigned>(blocks), threads, 0, s>>>(window, y, forcing, kind, src, T, G, V_in,
                                                                           V_out, F, constants == ConstantSource::Oldest);
    VARDA_CUDA_LAUNCH();
}

float* DeviceRollout::dalloc(size_t n) {
    void* p = nullptr;
    VARDA_CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    dev_allocs_.push_back(p);
    dev_bytes_ += n * sizeof(float);
    return static_cast<float*>(p);
}

float* DeviceRollout::halloc(size_t n) {
    void* p = nullptr;
    VARDA_CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(float), cudaHostAllocDefault));
    host_allocs_.push_back(p);
    pin_bytes_ += n * sizeof(float);
    return static_cast<float*>(p);
}

DeviceRollout::DeviceRollout(Model& model, bool cuda_graph, ConstantSource cs) : m_(model), use_graph_(cuda_graph) {
    check_rollout_model(model);
    be_ = as_cuda(model);
    if (!be_) throw std::runtime_error("DeviceRollout: the model is not on a CUDA device");
    VARDA_CUDA_CHECK(cudaSetDevice(be_->device()));
    upd_ = StateUpdate::from_config(model.config(), cs);
    in_n_ = be_->input_numel();
    out_n_ = be_->output_numel();
    f_n_ = upd_.G * upd_.F;
    if (in_n_ != int64_t(upd_.T) * upd_.G * upd_.V_in || out_n_ != upd_.G * upd_.V_out)
        throw std::runtime_error("DeviceRollout: backend tensor sizes do not match the config");

    win_ = dalloc(in_n_);
    xn_ = dalloc(in_n_);
    yn_ = dalloc(out_n_);
    for (int p = 0; p < 2; ++p) {
        y_[p] = dalloc(out_n_);
        f_[p] = dalloc(std::max<int64_t>(f_n_, 1));
        hy_[p] = halloc(out_n_);
        hf_[p] = halloc(std::max<int64_t>(f_n_, 1));
    }
    VARDA_CUDA_CHECK(cudaMalloc(&kind_, upd_.V_in * sizeof(int8_t)));
    dev_allocs_.push_back(kind_);
    VARDA_CUDA_CHECK(cudaMalloc(&src_, upd_.V_in * sizeof(int)));
    dev_allocs_.push_back(src_);
    VARDA_CUDA_CHECK(cudaMemcpy(kind_, upd_.kind.data(), upd_.V_in * sizeof(int8_t), cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemcpy(src_, upd_.src.data(), upd_.V_in * sizeof(int), cudaMemcpyHostToDevice));
    VARDA_CUDA_CHECK(cudaMemset(win_, 0, in_n_ * sizeof(float)));
    for (int p = 0; p < 2; ++p) VARDA_CUDA_CHECK(cudaMemset(f_[p], 0, std::max<int64_t>(f_n_, 1) * sizeof(float)));

    VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&main_, cudaStreamNonBlocking));
    VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_, cudaStreamNonBlocking));
    VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&h2d_, cudaStreamNonBlocking));
    for (int p = 0; p < 2; ++p) {
        VARDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_up_[p], cudaEventDisableTiming));
        VARDA_CUDA_CHECK(cudaEventCreateWithFlags(&ev_d2h_[p], cudaEventDisableTiming));
    }
    be_->prepare();  // one eager forward: every cuBLASLt plan exists before the capture
    if (use_graph_) capture();
}

DeviceRollout::~DeviceRollout() {
    if (main_) cudaStreamSynchronize(main_);
    if (d2h_) cudaStreamSynchronize(d2h_);
    if (h2d_) cudaStreamSynchronize(h2d_);
    for (auto& gp : graph_)
        for (auto& g : gp)
            if (g) cudaGraphExecDestroy(g);
    for (cudaEvent_t e : ev_beg_) cudaEventDestroy(e);
    for (cudaEvent_t e : ev_end_) cudaEventDestroy(e);
    for (int p = 0; p < 2; ++p) {
        if (ev_up_[p]) cudaEventDestroy(ev_up_[p]);
        if (ev_d2h_[p]) cudaEventDestroy(ev_d2h_[p]);
    }
    for (cudaStream_t s : {main_, d2h_, h2d_})
        if (s) cudaStreamDestroy(s);
    for (void* p : dev_allocs_) cudaFree(p);
    for (void* p : host_allocs_) cudaFreeHost(p);
}

void DeviceRollout::enqueue_step(int p, bool update) {
    be_->normalise(win_, xn_, main_);
    be_->forward_normalised(xn_, yn_, main_);
    be_->denormalise(yn_, y_[p], main_);
    if (update)
        rollout_update(win_, y_[p], upd_.F ? f_[p] : nullptr, kind_, src_, upd_.T, upd_.G, upd_.V_in, upd_.V_out,
                       upd_.F, upd_.constants, main_);
}

void DeviceRollout::capture() {
    for (int p = 0; p < 2; ++p)
        for (int u = 0; u < 2; ++u) {
            cudaGraph_t g;
            VARDA_CUDA_CHECK(cudaStreamBeginCapture(main_, cudaStreamCaptureModeThreadLocal));
            enqueue_step(p, u == 1);
            VARDA_CUDA_CHECK(cudaStreamEndCapture(main_, &g));
            VARDA_CUDA_CHECK(cudaGraphInstantiate(&graph_[p][u], g, 0));
            cudaGraphDestroy(g);
            VARDA_CUDA_CHECK(cudaGraphUpload(graph_[p][u], main_));
        }
    VARDA_CUDA_CHECK(cudaStreamSynchronize(main_));
}

void DeviceRollout::run(const RolloutRequest& req) {
    const Config& cfg = m_.config();
    if (!req.x0 || req.n_steps < 1) throw std::runtime_error("DeviceRollout: need x0 and n_steps >= 1");
    VARDA_CUDA_CHECK(cudaSetDevice(be_->device()));
    const int n = req.n_steps;
    for (int s = static_cast<int>(ev_beg_.size()); s < n; ++s) {
        cudaEvent_t b, e;
        VARDA_CUDA_CHECK(cudaEventCreate(&b));
        VARDA_CUDA_CHECK(cudaEventCreate(&e));
        ev_beg_.push_back(b);
        ev_end_.push_back(e);
    }
    if (req.on_window && !hwin_) hwin_ = halloc(in_n_);
    forcing_s_.assign(n, 0.0);

    // Forcing of step s (written after step s) into pinned buffer hf_[s & 1], then uploaded on h2d_.
    auto produce_forcing = [&](int s) {
        const int b = s & 1;
        const double t0 = now_s();
        if (req.replay_forcings) {
            std::memcpy(hf_[b], req.replay_forcings + size_t(s) * f_n_, f_n_ * sizeof(float));
        } else {
            compute_forcings(valid_date(req.date0, s, cfg.timestep_hours), cfg.forcing_dynamic, m_.latitudes().data(),
                             m_.longitudes().data(), size_t(upd_.G), hf_[b]);
        }
        forcing_s_[s] = now_s() - t0;
        if (s >= 2) VARDA_CUDA_CHECK(cudaStreamWaitEvent(h2d_, ev_end_[s - 2], 0));  // step s-2 read f_[b]
        VARDA_CUDA_CHECK(cudaMemcpyAsync(f_[b], hf_[b], f_n_ * sizeof(float), cudaMemcpyHostToDevice, h2d_));
        VARDA_CUDA_CHECK(cudaEventRecord(ev_up_[b], h2d_));
    };
    auto deliver = [&](int s) {
        VARDA_CUDA_CHECK(cudaEventSynchronize(ev_d2h_[s & 1]));
        if (req.on_step) req.on_step(s, valid_date(req.date0, s, cfg.timestep_hours), hy_[s & 1]);
    };

    // Everything before the loop is set-up: previous run finished, x0 on the device.
    VARDA_CUDA_CHECK(cudaDeviceSynchronize());
    VARDA_CUDA_CHECK(cudaMemcpy(win_, req.x0, in_n_ * sizeof(float), cudaMemcpyHostToDevice));
    const double t_start = now_s();
    if (n > 1) produce_forcing(0);
    for (int s = 0; s < n; ++s) {
        const int p = s & 1;
        const bool update = s + 1 < n;  // anemoi-inference: no update after the last step
        if (req.on_window) {             // verification: the window this step sees
            VARDA_CUDA_CHECK(cudaMemcpyAsync(hwin_, win_, in_n_ * sizeof(float), cudaMemcpyDeviceToHost, main_));
            VARDA_CUDA_CHECK(cudaStreamSynchronize(main_));
            req.on_window(s, hwin_);
        }
        if (update) VARDA_CUDA_CHECK(cudaStreamWaitEvent(main_, ev_up_[p], 0));    // forcing s is on the device
        if (s >= 2) VARDA_CUDA_CHECK(cudaStreamWaitEvent(main_, ev_d2h_[p], 0));  // y_[p] copied out (step s-2)
        VARDA_CUDA_CHECK(cudaEventRecord(ev_beg_[s], main_));
        if (use_graph_) VARDA_CUDA_CHECK(cudaGraphLaunch(graph_[p][update ? 1 : 0], main_));
        else enqueue_step(p, update);
        VARDA_CUDA_CHECK(cudaEventRecord(ev_end_[s], main_));
        VARDA_CUDA_CHECK(cudaStreamWaitEvent(d2h_, ev_end_[s], 0));
        VARDA_CUDA_CHECK(cudaMemcpyAsync(hy_[p], y_[p], out_n_ * sizeof(float), cudaMemcpyDeviceToHost, d2h_));
        VARDA_CUDA_CHECK(cudaEventRecord(ev_d2h_[p], d2h_));
        if (s + 1 < n - 1) {
            // pinned hf_[1-p] was last uploaded for step s-1; that copy finished before step s-1 ran.
            if (s >= 1) VARDA_CUDA_CHECK(cudaEventSynchronize(ev_up_[1 - p]));
            produce_forcing(s + 1);  // host work overlapping step s on the GPU
        }
        if (s >= 1) deliver(s - 1);  // step s-1's output, while step s runs
    }
    deliver(n - 1);
    VARDA_CUDA_CHECK(cudaStreamSynchronize(main_));
    wall_s_ = now_s() - t_start;
    VARDA_CUDA_CHECK(cudaStreamSynchronize(d2h_));
    VARDA_CUDA_CHECK(cudaStreamSynchronize(h2d_));
    step_ms_.assign(n, 0.0f);
    for (int s = 0; s < n; ++s) VARDA_CUDA_CHECK(cudaEventElapsedTime(&step_ms_[s], ev_beg_[s], ev_end_[s]));
}

std::vector<float> DeviceRollout::time_forward(int iters) {
    VARDA_CUDA_CHECK(cudaSetDevice(be_->device()));
    VARDA_CUDA_CHECK(cudaDeviceSynchronize());
    for (int s = static_cast<int>(ev_beg_.size()); s < iters; ++s) {
        cudaEvent_t b, e;
        VARDA_CUDA_CHECK(cudaEventCreate(&b));
        VARDA_CUDA_CHECK(cudaEventCreate(&e));
        ev_beg_.push_back(b);
        ev_end_.push_back(e);
    }
    for (int i = 0; i < iters; ++i) {
        VARDA_CUDA_CHECK(cudaEventRecord(ev_beg_[i], main_));
        if (use_graph_) VARDA_CUDA_CHECK(cudaGraphLaunch(graph_[0][0], main_));
        else enqueue_step(0, false);
        VARDA_CUDA_CHECK(cudaEventRecord(ev_end_[i], main_));
    }
    VARDA_CUDA_CHECK(cudaStreamSynchronize(main_));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) VARDA_CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev_beg_[i], ev_end_[i]));
    return ms;
}

}  // namespace varda::cuda
