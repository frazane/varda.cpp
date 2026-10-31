// varda-forecast: a forecast from real data, GRIB in and GRIB out, as anemoi-inference runs the
// Hugging Face config (docs/spec-io.md): KENDA-CH1 + IFS open data -> x0 (varda::io::build_input),
// the autoregressive rollout (docs/rollout-engine.md), then the lam (ICON templates) and global (IFS
// templates) GRIB files per step, including the initial state (varda::io::GribOutput).
//
//   varda-forecast ( --assets BUNDLE | --model M.safetensors --io-config varda-io.json
//                    --regrid-matrix R.safetensors --templates HF_TEMPLATES_DIR [--definitions D] )
//                  [--downscaler TD.safetensors --downscaler-io-config varda-td-io.json | --no-downscaler]
//                  [--downscaler-device cuda:N]
//                  --out DIR
//                  ( --realdata DIR | --date T --kenda DIR --ifs-prev F.. --ifs F.. --ifs-constants F.. )
//                  [--lead-time-hours H] [--device cuda|cpu|metal] [--precision fp16|fp32] [--no-graph]
//                  [--host-loop] [--allow-nans] [--threads N] [--dump-y] [--write-queue N] [--quiet]
//
// --assets BUNDLE fills the model-release files from one directory (docs/quickstart.md):
// BUNDLE/model.safetensors, BUNDLE/varda-io.json, BUNDLE/regrid-ifs0p25-n320.safetensors,
// BUNDLE/templates and BUNDLE/definitions, and, when present, the temporal downscaler
// BUNDLE/downscaler.safetensors + BUNDLE/varda-td-io.json. An explicit option overrides its bundle entry.
//
// With a downscaler (docs/downscaler.md), the 1-hourly temporal downscaler runs over the 6-hourly rollout
// as the Hugging Face notebook does: the hourly LAM files outputs/varda-single_<dateTime>_<h>.grib
// (h = 0 .. lead time), step 0 from the analysis, every other field of hours 6, 12, .. from the forecaster
// (anemoi's copy-prognostic-from-forecaster) and tp accumulated hourly from the start. The downscaler
// weights load (and its configuration is checked) after the forecaster's set-up, overlapping the input
// preparation. With the device rollout (CUDA) the downscaler runs on its own thread DURING the rollout:
// window k starts as soon as step k is out, on a high-priority stream of the same GPU, and the hourly
// writer thread starts with the rollout (VARDA_TD_OVERLAP=0: after the rollout, as on the other
// engines). Each window's six states reach the hourly writer by pointer (VARDA_TD_BUFFERS window
// buffers, default 2; VARDA_TD_PINNED=1 page-locks them), not copied. The hourly writer encodes
// VARDA_TD_WRITER_STATES hours at once (default 3, VARDA_TD_WRITER_THREADS OpenMP threads each; the tp
// accumulation stays in hour order), byte-identical to one hour at a time (docs/downscaler.md §7).
// --downscaler-device cuda:N (or N) runs the downscaler on CUDA device N, e.g. a second GPU (default: the
// forecaster's device, 0): the two models then neither time-slice one GPU nor share its memory. The device
// loop below then copies the rollout's steps across devices (cudaMemcpyPeerAsync); the host loop takes them
// through host memory (StepFeed) as before. The start-up sequencing (downscaler loaded after the
// forecaster's set-up) is kept. forecast.json records the device.
// On CUDA with the device rollout the downscaler's loop is device-resident (docs/downscaler.md §2):
// the rollout hands each step's output on the GPU to the loop (a device copy into a small ring), the
// window inputs are assembled there, and the outputs come back through a page-locked staging ring
// (VARDA_TD_STAGING_MB per slot, default 64; VARDA_TD_STAGING_SLOTS, default 4) and an OpenMP copy
// (VARDA_TD_COPY_THREADS, default 8) into the window buffers. VARDA_TD_DEVICE_LOOP=0 forces the host
// loop (inputs assembled on the host, predict_step's own copies). VARDA_TD_CHECK_INPUTS=1 compares
// every device-assembled window input with the host assembly, bitwise (verification; slower).
// --quiet leaves the input log (every file and message used) out of stdout; it stays in forecast.json.
//
// DIR receives the GRIB files under the configured relative paths (outputs/<dateTime>_<step>.grib); DIR/forecast.json the provenance and
// timings. --dump-y also writes DIR/y.safetensors (`y.<step>` F32 [G, V_out], the model outputs
// before post-processing, plus `x0` F32 [T, G, V_in] and the variable names in `meta`) for
// comparisons against the reference dumps and for varda-downscale --forecast.
//
// Overlap: the input is prepared on its own thread while the weights load and the rollout is set up;
// GRIB encoding (initial state, then each step) runs on a writer thread during the rollout, the rollout
// callback only copying y into a queue of --write-queue steps (default min(steps, 8); 452 MB each).
#include <omp.h>
#include <sys/stat.h>

#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "nlohmann/json.hpp"
#include "varda/format.h"
#include "varda/io/grib.h"
#include "varda/io/input.h"
#include "varda/io/io_config.h"
#include "varda/io/output.h"
#include "varda/downscale.h"
#include "varda/rollout.h"
#include "varda/varda.h"
#ifdef VARDA_WITH_CUDA
#include "varda/cuda/cuda_backend.h"
#include "varda/cuda/downscale_cuda.h"
#include "varda/cuda/rollout_cuda.h"
#endif

namespace {

using json = nlohmann::json;
using namespace varda::io;

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-forecast (--assets BUNDLE | --model M --io-config J --regrid-matrix R --templates DIR)\n"
                 "         --out DIR (--realdata DIR | --date T --kenda DIR --ifs-prev F.. --ifs F.. --ifs-constants F..)\n"
                 "         [--lead-time-hours H] [--device cuda|cpu|metal] [--precision fp16|fp32] [--no-graph]\n"
                 "         [--host-loop] [--definitions D] [--allow-nans] [--threads N] [--dump-y]\n"
                 "         [--write-queue N] [--quiet]\n"
                 "         [--downscaler TD.safetensors --downscaler-io-config varda-td-io.json | --no-downscaler]\n"
                 "         [--downscaler-device cuda:N]\n"
                 "  --assets BUNDLE: model.safetensors, varda-io.json, regrid-ifs0p25-n320.safetensors, templates/,\n"
                 "                   definitions/ and, if present, downscaler.safetensors + varda-td-io.json of BUNDLE\n"
                 "                   (explicit options override)\n"
                 "  --downscaler-device cuda:N (or N): the downscaler's CUDA device (default: the forecaster's)\n");
    std::exit(2);
}

void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i <= p.size(); ++i) {
        if ((i == p.size() || p[i] == '/') && !cur.empty()) ::mkdir(cur.c_str(), 0755);
        if (i < p.size()) cur += p[i];
    }
}

// dst[0, n) = src[0, n) with `threads` OpenMP threads (contiguous pieces).
void parallel_copy(float* dst, const float* src, size_t n, int threads) {
#pragma omp parallel num_threads(threads)
    {
        const size_t t = size_t(omp_get_thread_num()), nt = size_t(omp_get_num_threads());
        const size_t lo = n * t / nt, hi = n * (t + 1) / nt;
        if (hi > lo) std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(float));
    }
}

int env_int(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}

// Encodes the initial state and the steps in order on its own thread, so GRIB encoding overlaps the
// rollout; the first error is rethrown by finish(). At most `depth` steps wait in the queue (each holds
// a copy of y, G * V_out floats): when the GPU is faster than the encoder, the rollout only blocks
// once the queue is full.
class StepWriter {
public:
    // omp_threads > 0: the OpenMP team size of the writer thread's encoding (0: the default). The
    // files do not depend on it.
    StepWriter(GribOutput& out, size_t n, size_t depth, int omp_threads = 0)
        : out_(out), n_(n), depth_(depth), th_([this, omp_threads] {
              if (omp_threads > 0) omp_set_num_threads(omp_threads);
              loop();
          }) {}
    // Without finish() (an exception unwinds): the queued items are dropped, not encoded (the step in
    // progress completes); every push_ref `done` is still called.
    ~StepWriter() {
        if (th_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                cancelled_ = true;
            }
            stop();
            th_.join();
        }
    }
    // `names` / `values` must stay alive and unchanged until finish().
    void push_initial(const std::vector<std::string>& names, const std::vector<std::vector<float>>& values) {
        Item it;
        it.initial = true;
        it.names = &names;
        it.values = &values;
        enqueue(std::move(it));
    }
    // `forecaster_y`: GribOutput::write_step's forecaster state (the temporal downscaler's 6-hourly steps);
    // not copied, so it must stay alive until finish().
    void push(int step_hours, const float* y, const float* forecaster_y = nullptr) {
        const double t0 = app::now_s();
        Item it;
        it.step_hours = step_hours;
        it.y.assign(y, y + n_);
        it.forecaster_y = forecaster_y;
        copy_s_ += app::now_s() - t0;
        enqueue(std::move(it));
    }
    // The same without the copy: `y` is used in place and must stay alive and unchanged until `done`
    // has been called (once GribOutput no longer reads y: on the writer thread, or on one of its
    // encoding workers in concurrent mode; or dropped after an error; here if the item is refused).
    // This is how the downscaler's window buffers reach the writer.
    void push_ref(int step_hours, const float* y, const float* forecaster_y, std::function<void()> done = {}) {
        Item it;
        it.step_hours = step_hours;
        it.yref = y;
        it.forecaster_y = forecaster_y;
        it.done = std::move(done);
        enqueue(std::move(it));
    }
    void finish() {
        stop();
        th_.join();
        if (err_) std::rethrow_exception(err_);
    }
    double initial_seconds() const { return initial_s_; }
    // seconds the writer thread spent in each write_step (concurrent GribOutput: the top-level
    // post-processors and the wait for a free encoding slot, not the encoding)
    const std::vector<double>& step_seconds() const { return secs_; }
    double blocked_seconds() const { return blocked_s_; }  // time the rollout waited for a free slot
    double copy_seconds() const { return copy_s_; }        // time the rollout callback spent copying y

private:
    struct Item {
        int step_hours = 0;
        bool initial = false;
        const std::vector<std::string>* names = nullptr;
        const std::vector<std::vector<float>>* values = nullptr;
        std::vector<float> y;
        const float* yref = nullptr;  // push_ref: y in place of the copy
        const float* forecaster_y = nullptr;
        std::function<void()> done;   // push_ref: called once the item is written or dropped
    };
    void enqueue(Item it) {
        const double t0 = app::now_s();
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return q_.size() < depth_ || err_; });
        blocked_s_ += app::now_s() - t0;
        if (err_) {
            lk.unlock();
            if (it.done) it.done();
            return;
        }
        q_.push_back(std::move(it));
        cv_.notify_all();
    }
    void stop() {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
        cv_.notify_all();
    }
    void loop() {
        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !q_.empty() || done_; });
                if (q_.empty()) return;
                it = std::move(q_.front());
                q_.pop_front();
                cv_.notify_all();
            }
            if (!err_ && !cancelled_) {  // else drain after an error, or when unwinding
                try {
                    const double t0 = app::now_s();
                    if (it.initial) {
                        out_.write_initial(*it.names, *it.values);
                        initial_s_ = app::now_s() - t0;
                    } else {
                        // GribOutput calls `done` once y is no longer read (its concurrent mode encodes
                        // the state after this call returns); a copied y lives until then.
                        std::function<void()> done;
                        done.swap(it.done);
                        if (it.yref) {
                            out_.write_step(it.step_hours, it.yref, it.forecaster_y, std::move(done));
                        } else {
                            auto keep = std::make_shared<std::vector<float>>(std::move(it.y));
                            out_.write_step(it.step_hours, keep->data(), it.forecaster_y,
                                            [keep, d = std::move(done)] {
                                                if (d) d();
                                            });
                        }
                        secs_.push_back(app::now_s() - t0);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lk(mu_);
                    err_ = std::current_exception();
                    cv_.notify_all();
                }
            }
            if (it.done) it.done();
        }
    }
    GribOutput& out_;
    size_t n_, depth_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> q_;
    bool done_ = false;
    bool cancelled_ = false;  // written under mu_ before stop(); read by the writer after it saw done_
    std::exception_ptr err_;
    double initial_s_ = 0, blocked_s_ = 0, copy_s_ = 0;
    std::vector<double> secs_;
    std::thread th_;
};

// The forecaster steps as they arrive (rollout callback) for a downscaler running concurrently:
// wait(s) blocks until step s was published, false if cancelled first.
class StepFeed {
public:
    void publish(int s) {
        std::lock_guard<std::mutex> lk(mu_);
        avail_ = std::max(avail_, s + 1);
        cv_.notify_all();
    }
    void publish_all(int n) { publish(n - 1); }
    bool wait(int s) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return avail_ > s || cancelled_; });
        return avail_ > s;
    }
    void cancel() {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    int avail_ = 0;
    bool cancelled_ = false;
};

// The downscaler's output buffers, one window (T_out, G, V_out) each, handed to the hourly writer by
// pointer: acquire() blocks while every buffer is in flight (written or waiting to be); the writer
// releases a buffer once the window's last state is written. Page-locked (cudaHostAlloc) when the
// downscaler runs on CUDA, so that predict_step's device-to-host copy is a direct DMA.
class WindowBuffers {
public:
    WindowBuffers(size_t floats, int count, bool pinned) : n_(floats), count_(std::max(count, 1)), pinned_(pinned) {}
    ~WindowBuffers() {
        for (const auto& [b, pinned] : all_) free_buffer(b, pinned);
    }
    WindowBuffers(const WindowBuffers&) = delete;
    WindowBuffers& operator=(const WindowBuffers&) = delete;
    // Allocates every buffer now (on the calling thread; nothing may wait on the pool meanwhile).
    void allocate_all() {
        const double t0 = app::now_s();
        std::vector<std::pair<float*, bool>> fresh;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = int(all_.size()); i < count_; ++i) fresh.emplace_back(nullptr, false);
        }
        for (auto& [b, pinned] : fresh) b = allocate(&pinned);
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& f : fresh) {
            all_.push_back(f);
            free_.push_back(f.first);
        }
        alloc_s_ += app::now_s() - t0;
        cv_.notify_all();
    }
    float* acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !free_.empty() || cancelled_; });
        if (cancelled_) return nullptr;
        float* b = free_.front();
        free_.pop_front();
        return b;
    }
    void release(float* b) {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push_back(b);
        cv_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_ = true;
        cv_.notify_all();
    }
    int count() const { return count_; }
    bool pinned() const { return pinned_; }
    double alloc_seconds() const { return alloc_s_; }

private:
    float* allocate(bool* pinned) {
        *pinned = false;
#ifdef VARDA_WITH_CUDA
        if (pinned_) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, n_ * sizeof(float), cudaHostAllocDefault) == cudaSuccess) {
                *pinned = true;
                return static_cast<float*>(p);
            }
            (void)cudaGetLastError();
            std::fprintf(stderr, "varda-forecast: cudaHostAlloc of %.2f GB failed; pageable window buffers\n",
                         n_ * sizeof(float) / 1e9);
            pinned_ = false;
        }
#endif
        return new float[n_];  // not zeroed: predict_step writes every element
    }
    static void free_buffer(float* b, bool pinned) {
#ifdef VARDA_WITH_CUDA
        if (pinned) {
            cudaFreeHost(b);
            return;
        }
#endif
        (void)pinned;
        delete[] b;
    }
    size_t n_;
    int count_;
    bool pinned_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::pair<float*, bool>> all_;  // buffer, page-locked
    std::deque<float*> free_;
    bool cancelled_ = false;
    double alloc_s_ = 0;
};

// --downscaler-device: "cuda:N" or "N" (a CUDA device ordinal), "cuda" (the forecaster's device: -1).
int parse_cuda_device(const std::string& s) {
    std::string n = s;
    if (n == "cuda") return -1;
    if (n.rfind("cuda:", 0) == 0) n = n.substr(5);
    if (n.empty() || n.find_first_not_of("0123456789") != std::string::npos || n.size() > 4)
        throw std::runtime_error("--downscaler-device '" + s + "': expected cuda:N or a device index N");
    return std::stoi(n);
}

// Makes `dev` the calling thread's current CUDA device for the scope and restores the previous one
// (cudaSetDevice is per thread). Inactive (no CUDA call at all) when `active` is false.
class DeviceScope {
public:
    DeviceScope(int dev, bool active) {
#ifdef VARDA_WITH_CUDA
        if (!active) return;
        cudaError_t e = cudaGetDevice(&prev_);
        if (e == cudaSuccess && prev_ != dev) e = cudaSetDevice(dev);
        if (e != cudaSuccess)
            throw std::runtime_error("CUDA device " + std::to_string(dev) + ": " + cudaGetErrorString(e));
        set_ = prev_ != dev;
#else
        (void)dev;
        (void)active;
#endif
    }
    ~DeviceScope() {
#ifdef VARDA_WITH_CUDA
        if (set_) cudaSetDevice(prev_);
#endif
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int prev_ = 0;
    bool set_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, io_config_path, templates, out_dir, realdata, assets, device = "cuda", precision = "fp16";
    std::string td_path, td_io_path;  // temporal downscaler (optional)
    std::string td_device_arg;        // --downscaler-device (empty: the forecaster's device)
    InputFiles files;
    int lead_hours = 0, threads = 0;
    bool graph = true, host_loop = false, dump_y = false, quiet = false, no_downscaler = false;
    size_t write_queue = 0;  // 0: min(steps, 8)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--assets") assets = next();
        else if (a == "--model") model_path = next();
        else if (a == "--io-config") io_config_path = next();
        else if (a == "--regrid-matrix") files.regrid_matrix = next();
        else if (a == "--templates") templates = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--realdata") realdata = next();
        else if (a == "--date") files.date = next();
        else if (a == "--kenda") files.kenda_dir = next();
        else if (a == "--ifs-prev") files.ifs_files_prev.push_back(next());
        else if (a == "--ifs") files.ifs_files_t.push_back(next());
        else if (a == "--ifs-constants") files.ifs_constant_files.push_back(next());
        else if (a == "--lead-time-hours") lead_hours = std::stoi(next());
        else if (a == "--device") device = next();
        else if (a == "--precision") precision = next();
        else if (a == "--no-graph") graph = false;
        else if (a == "--host-loop") host_loop = true;
        else if (a == "--definitions") files.cosmo_definitions = next();
        else if (a == "--allow-nans") files.allow_nans = true;
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--dump-y") dump_y = true;
        else if (a == "--write-queue") write_queue = size_t(std::stoul(next()));
        else if (a == "--quiet") quiet = true;
        else if (a == "--downscaler") td_path = next();
        else if (a == "--downscaler-io-config") td_io_path = next();
        else if (a == "--no-downscaler") no_downscaler = true;
        else if (a == "--downscaler-device") td_device_arg = next();
        else usage();
    }
    if (no_downscaler) td_path = td_io_path = "";
    if (!assets.empty()) {
        // The assets bundle (tools/make_assets.py): fixed names; explicit options win.
        while (assets.size() > 1 && assets.back() == '/') assets.pop_back();
        struct stat sb;
        if (::stat(assets.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
            std::fprintf(stderr, "varda-forecast: --assets %s: not a directory\n", assets.c_str());
            return 1;
        }
        auto fill = [&](std::string& v, const char* name) {
            if (!v.empty()) return;
            v = assets + "/" + name;
            if (::stat(v.c_str(), &sb) != 0) {
                std::fprintf(stderr, "varda-forecast: assets bundle %s has no %s (tools/make_assets.py check)\n",
                             assets.c_str(), name);
                std::exit(1);
            }
        };
        fill(model_path, "model.safetensors");
        fill(io_config_path, "varda-io.json");
        fill(files.regrid_matrix, "regrid-ifs0p25-n320.safetensors");
        fill(templates, "templates");
        fill(files.cosmo_definitions, "definitions");
        if (!no_downscaler) {
            // optional in the bundle: both files or neither
            auto has = [&](const char* name) { return ::stat((assets + "/" + name).c_str(), &sb) == 0; };
            if (td_path.empty() && td_io_path.empty() && has("downscaler.safetensors") != has("varda-td-io.json")) {
                std::fprintf(stderr, "varda-forecast: assets bundle %s has only one of downscaler.safetensors / "
                             "varda-td-io.json\n", assets.c_str());
                return 1;
            }
            const bool td_auto = td_path.empty() && td_io_path.empty() && has("downscaler.safetensors");
            if (td_path.empty() && has("downscaler.safetensors")) td_path = assets + "/downscaler.safetensors";
            if (td_io_path.empty() && has("varda-td-io.json")) td_io_path = assets + "/varda-td-io.json";
            if (td_auto && device == "metal") {
                // the Metal backend runs forecasters only (n_output_steps 1): 6-hourly files only
                std::printf("note: the assets bundle holds a temporal downscaler, which the Metal backend does not "
                            "run; writing the 6-hourly files only (--no-downscaler silences this note)\n");
                td_path = td_io_path = "";
            }
        }
    }
    if (model_path.empty() || io_config_path.empty() || files.regrid_matrix.empty() || templates.empty() ||
        out_dir.empty())
        usage();
    if (td_path.empty() != td_io_path.empty()) {
        std::fprintf(stderr, "varda-forecast: --downscaler and --downscaler-io-config go together\n");
        return 1;
    }
    const bool with_td = !td_path.empty();
    if (threads > 0) omp_set_num_threads(threads);
    int exit_code = 0;  // 3: VARDA_TD_CHECK_INPUTS found a difference
    try {
        const double t_start = app::now_s();
        if (!realdata.empty()) files_from_realdata(realdata, files);
        if (files.date.empty() || files.kenda_dir.empty() || files.ifs_files_prev.empty() ||
            files.ifs_files_t.empty() || files.ifs_constant_files.empty())
            usage();
        if (files.cosmo_definitions.empty()) {
            const char* d = std::getenv("VARDA_COSMO_DEFINITIONS");
            if (!d) throw std::runtime_error("no COSMO definitions: pass --definitions or set VARDA_COSMO_DEFINITIONS");
            files.cosmo_definitions = d;
        }
        // Process-wide, before any handle (spec §1.2); GribOutput's own call is then a no-op.
        set_definitions_path(files.cosmo_definitions);
        ::setenv("VARDA_COSMO_DEFINITIONS", files.cosmo_definitions.c_str(), 1);
        files.io_config = io_config_path;
        files.n320_template = templates + "/ifs-levtype=sfc.grib";
        const IoConfig cfg = IoConfig::load(io_config_path);
        if (lead_hours == 0) lead_hours = cfg.lead_time_hours;

        varda::Options opts;
        opts.device = app::parse_device(device);
        opts.precision = app::parse_precision(precision);
        opts.cuda_graph = graph;
        // The downscaler's options: the forecaster's, on --downscaler-device if given.
        varda::Options td_opts = opts;
        const bool td_cuda = with_td && opts.device == varda::Device::CUDA;
        if (!td_device_arg.empty()) {
            const int d = parse_cuda_device(td_device_arg);
            if (opts.device != varda::Device::CUDA)
                throw std::runtime_error("--downscaler-device " + td_device_arg + " needs --device cuda");
            if (d >= 0) td_opts.cuda_device = d;
#ifdef VARDA_WITH_CUDA
            int n_dev = 0;
            if (cudaGetDeviceCount(&n_dev) != cudaSuccess) {
                (void)cudaGetLastError();
                n_dev = 0;
            }
            if (td_opts.cuda_device >= n_dev)
                throw std::runtime_error("--downscaler-device " + td_device_arg + ": no CUDA device " +
                                         std::to_string(td_opts.cuda_device) + " (" + std::to_string(n_dev) +
                                         " visible" + (n_dev ? ": 0.." + std::to_string(n_dev - 1) : std::string()) +
                                         "; CUDA_VISIBLE_DEVICES / --gres=gpu:N)");
#endif
            if (!with_td) std::printf("note: --downscaler-device without a downscaler: ignored\n");
        }
        const std::string td_device_name =
            td_cuda ? "cuda:" + std::to_string(td_opts.cuda_device) : device;  // forecast.json

        // Start-up in parallel: the input is prepared on its own thread from the converted file's
        // config, grid and masks only, while this thread loads the weights onto the device and sets
        // up the rollout (one eager forward and the CUDA graph capture).
        InputState in;
        double input_s = 0;
        std::exception_ptr input_err;
        std::thread input_thread([&] {
            try {
                const double ti = app::now_s();
                const InputModel im = InputModel::load(model_path);
                in = build_input(im, files);
                input_s = app::now_s() - ti;
            } catch (...) {
                input_err = std::current_exception();
            }
        });
        struct Joiner {
            std::thread& t;
            ~Joiner() {
                if (t.joinable()) t.join();
            }
        } joiner{input_thread};

        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        const varda::Config& mc = model->config();

        std::unique_ptr<varda::Model> td_model;  // loaded after the rollout set-up (below)
        std::unique_ptr<IoConfig> tcfg;          // its io config, checked right after the load
        std::unique_ptr<varda::Downscaler> ds;
#ifdef VARDA_WITH_CUDA
        std::unique_ptr<varda::cuda::DeviceDownscaler> td_device;  // the device-resident loop, if used
#endif
        int W = 0, T_out = 0;                     // window hours, outputs per window
        size_t td_out_n = 0;                      // G * V_out of one hourly state
        double td_load_s = 0;
        if (cfg.timestep_hours != mc.timestep_hours)
            throw std::runtime_error("io config timestep " + std::to_string(cfg.timestep_hours) + " h != model's " +
                                     std::to_string(mc.timestep_hours) + " h");
        if (lead_hours <= 0 || lead_hours % mc.timestep_hours != 0)
            throw std::runtime_error("lead time must be a positive multiple of " +
                                     std::to_string(mc.timestep_hours) + " h");
        const int n_steps = lead_hours / mc.timestep_hours;
        std::printf("model %s (uuid %s) on %s %s, loaded in %.2f s; %d steps (%d h); %d threads\n",
                    model_path.c_str(), mc.uuid.c_str(), device.c_str(), precision.c_str(), load_s, n_steps,
                    lead_hours, omp_get_max_threads());

        t0 = app::now_s();
        std::string engine;
#ifdef VARDA_WITH_CUDA
        std::unique_ptr<varda::cuda::DeviceRollout> device_rollout;
        if (opts.device == varda::Device::CUDA && !host_loop) {
            device_rollout = std::make_unique<varda::cuda::DeviceRollout>(*model, opts.cuda_graph);
            engine = "device";
        }
#endif
        std::unique_ptr<varda::Rollout> host_rollout;
        if (engine.empty()) {
            host_rollout = std::make_unique<varda::Rollout>(*model);
            engine = "host";
        }
        const double setup_s = app::now_s() - t0;
        const char* ov_env = std::getenv("VARDA_TD_OVERLAP");
        const bool td_overlap = with_td && engine == "device" && !(ov_env && std::string(ov_env) == "0");
        // The downscaler loads here, on this thread, after the forecaster's set-up: two backends must not
        // initialise cuBLASLt / capture graphs concurrently on one device (a second thread's allocations
        // invalidate a capture in progress). The input preparation still overlaps it.
        if (with_td) {
            // the downscaler's device for this block (its load, stream), then back to the forecaster's
            DeviceScope td_scope(td_opts.cuda_device, td_cuda);
            const double tt = app::now_s();
            td_model = varda::Model::load(td_path, td_opts);
            td_load_s = app::now_s() - tt;
            // Everything about the downscaler that can be wrong is checked now, before the rollout.
            const varda::Config& tc = td_model->config();
            tcfg = std::make_unique<IoConfig>(IoConfig::load(td_io_path));
            if (tcfg->runner != "temporal_downscaler")
                throw std::runtime_error("downscaler io config " + td_io_path + " is not a temporal_downscaler config");
            if (!tc.is_interpolator()) throw std::runtime_error("--downscaler " + td_path + " is not an interpolator");
            varda::ForecasterInfo fc = varda::ForecasterInfo::from_config(mc, model->latitudes(), model->longitudes());
            ds = std::make_unique<varda::Downscaler>(*td_model, fc);  // coverage, grid, window checks
            W = ds->times().window_hours;
            if (W != mc.timestep_hours)
                throw std::runtime_error("downscaler window " + std::to_string(W) + " h != forecaster step");
            T_out = tc.n_output_steps;
            if (tcfg->timestep_hours * T_out != W)
                throw std::runtime_error("downscaler io config timestep x outputs != the window");
            if (tc.var_output != mc.var_output)
                throw std::runtime_error("the downscaler's output variables differ from the forecaster's");
            // the forecaster's state is passed at the window ends (multiples of W) only
            for (const auto& o : tcfg->outputs)
                for (const auto& pp : o.post_processors)
                    if (pp.type == PostProcessorDef::CopyFromForecaster &&
                        (pp.common_leadtime_hours <= 0 || pp.common_leadtime_hours % W != 0))
                        throw std::runtime_error("downscaler io config: output " + o.name +
                                                 " copies from the forecaster every " +
                                                 std::to_string(pp.common_leadtime_hours) +
                                                 " h, not a positive multiple of the window (" + std::to_string(W) + " h)");
            for (const std::string& f : tcfg->initial_state_fields)
                if (std::find(mc.var_input.begin(), mc.var_input.end(), f) == mc.var_input.end())
                    throw std::runtime_error("downscaler initial-state field " + f + " is not a forecaster input");
            td_out_n = size_t(tc.n_data_nodes) * size_t(tc.n_output_vars);
#ifdef VARDA_WITH_CUDA
            // The device-resident loop (docs/downscaler.md §2): a CUDA downscaler fed by the device rollout
            // (during it; from the host copies when VARDA_TD_OVERLAP=0). Its device buffers and the pinned
            // staging ring are allocated here, before another thread uses the GPU.
            if (engine == "device" && varda::cuda::as_cuda(*td_model) && env_int("VARDA_TD_DEVICE_LOOP", 1) != 0) {
                varda::cuda::DeviceLoopOptions lo;
                lo.ring_states = std::max(1, env_int("VARDA_TD_RING", lo.ring_states));
                lo.staging_bytes = size_t(std::max(1, env_int("VARDA_TD_STAGING_MB", 64))) << 20;
                lo.staging_slots = std::max(1, env_int("VARDA_TD_STAGING_SLOTS", lo.staging_slots));
                lo.copy_threads = std::max(1, env_int("VARDA_TD_COPY_THREADS", lo.copy_threads));
                lo.hold_output = env_int("VARDA_TD_HOLD", -1);
                const int producer = td_overlap ? varda::cuda::as_cuda(*model)->device() : -1;
                td_device = std::make_unique<varda::cuda::DeviceDownscaler>(*ds, producer, lo);
            }
#endif
#ifdef VARDA_WITH_CUDA
            // During the rollout the downscaler's kernels go first: the hourly writer, the critical
            // path, waits for its windows; the 6-hourly files have time to spare.
            const char* pr = std::getenv("VARDA_TD_PRIORITY");
            if (td_overlap && !(pr && std::string(pr) == "0"))
                if (auto* cb = varda::cuda::as_cuda(*td_model)) cb->use_high_priority_stream();
#endif
        }

        input_thread.join();
        if (input_err) std::rethrow_exception(input_err);
        const double ready_s = app::now_s() - t_start;
        if (!quiet)
            for (const auto& l : in.log) std::printf("  %s\n", l.c_str());
        std::printf("input for %s prepared in %.2f s; rollout set up in %.2f s; ready after %.2f s\n",
                    varda::format_iso8601_utc(in.date0).c_str(), input_s, setup_s, ready_s);

        // The configured paths are relative (outputs/<dateTime>_<step>.grib), as anemoi-inference
        // resolves them against its working directory: here against --out.
        mkdirs(out_dir);
        GribOutput out(*model, cfg, templates, out_dir, in.date0);
        const size_t out_n = size_t(mc.n_data_nodes) * size_t(mc.n_output_vars);
        if (write_queue == 0) write_queue = size_t(std::min(n_steps, 8));
        std::vector<float> ys;  // every step's y: --dump-y, and the downscaler's inputs
        if (dump_y || with_td) ys.resize(out_n * size_t(n_steps));
        // While the downscaler runs during the rollout, its hourly writer is the critical path: the
        // 6-hourly writer then encodes with a smaller OpenMP team (8, VARDA_FC_WRITER_THREADS; 0 = the
        // default team), leaving the cores to the hourly one. The files do not depend on it.
        int fc_writer_threads = td_overlap ? 8 : 0;
        if (const char* e = std::getenv("VARDA_FC_WRITER_THREADS"); e && td_overlap) fc_writer_threads = std::atoi(e);
        StepWriter writer(out, out_n, write_queue, fc_writer_threads);
        if (cfg.write_initial_state) writer.push_initial(in.initial_names, in.initial_values);

        // ---- the temporal downscaler (docs/downscaler.md §7): hourly states between the 6-hourly ones.
        // With the device rollout it runs on its own thread during the rollout: window k as soon as step k
        // is out (y_at waits for it), on its own stream of the same GPU, and its hourly writer starts with
        // the rollout. Both models are loaded and the rollout's graphs captured before that thread starts,
        // so no capture or cuBLASLt set-up of one backend runs while the other thread is inside a capture
        // (the rollout captures nothing once set up; the downscaler's first window is eager, its second
        // captures, while this thread only replays graphs and copies). Otherwise (host loop, CPU, Metal, or
        // VARDA_TD_OVERLAP=0) it runs after the rollout, as before. The window outputs go to the hourly
        // writer by pointer (WindowBuffers), not copied.
        int td_buffers = 2;  // window buffers: the writer encodes window k while the GPU computes window k+1
        if (const char* e = std::getenv("VARDA_TD_BUFFERS")) td_buffers = std::max(1, std::atoi(e));
        int td_writer_states = 3, td_writer_threads = 0;
        StepFeed feed;
        std::vector<float> hourly;  // --dump-y: every hourly y (G, V_out), hour 1 .. lead time
        // Declared before the writer that points at them (destroyed after it has stopped).
        std::vector<std::string> init_names;
        std::vector<std::vector<float>> init_values;
        std::vector<float*> win_buf(size_t(n_steps), nullptr);
        // per window: states not yet released by the hourly writer (its encoding workers release them in
        // any order); the buffer goes back to the pool at zero
        std::vector<std::shared_ptr<std::atomic<int>>> win_left(static_cast<size_t>(n_steps));
        std::unique_ptr<WindowBuffers> tbufs;
        std::unique_ptr<GribOutput> tout;
        std::unique_ptr<StepWriter> twriter;
        double td0 = 0, td_setup_s = 0, td_loop_t0 = 0, td_loop_s = 0;
        std::exception_ptr td_err;
        const bool check_inputs = env_int("VARDA_TD_CHECK_INPUTS", 0) != 0;
        int check_windows = 0, check_mismatch = 0;
        auto td_setup = [&] {
            td0 = app::now_s();
            std::printf("downscaler %s (uuid %s) on %s, loaded in %.2f s; %d windows of %d h, %d outputs each%s\n",
                        td_path.c_str(), td_model->config().uuid.c_str(), td_device_name.c_str(), td_load_s, n_steps,
                        W, T_out, td_overlap ? ", during the rollout" : "");
            // Page-locked buffers make the 2.7 GB device-to-host copy of a window a DMA, but locking
            // 2 x 2.7 GB takes ~3 s of kernel time that stalls the other threads' page faults; off
            // by default (VARDA_TD_PINNED=1; docs/downscaler.md §7).
            const char* pin_env = std::getenv("VARDA_TD_PINNED");
            const bool pin = opts.device == varda::Device::CUDA && pin_env && std::string(pin_env) == "1";
            tbufs = std::make_unique<WindowBuffers>(td_out_n * size_t(T_out), std::min(td_buffers, n_steps), pin);
            tout = std::make_unique<GribOutput>(*td_model, *tcfg, templates, out_dir, in.date0);
            // The hourly writer encodes several hours at once (their only link, the tp accumulation, runs
            // in hour order on the writer thread): VARDA_TD_WRITER_STATES states (default 3; 1 = one hour
            // at a time), each with an OpenMP team of VARDA_TD_WRITER_THREADS (default: threads / states).
            if (const char* e = std::getenv("VARDA_TD_WRITER_STATES")) td_writer_states = std::max(1, std::atoi(e));
            if (const char* e = std::getenv("VARDA_TD_WRITER_THREADS")) td_writer_threads = std::max(0, std::atoi(e));
            tout->set_concurrency(td_writer_states, td_writer_threads);
            // every state of the buffers in flight fits in the queue: the buffers are the bound
            twriter = std::make_unique<StepWriter>(*tout, td_out_n, size_t(tbufs->count() * T_out + 1));
            td_setup_s = app::now_s() - td0;
        };
        auto td_loop = [&] {
            try {
                // every CUDA call of this loop (window buffers, predict_step) on the downscaler's device
                DeviceScope td_scope(td_opts.cuda_device, td_cuda);
                td_loop_t0 = app::now_s();
                // Step 0: the analysis (the forecaster's newest input slot), the downscaler's initial-state
                // fields (the prognostics that are also outputs, in its input order), as the runner writes it.
                init_names = tcfg->initial_state_fields;
                init_values.assign(init_names.size(), {});
                const size_t G = size_t(mc.n_data_nodes), Vf = size_t(mc.n_input_vars);
                const float* newest = in.x0.data() + size_t(in.T - 1) * G * Vf;
                for (size_t i = 0; i < init_names.size(); ++i) {
                    auto it = std::find(mc.var_input.begin(), mc.var_input.end(), init_names[i]);  // checked at load
                    const size_t j = size_t(it - mc.var_input.begin());
                    init_values[i].resize(G);
                    float* v = init_values[i].data();
#pragma omp parallel for schedule(static)
                    for (size_t g = 0; g < G; ++g) v[g] = newest[g * Vf + j];
                }
                if (tcfg->write_initial_state) twriter->push_initial(init_names, init_values);
                if (dump_y) hourly.resize(td_out_n * size_t(n_steps) * size_t(T_out));
                tbufs->allocate_all();
                varda::DownscaleRequest rq;
                rq.x0 = in.x0.data();
                rq.forecaster_T = int(in.T);
                rq.date0 = in.date0;
                rq.n_windows = n_steps;
                rq.y_at = [&](int s) -> const float* { return feed.wait(s) ? &ys[size_t(s) * out_n] : nullptr; };
                rq.out_buffer = [&](int k) {
                    win_buf[size_t(k)] = tbufs->acquire();
                    return win_buf[size_t(k)];
                };
                rq.on_output = [&](int k, int t, int step_h, const varda::DateTime& valid, const float* y_t) {
                    if (dump_y)
                        std::memcpy(&hourly[(size_t(k) * size_t(T_out) + size_t(t)) * td_out_n], y_t, td_out_n * 4);
                    // the forecaster's state of this hour, for the copy-prognostic-from-forecaster post-processor
                    // (its host copy: published by the rollout callback, which the device loop does not wait for)
                    const float* fy = nullptr;
                    if (step_h % W == 0) {
                        if (!feed.wait(step_h / W - 1)) throw std::runtime_error("forecaster step not delivered");
                        fy = &ys[size_t(step_h / W - 1) * out_n];
                    }
                    // every state of the window released (no longer read): its buffer is free again
                    if (t == 0) win_left[size_t(k)] = std::make_shared<std::atomic<int>>(T_out);
                    std::function<void()> done = [pool = tbufs.get(), b = win_buf[size_t(k)], left = win_left[size_t(k)]] {
                        if (left->fetch_sub(1) == 1) pool->release(b);
                    };
                    twriter->push_ref(step_h, y_t, fy, std::move(done));
                    if (t + 1 == T_out) {
                        std::printf("hours %d-%d (window %d, to %s) computed\n", k * W + 1, step_h, k + 1,
                                    varda::format_iso8601_utc(valid).c_str());
                        std::fflush(stdout);
                    }
                };
#ifdef VARDA_WITH_CUDA
                if (td_device) {
                    if (check_inputs) {
                        // the device-assembled input against the host assembly from the same states, bitwise
                        rq.on_input = [&](int k, const float* x) {
                            const varda::DownscaleAssembly& a = ds->assembly();
                            const size_t G = size_t(a.G), slot = G * size_t(a.V_in);
                            const float* x0n = in.x0.data() + size_t(in.T - 1) * G * size_t(a.V_fin);
                            std::vector<float> fs(G * size_t(a.F)), fe(G * size_t(a.F)), xh(2 * slot);
                            const auto& lat = td_model->latitudes();
                            const auto& lon = td_model->longitudes();
                            if (a.F > 0) {
                                varda::compute_forcings(ds->times().window_start(in.date0, k), a.forcing_names, lat.data(),
                                                 lon.data(), G, fs.data());
                                varda::compute_forcings(ds->times().window_end(in.date0, k), a.forcing_names, lat.data(),
                                                 lon.data(), G, fe.data());
                            }
                            if (!feed.wait(k)) throw std::runtime_error("forecaster step not delivered");
                            if (k == 0) a.fill_slot(xh.data(), nullptr, x0n, x0n, a.F ? fs.data() : nullptr);
                            else a.fill_slot(xh.data(), &ys[size_t(k - 1) * out_n], nullptr, x0n, a.F ? fs.data() : nullptr);
                            a.fill_slot(xh.data() + slot, &ys[size_t(k) * out_n], nullptr, x0n, a.F ? fe.data() : nullptr);
                            const bool same0 = std::memcmp(xh.data(), x, slot * 4) == 0;
                            const bool same1 = std::memcmp(xh.data() + slot, x + slot, slot * 4) == 0;
                            ++check_windows;
                            if (!same0 || !same1) ++check_mismatch;
                            std::printf("input check window %d: slot 0 %s, slot 1 %s\n", k, same0 ? "bitwise equal" : "DIFFERS",
                                        same1 ? "bitwise equal" : "DIFFERS");
                        };
                    }
                    td_device->run(rq);
                } else
#endif
                    ds->run(rq);
                td_loop_s = app::now_s() - td_loop_t0;
            } catch (...) {
                td_err = std::current_exception();
            }
        };
        if (td_overlap) td_setup();
        std::thread td_thread;
        std::function<void()> td_cancel_device = [] {};
#ifdef VARDA_WITH_CUDA
        if (td_device) td_cancel_device = [&] { td_device->cancel(); };
        // The rollout hands each step's device output to the device loop: a copy into the loop's ring,
        // which step s + 2 (the next writer of that output buffer) waits for on the GPU.
        if (td_device && td_overlap && device_rollout)
            device_rollout->set_step_hook([&](int s, const float* d_y, cudaEvent_t ready) {
                return td_device->push_device(s, d_y, ready);
            });
#endif
        struct TdGuard {  // on an exception: unblock the downscaler thread and wait for it
            std::thread& t;
            StepFeed& f;
            std::unique_ptr<WindowBuffers>& b;
            std::function<void()>& c;
            ~TdGuard() {
                if (!t.joinable()) return;
                f.cancel();
                if (b) b->cancel();
                c();
                t.join();
            }
        } td_guard{td_thread, feed, tbufs, td_cancel_device};

        double ys_copy_s = 0;  // the rollout callback's copy of y into ys
        // that copy (452 MB a step) with a few OpenMP threads (VARDA_FC_COPY_THREADS, default 8)
        const int ys_copy_threads = std::max(1, env_int("VARDA_FC_COPY_THREADS", 8));
        varda::RolloutRequest q;
        q.x0 = in.x0.data();
        q.date0 = in.date0;
        q.n_steps = n_steps;
        q.on_step = [&](int s, const varda::DateTime& valid, const float* y) {
            if (!ys.empty()) {
                float* ys_s = &ys[size_t(s) * out_n];
                const double tc0 = app::now_s();
                parallel_copy(ys_s, y, out_n, ys_copy_threads);
                ys_copy_s += app::now_s() - tc0;
                feed.publish(s);
                writer.push_ref((s + 1) * mc.timestep_hours, ys_s, nullptr);  // ys outlives the writer
            } else {
                writer.push((s + 1) * mc.timestep_hours, y);
            }
            std::printf("step %d (%s) computed\n", s + 1, varda::format_iso8601_utc(valid).c_str());
            std::fflush(stdout);
        };
        if (td_overlap) td_thread = std::thread(td_loop);
        t0 = app::now_s();
        std::vector<float> step_ms;
#ifdef VARDA_WITH_CUDA
        if (device_rollout) {
            device_rollout->run(q);
            step_ms = device_rollout->step_ms();
        }
#endif
        if (host_rollout) {
            host_rollout->run(q);
            for (const auto& t : host_rollout->times())
                step_ms.push_back(float(1e3 * (t.predict + t.forcings + t.update)));
        }
        const double rollout_s = app::now_s() - t0;
        writer.finish();
        out.close();
        const double write_tail_s = app::now_s() - t0 - rollout_s;
        const GribOutput::Stats gs = out.stats();

        json td_rep;
        if (with_td) {
            if (td_overlap) {
                td_thread.join();
            } else {
                td_setup();
                feed.publish_all(n_steps);
                td_loop();
            }
            if (td_err) std::rethrow_exception(td_err);
            const double td_loop_end = td_loop_t0 + td_loop_s;
            twriter->finish();
            tout->close();
            const double td_end = app::now_s();
            const double td_tail_s = td_end - td_loop_end;
            const varda::Config& tc = td_model->config();
            const GribOutput::Stats ts = tout->stats();
            std::vector<double> predict_ms, assemble_ms, forcing_ms, wait_in_ms, wait_out_ms, d2h_ms, host_copy_ms, window_ms;
            double wait_in = 0, wait_out = 0;
            const std::vector<varda::Downscaler::WindowTimes>* wts = &ds->window_times();
            json dl_rep = nullptr;
#ifdef VARDA_WITH_CUDA
            if (td_device) {
                wts = &td_device->window_times();
                const auto& lo = td_device->options();
                dl_rep = {{"device", td_device->device()},
                          {"feed", td_device->feed_path()},
                          {"ring_states", lo.ring_states},
                          {"staging_bytes", lo.staging_bytes},
                          {"staging_slots", lo.staging_slots},
                          {"copy_threads", lo.copy_threads},
                          {"output_held", td_device->output_held()},
                          {"device_bytes", td_device->device_bytes()},
                          {"pinned_bytes", td_device->pinned_bytes()},
                          {"free_bytes_at_setup", td_device->free_bytes_at_setup()},
                          {"setup_seconds", td_device->setup_seconds()}};
                if (check_inputs) dl_rep["input_check"] = {{"windows", check_windows}, {"mismatched", check_mismatch}};
            }
#endif
            for (const auto& w : *wts) {
                predict_ms.push_back(1e3 * w.predict);
                assemble_ms.push_back(1e3 * w.assemble);
                forcing_ms.push_back(1e3 * w.forcings);
                wait_in_ms.push_back(1e3 * w.wait_input);
                wait_out_ms.push_back(1e3 * w.wait_output);
                d2h_ms.push_back(1e3 * w.d2h);
                host_copy_ms.push_back(1e3 * w.host_copy);
                window_ms.push_back(1e3 * w.window);
                wait_in += w.wait_input;
                wait_out += w.wait_output;
            }
            const double blocked = wait_out + twriter->blocked_seconds();
            td_rep = {{"model", td_path},
                      {"io_config", td_io_path},
                      {"device", td_device_name},
                      {"checkpoint_uuid", tc.uuid},
                      {"n_windows", n_steps},
                      {"window_hours", W},
                      {"outputs_per_window", T_out},
                      {"during_rollout", td_overlap},
                      {"loop", dl_rep.is_null() ? "host" : "device"},
                      {"device_loop", dl_rep},
                      {"window_buffers", tbufs->count()},
                      {"window_buffers_pinned", tbufs->pinned()},
                      {"writer_states", tout->concurrency()},
                      {"writer_threads", td_writer_threads > 0 ? td_writer_threads
                                                               : std::max(1, omp_get_max_threads() / tout->concurrency())},
                      {"files", tout->files()},
                      {"seconds",
                       {{"load", td_load_s},
                        {"setup", td_setup_s},
                        {"start", td_loop_t0 - t_start},
                        {"end", td_end - t_start},
                        {"buffer_alloc", tbufs->alloc_seconds()},
                        {"loop", td_loop_s},
                        {"loop_waiting_for_forecaster", wait_in},
                        {"loop_blocked_on_writer", blocked},
                        {"loop_callback_copy", twriter->copy_seconds()},
                        {"initial_state_write", twriter->initial_seconds()},
                        {"write_after_loop", td_tail_s},
                        {"step_write", twriter->step_seconds()},
                        {"grib", {{"post", ts.post}, {"encode", ts.encode}, {"write", ts.write},
                                  {"write_wait", ts.write_wait}, {"cpu_gather", ts.cpu_gather},
                                  {"cpu_keys", ts.cpu_keys}, {"cpu_pack", ts.cpu_pack}, {"cpu_check", ts.cpu_check},
                                  {"states", tout->concurrency()}, {"max_states_in_flight", ts.max_states_in_flight},
                                  {"messages", ts.messages}, {"bytes", ts.bytes}}},
                        {"total", td_end - td0}}},
                      {"window_predict_ms", predict_ms},
                      {"window_assemble_ms", assemble_ms},
                      {"window_forcings_ms", forcing_ms},
                      {"window_wait_forecaster_ms", wait_in_ms},
                      {"window_wait_buffer_ms", wait_out_ms},
                      {"window_d2h_ms", d2h_ms},
                      {"window_host_copy_ms", host_copy_ms},
                      {"window_ms", window_ms}};
            if (check_inputs) {
                std::printf("input check: %d of %d device-assembled window inputs differ from the host assembly\n",
                            check_mismatch, check_windows);
                if (check_mismatch > 0) exit_code = 3;
            }
            std::printf("downscaler done in %.2f s (%.2f .. %.2f s): loop %.2f (%.2f s waiting for the forecaster, %.2f s "
                        "blocked on the writer), +%.2f s writing after the last window; GRIB encode %.2f s, write %.2f s; "
                        "%zu hourly files\n",
                        td_end - td0, td0 - t_start, td_end - t_start, td_loop_s, wait_in, blocked, td_tail_s, ts.encode,
                        ts.write, tout->files().size());
        }
        const double total_s = app::now_s() - t_start;

        if (dump_y) {
            std::vector<varda::TensorOut> t;
            std::vector<std::string> names(n_steps);
            for (int s = 0; s < n_steps; ++s)
                t.push_back({"y." + std::to_string(s + 1), varda::DType::F32,
                             {mc.n_data_nodes, mc.n_output_vars}, &ys[size_t(s) * out_n]});
            // x0 (T, G, V_in) and the input names: what the downscaler needs (varda-downscale --forecast).
            t.push_back({"x0", varda::DType::F32, {in.T, in.G, in.V_in}, in.x0.data()});
            varda::write_safetensors(out_dir + "/y.safetensors", t,
                                     {{"varda_forecast_y", "1"},
                                      {"meta", json({{"out_vars", mc.var_output},
                                                     {"in_vars", mc.var_input},
                                                     {"n_input_steps", mc.n_input_steps},
                                                     {"timestep_hours", mc.timestep_hours},
                                                     {"date", varda::format_iso8601_utc(in.date0)}}).dump()}});
            if (!hourly.empty()) {
                // the downscaler's raw hourly outputs (before any post-processor): `y.<hour>` (G, V_out)
                const varda::Config& tc = td_model->config();
                const size_t td_out_n = size_t(tc.n_data_nodes) * size_t(tc.n_output_vars);
                const int T_out = tc.n_output_steps, W = mc.timestep_hours;
                std::vector<varda::TensorOut> h;
                for (int k = 0; k < n_steps; ++k)
                    for (int t = 0; t < T_out; ++t)
                        h.push_back({"y." + std::to_string(k * W + (tc.output_times[t] - tc.input_times[0]) * tc.frequency_hours),
                                     varda::DType::F32, {tc.n_data_nodes, tc.n_output_vars},
                                     &hourly[(size_t(k) * size_t(T_out) + size_t(t)) * td_out_n]});
                varda::write_safetensors(out_dir + "/hourly.safetensors", h,
                                         {{"varda_downscale_y", "1"},
                                          {"meta", json({{"out_vars", tc.var_output},
                                                         {"date", varda::format_iso8601_utc(in.date0)},
                                                         {"checkpoint_uuid", tc.uuid}}).dump()}});
            }
        }
        json rep = {{"model", model_path},
                    {"assets", assets},
                    {"io_config", io_config_path},
                    {"regrid_matrix", files.regrid_matrix},
                    {"templates", templates},
                    {"definitions", files.cosmo_definitions},
                    {"realdata", realdata},
                    {"checkpoint_uuid", mc.uuid},
                    {"date", varda::format_iso8601_utc(in.date0)},
                    {"device", device},
                    {"precision", precision},
                    {"cuda_graph", graph},
                    {"engine", engine},
                    {"threads", omp_get_max_threads()},
                    {"n_steps", n_steps},
                    {"files", out.files()},
                    {"input_log", in.log},
                    {"write_queue", write_queue},
                    {"seconds",
                     {{"load", load_s},
                      {"input", input_s},
                      {"input_stages", in.seconds},
                      {"rollout_setup", setup_s},
                      {"ready", ready_s},
                      {"initial_state_write", writer.initial_seconds()},
                      {"rollout", rollout_s},
                      {"rollout_blocked_on_writer", writer.blocked_seconds()},
                      {"rollout_callback_copy", writer.copy_seconds() + ys_copy_s},
                      {"write_after_rollout", write_tail_s},
                      {"step_write", writer.step_seconds()},
                      {"grib", {{"post", gs.post}, {"encode", gs.encode}, {"write", gs.write},
                                {"messages", gs.messages}, {"bytes", gs.bytes}}},
                      {"total", total_s}}},
                    {"step_ms", step_ms}};
        if (with_td) rep["downscaler"] = td_rep;
        std::ofstream(out_dir + "/forecast.json") << rep.dump(1) << "\n";
        std::printf("forecast done in %.2f s: ready after %.2f (load %.2f + set-up %.2f | input %.2f), rollout "
                    "%.2f (callback: %.2f s copying y, %.2f s blocked on the writer), +%.2f s writing after the last step; GRIB encode %.2f s, "
                    "write %.2f s; %zu files under %s\n",
                    total_s, ready_s, load_s, setup_s, input_s, rollout_s, writer.copy_seconds() + ys_copy_s, writer.blocked_seconds(), write_tail_s,
                    gs.encode, gs.write, out.files().size(), out_dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "varda-forecast: %s\n", e.what());
        return 1;
    }
    return exit_code;
}
