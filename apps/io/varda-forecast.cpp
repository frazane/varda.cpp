// varda-forecast: a forecast from real data, GRIB in and GRIB out, as anemoi-inference runs the
// Hugging Face config (docs/spec-io.md): KENDA-CH1 + IFS open data -> x0 (varda::io::build_input),
// the autoregressive rollout (docs/rollout-engine.md), then the lam (ICON templates) and global (IFS
// templates) GRIB files per step, including the initial state (varda::io::GribOutput).
//
//   varda-forecast --model M.safetensors --io-config varda-io.json --regrid-matrix R.safetensors
//                  --templates HF_TEMPLATES_DIR --out DIR
//                  ( --realdata DIR | --date T --kenda DIR --ifs-prev F.. --ifs F.. --ifs-constants F.. )
//                  [--lead-time-hours H] [--device cuda|cpu] [--precision fp16|fp32] [--no-graph]
//                  [--host-loop] [--definitions D] [--allow-nans] [--threads N] [--dump-y]
//                  [--write-queue N]
//
// DIR receives the GRIB files under the configured relative paths (outputs/<dateTime>_<step>.grib); DIR/forecast.json the provenance and
// timings. --dump-y also writes DIR/y.safetensors (`y.<step>` F32 [G, V_out], the model outputs
// before post-processing) for comparisons against the reference dumps.
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
#include <deque>
#include <memory>
#include <exception>
#include <fstream>
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
#include "varda/rollout.h"
#include "varda/varda.h"
#ifdef VARDA_WITH_CUDA
#include "varda/cuda/rollout_cuda.h"
#endif

namespace {

using json = nlohmann::json;
using namespace varda::io;

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-forecast --model M --io-config J --regrid-matrix R --templates DIR --out DIR\n"
                 "         (--realdata DIR | --date T --kenda DIR --ifs-prev F.. --ifs F.. --ifs-constants F..)\n"
                 "         [--lead-time-hours H] [--device cuda|cpu] [--precision fp16|fp32] [--no-graph]\n"
                 "         [--host-loop] [--definitions D] [--allow-nans] [--threads N] [--dump-y]\n");
    std::exit(2);
}

void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i <= p.size(); ++i) {
        if ((i == p.size() || p[i] == '/') && !cur.empty()) ::mkdir(cur.c_str(), 0755);
        if (i < p.size()) cur += p[i];
    }
}

// Encodes the initial state and the steps in order on its own thread, so GRIB encoding overlaps the
// rollout; the first error is rethrown by finish(). At most `depth` steps wait in the queue (each holds
// a copy of y, G * V_out floats): when the GPU is faster than the encoder, the rollout only blocks
// once the queue is full.
class StepWriter {
public:
    StepWriter(GribOutput& out, size_t n, size_t depth)
        : out_(out), n_(n), depth_(depth), th_([this] { loop(); }) {}
    ~StepWriter() {
        if (th_.joinable()) {
            stop();
            th_.join();
        }
    }
    // `in` must stay alive and unchanged until finish().
    void push_initial(const InputState& in) { enqueue({0, true, &in, {}}); }
    void push(int step_hours, const float* y) {
        const double t0 = app::now_s();
        Item it{step_hours, false, nullptr, std::vector<float>(y, y + n_)};
        copy_s_ += app::now_s() - t0;
        enqueue(std::move(it));
    }
    void finish() {
        stop();
        th_.join();
        if (err_) std::rethrow_exception(err_);
    }
    double initial_seconds() const { return initial_s_; }
    const std::vector<double>& step_seconds() const { return secs_; }
    double blocked_seconds() const { return blocked_s_; }  // time the rollout waited for a free slot
    double copy_seconds() const { return copy_s_; }        // time the rollout callback spent copying y

private:
    struct Item {
        int step_hours = 0;
        bool initial = false;
        const InputState* in = nullptr;
        std::vector<float> y;
    };
    void enqueue(Item it) {
        const double t0 = app::now_s();
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return q_.size() < depth_ || err_; });
        blocked_s_ += app::now_s() - t0;
        if (err_) return;
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
            if (err_) continue;  // drain after an error
            try {
                const double t0 = app::now_s();
                if (it.initial) {
                    out_.write_initial(it.in->initial_names, it.in->initial_values);
                    initial_s_ = app::now_s() - t0;
                } else {
                    out_.write_step(it.step_hours, it.y.data());
                    secs_.push_back(app::now_s() - t0);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu_);
                err_ = std::current_exception();
                cv_.notify_all();
            }
        }
    }
    GribOutput& out_;
    size_t n_, depth_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> q_;
    bool done_ = false;
    std::exception_ptr err_;
    double initial_s_ = 0, blocked_s_ = 0, copy_s_ = 0;
    std::vector<double> secs_;
    std::thread th_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, io_config_path, templates, out_dir, realdata, device = "cuda", precision = "fp16";
    InputFiles files;
    int lead_hours = 0, threads = 0;
    bool graph = true, host_loop = false, dump_y = false;
    size_t write_queue = 0;  // 0: min(steps, 8)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--model") model_path = next();
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
        else usage();
    }
    if (model_path.empty() || io_config_path.empty() || files.regrid_matrix.empty() || templates.empty() ||
        out_dir.empty())
        usage();
    if (threads > 0) omp_set_num_threads(threads);
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

        input_thread.join();
        if (input_err) std::rethrow_exception(input_err);
        const double ready_s = app::now_s() - t_start;
        for (const auto& l : in.log) std::printf("  %s\n", l.c_str());
        std::printf("input for %s prepared in %.2f s; rollout set up in %.2f s; ready after %.2f s\n",
                    varda::format_iso8601_utc(in.date0).c_str(), input_s, setup_s, ready_s);

        // The configured paths are relative (outputs/<dateTime>_<step>.grib), as anemoi-inference
        // resolves them against its working directory: here against --out.
        mkdirs(out_dir);
        GribOutput out(*model, cfg, templates, out_dir, in.date0);
        const size_t out_n = size_t(mc.n_data_nodes) * size_t(mc.n_output_vars);
        if (write_queue == 0) write_queue = size_t(std::min(n_steps, 8));
        StepWriter writer(out, out_n, write_queue);
        if (cfg.write_initial_state) writer.push_initial(in);

        std::vector<float> ys;
        if (dump_y) ys.resize(out_n * size_t(n_steps));
        varda::RolloutRequest q;
        q.x0 = in.x0.data();
        q.date0 = in.date0;
        q.n_steps = n_steps;
        q.on_step = [&](int s, const varda::DateTime& valid, const float* y) {
            if (dump_y) std::memcpy(&ys[size_t(s) * out_n], y, out_n * 4);
            writer.push((s + 1) * mc.timestep_hours, y);
            std::printf("step %d (%s) computed\n", s + 1, varda::format_iso8601_utc(valid).c_str());
            std::fflush(stdout);
        };
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
        const double total_s = app::now_s() - t_start;

        if (dump_y) {
            std::vector<varda::TensorOut> t;
            std::vector<std::string> names(n_steps);
            for (int s = 0; s < n_steps; ++s)
                t.push_back({"y." + std::to_string(s + 1), varda::DType::F32,
                             {mc.n_data_nodes, mc.n_output_vars}, &ys[size_t(s) * out_n]});
            varda::write_safetensors(out_dir + "/y.safetensors", t,
                                     {{"varda_forecast_y", "1"},
                                      {"meta", json({{"out_vars", mc.var_output},
                                                     {"date", varda::format_iso8601_utc(in.date0)}}).dump()}});
        }
        json rep = {{"model", model_path},
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
                      {"rollout_callback_copy", writer.copy_seconds()},
                      {"write_after_rollout", write_tail_s},
                      {"step_write", writer.step_seconds()},
                      {"grib", {{"post", gs.post}, {"encode", gs.encode}, {"write", gs.write},
                                {"messages", gs.messages}, {"bytes", gs.bytes}}},
                      {"total", total_s}}},
                    {"step_ms", step_ms}};
        std::ofstream(out_dir + "/forecast.json") << rep.dump(1) << "\n";
        std::printf("forecast done in %.2f s: ready after %.2f (load %.2f + set-up %.2f | input %.2f), rollout "
                    "%.2f (callback: %.2f s copying y, %.2f s blocked on the writer), +%.2f s writing after the last step; GRIB encode %.2f s, "
                    "write %.2f s; %zu files under %s\n",
                    total_s, ready_s, load_s, setup_s, input_s, rollout_s, writer.copy_seconds(), writer.blocked_seconds(), write_tail_s,
                    gs.encode, gs.write, out.files().size(), out_dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "varda-forecast: %s\n", e.what());
        return 1;
    }
    return 0;
}
