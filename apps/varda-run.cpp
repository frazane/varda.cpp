// Runs predict_step on an input read from a safetensors file and reports timings.
//
//   varda-run --model M.safetensors --input X.safetensors [--key in.x] [--steps N]
//             [--device cpu|cuda|metal] [--precision fp32|fp16] [--threads N] [--no-cuda-graph]
//             [--out Y.safetensors]
//
// The input tensor (default key `in.x`, e.g. from a golden file) must have the contract shape
// (1, T_in, G, V_in). predict_step runs N times (default 1) on the same input, without the
// activation hook; each call is timed, and every output must be bitwise equal to the first.
// Prints load time, per-step wall time, and peak RSS (VmHWM: includes the resident pages of
// the mapped model and input files). --out writes the first output as `out.y`.
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <cstdio>
#include <fstream>

#include "common.h"
#include "varda/format.h"
#include "varda/varda.h"

namespace {

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-run --model M --input X [--key in.x] [--steps N] [--device cpu|cuda|metal]\n"
                 "                 [--precision fp32|fp16] [--threads N] [--no-cuda-graph] [--out Y]\n");
    std::exit(2);
}

// KiB value of a memory statistic, or -1 when unavailable.
long proc_status_kb(const char* field) {
#ifdef __APPLE__
    if (std::string(field) == "VmHWM") {
        struct rusage ru {};
        return getrusage(RUSAGE_SELF, &ru) == 0 ? ru.ru_maxrss / 1024 : -1;
    }
    if (std::string(field) == "VmRSS") {
        mach_task_basic_info_data_t info {};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
            return static_cast<long>(info.resident_size / 1024);
    }
    return -1;
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::string want = std::string(field) + ":";
    while (std::getline(f, line))
        if (line.compare(0, want.size(), want) == 0) return std::atol(line.c_str() + want.size());
    return -1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, input_path, key = "in.x", out_path;
    int steps = 1;
    varda::Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--key") key = next();
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--device") opts.device = app::parse_device(next());
        else if (a == "--precision") opts.precision = app::parse_precision(next());
        else if (a == "--threads") opts.num_threads = std::stoi(next());
        else if (a == "--no-cuda-graph") opts.cuda_graph = false;
        else if (a == "--out") out_path = next();
        else usage();
    }
    if (model_path.empty() || input_path.empty() || steps < 1) usage();

    try {
        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        const varda::Config& cfg = model->config();
        std::printf("model %s (uuid %s), %s %s, threads %d, load %.2f s, RSS after load %.2f GB\n",
                    model_path.c_str(), cfg.uuid.c_str(), varda::to_string(opts.device).c_str(),
                    varda::to_string(opts.precision).c_str(), omp_get_max_threads(), load_s,
                    proc_status_kb("VmRSS") / 1048576.0);

        varda::SafeTensorsFile in(input_path);
        const varda::TensorView& x = in.peek(key);
        const std::vector<int64_t> x_shape = {1, cfg.n_input_steps, cfg.n_data_nodes, cfg.n_input_vars};
        if (x.dtype != varda::DType::F32 || x.shape != x_shape)
            throw std::runtime_error(input_path + ": " + key + " is " + x.shape_str() + ", expected " +
                                     app::shape_str(x_shape) + " F32");

        const std::vector<int64_t> y_shape = {1, cfg.n_output_steps, 1, cfg.n_data_nodes, cfg.n_output_vars};
        const int64_t y_numel = cfg.n_output_steps * cfg.n_data_nodes * cfg.n_output_vars;
        std::vector<float> y0(y_numel), y(y_numel);
        bool repeat_bitwise = true;
        std::vector<double> times;
        for (int s = 0; s < steps; ++s) {
            float* dst = s == 0 ? y0.data() : y.data();
            t0 = app::now_s();
            model->predict_step(x.as<float>(), dst, nullptr);
            times.push_back(app::now_s() - t0);
            if (s > 0 && std::memcmp(y0.data(), y.data(), sizeof(float) * y_numel) != 0) repeat_bitwise = false;
            std::printf("step %d: %.3f s\n", s, times.back());
            std::fflush(stdout);
        }
        double best = times[0], sum = 0;
        for (double t : times) best = std::min(best, t), sum += t;
        std::printf("predict_step: first %.3f s, best %.3f s, mean %.3f s over %d (threads %d)%s\n", times[0], best,
                    sum / steps, steps, omp_get_max_threads(),
                    steps > 1 ? (repeat_bitwise ? ", repeats bitwise equal" : ", REPEATS DIFFER") : "");
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
        std::printf("peak RSS %.2f GB (ru_maxrss), RSS now %.2f GB\n",
                    ru.ru_maxrss / 1073741824.0, proc_status_kb("VmRSS") / 1048576.0);
#else
        std::printf("peak RSS %.2f GB (VmHWM; ru_maxrss %.2f GB), RSS now %.2f GB (anon %.2f, file %.2f)\n",
                    proc_status_kb("VmHWM") / 1048576.0, ru.ru_maxrss / 1048576.0, proc_status_kb("VmRSS") / 1048576.0,
                    proc_status_kb("RssAnon") / 1048576.0, proc_status_kb("RssFile") / 1048576.0);
#endif

        if (!out_path.empty()) {
            varda::write_safetensors(out_path, {{"out.y", varda::DType::F32, y_shape, y0.data()}},
                                     {{"model", model_path}, {"uuid", cfg.uuid}, {"input", input_path},
                                      {"device", varda::to_string(opts.device)},
                                      {"precision", varda::to_string(opts.precision)},
                                      {"threads", std::to_string(omp_get_max_threads())}});
            std::printf("wrote %s\n", out_path.c_str());
        }
        return repeat_bitwise ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
