// Benchmark of the CUDA backend (docs/cuda.md §8).
//
//   varda-bench --model M.safetensors --input GOLDEN.safetensors [--device cuda] [--precision fp32|fp16]
//               [--no-cuda-graph] [--warmup N] [--iters N] [--cuda-device I] [--json OUT.json]
//               [--host-iters N] [--determinism]
//
// The input `in.x` is read from a golden file (copied to host memory first). One configuration
// (precision, CUDA graph on/off) per run. Measures, in one process:
//   load_s        Model::load wall time (weights, graphs, buffers)
//   first_call_s  the first Model::predict_step (host in/out; eager, includes lazy kernel loading)
//   times_ms      `iters` warm forwards after `warmup` more, hook-free, timed with CUDA events on
//                 device-resident input and output: predict_step_device (normalise, forward,
//                 residual, bounding, denormalise); with the graph on, as replays of a CUDA graph of
//                 it captured by this app, otherwise eager launches. median_ms / min_ms of these.
//   host_ms       warm Model::predict_step (host in/out, H2D + forward + D2H), wall time; the
//                 backend's own CUDA graph follows the --no-cuda-graph setting
//   peak_mem_gb   peak device memory used (cudaMemGetInfo total - free, sampled after load, after
//                 each phase and at the end; includes the CUDA context and this app's buffers)
// Checks, bitwise: device output == host predict_step output == first-call output, graph replay ==
// eager launch, predict_step_device == denormalise(forward_normalised(normalise(x))) on caller
// buffers. --determinism also hashes every contract activation of hooked runs (two runs of one
// model instance, one of a second instance) and reports keys whose bytes differ.
// Exit status 1 if a check fails.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "varda/varda.h"

#ifdef VARDA_WITH_CUDA
#include <cuda_runtime_api.h>

#include "varda/cuda/cuda_backend.h"
#endif

using app::json;

namespace {
[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: varda-bench --model M --input GOLDEN [--device cuda] [--precision fp32|fp16] [--no-cuda-graph]\n"
                 "                   [--warmup N] [--iters N] [--host-iters N] [--cuda-device I] [--json OUT.json]\n"
                 "                   [--determinism]\n");
    std::exit(2);
}
}  // namespace

#ifndef VARDA_WITH_CUDA
int main() {
    std::fprintf(stderr, "varda-bench: built without CUDA\n");
    return 1;
}
#else

#define CK(expr)                                                                                    \
    do {                                                                                            \
        cudaError_t e_ = (expr);                                                                    \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e_)); \
    } while (0)

namespace {
double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
double vmin(const std::vector<double>& v) { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
double vmean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // logs survive a crash
    std::string model_path, input_path, json_path;
    varda::Options opts;
    opts.device = varda::Device::CUDA;
    int warmup = 3, iters = 10, host_iters = 3;
    bool determinism = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--device") { if (next() != "cuda") { std::fprintf(stderr, "varda-bench: CUDA only\n"); return 2; } }
        else if (a == "--precision") opts.precision = next() == "fp16" ? varda::Precision::FP16 : varda::Precision::FP32;
        else if (a == "--no-cuda-graph") opts.cuda_graph = false;
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--iters" || a == "--warm") iters = std::stoi(next());
        else if (a == "--host-iters") host_iters = std::stoi(next());
        else if (a == "--cuda-device") opts.cuda_device = std::stoi(next());
        else if (a == "--json") json_path = next();
        else if (a == "--determinism") determinism = true;
        else usage();
    }
    if (model_path.empty() || input_path.empty() || iters < 1) usage();

    json rep;
    int status = 0;
    try {
        app::Golden golden(input_path);
        const varda::TensorView& xv = golden.file->peek("in.x");
        CK(cudaSetDevice(opts.cuda_device));
        CK(cudaFree(nullptr));  // create the context
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, opts.cuda_device));
        size_t free_b, total_b;
        double peak_gb = 0;
        auto sample_mem = [&] {
            CK(cudaMemGetInfo(&free_b, &total_b));
            peak_gb = std::max(peak_gb, (total_b - free_b) / 1e9);
        };
        sample_mem();
        const double context_gb = peak_gb;

        double t0 = app::now_s();
        auto model = varda::Model::load(model_path, opts);
        const double load_s = app::now_s() - t0;
        sample_mem();
        const varda::Config cfg = model->config();  // a copy: --determinism reloads the model
        if (golden.uuid != cfg.uuid) throw std::runtime_error("golden checkpoint uuid does not match the model");
        const int64_t in_n = int64_t(cfg.n_input_steps) * cfg.n_data_nodes * cfg.n_input_vars;
        const int64_t out_n = cfg.n_data_nodes * cfg.n_output_vars;
        if (xv.numel() != in_n) throw std::runtime_error("in.x has the wrong size");
        auto* be = varda::cuda::as_cuda(*model);
        const char* graph_s = opts.cuda_graph ? "on" : "off";
        std::printf("model %s\nGPU %s, %s, CUDA graph %s, load %.2f s, decoder chunk %lld rows\n", model_path.c_str(),
                    prop.name, varda::to_string(opts.precision).c_str(), graph_s, load_s,
                    (long long)be->decoder_chunk_rows());

        // host copy of the input: the golden file is mmapped (its first touch would be timed otherwise)
        std::vector<float> x_host(xv.as<float>(), xv.as<float>() + in_n);
        const float* x = x_host.data();
        std::vector<float> y_first(out_n), y_host(out_n), y_dev(out_n), y_eager(out_n), y_split(out_n);

        t0 = app::now_s();
        model->predict_step(x, y_first.data());  // eager (the backend captures its graph on the next call)
        const double first_s = app::now_s() - t0;
        sample_mem();

        std::vector<double> host_ms;
        for (int i = 0; i < host_iters + 1; ++i) {  // +1: the backend's graph capture (graph on)
            t0 = app::now_s();
            model->predict_step(x, y_host.data());
            if (i > 0) host_ms.push_back(1e3 * (app::now_s() - t0));
        }
        sample_mem();

        // ---- device-level timing on caller buffers
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        float *dx, *dxn, *dy, *dyn, *dy2;
        CK(cudaMalloc((void**)&dx, in_n * 4));
        CK(cudaMalloc((void**)&dxn, in_n * 4));
        CK(cudaMalloc((void**)&dy, out_n * 4));
        CK(cudaMalloc((void**)&dyn, out_n * 4));
        CK(cudaMalloc((void**)&dy2, out_n * 4));
        CK(cudaMemcpy(dx, x, in_n * 4, cudaMemcpyHostToDevice));
        be->prepare();
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        if (opts.cuda_graph) {
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            be->predict_step_device(dx, dy, s);
            CK(cudaStreamEndCapture(s, &graph));
            CK(cudaGraphInstantiate(&exec, graph, 0));
        }
        auto forward = [&] {
            if (exec) CK(cudaGraphLaunch(exec, s));
            else be->predict_step_device(dx, dy, s);
        };
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int i = 0; i < warmup; ++i) forward();
        CK(cudaStreamSynchronize(s));
        std::vector<double> times;
        for (int i = 0; i < iters; ++i) {
            CK(cudaEventRecord(e0, s));
            forward();
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            times.push_back(ms);
        }
        sample_mem();
        CK(cudaMemcpy(y_dev.data(), dy, out_n * 4, cudaMemcpyDeviceToHost));
        // eager launch on other output buffers (== graph replay when the graph is on)
        be->predict_step_device(dx, dy2, s);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(y_eager.data(), dy2, out_n * 4, cudaMemcpyDeviceToHost));
        // the rollout entry points, on caller buffers
        be->normalise(dx, dxn, s);
        be->forward_normalised(dxn, dyn, s);
        be->denormalise(dyn, dy2, s);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(y_split.data(), dy2, out_n * 4, cudaMemcpyDeviceToHost));
        sample_mem();

        auto same = [&](const std::vector<float>& a, const std::vector<float>& b) {
            return std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        };
        const bool b_eager = same(y_eager, y_dev), b_split = same(y_split, y_dev), b_host = same(y_host, y_dev),
                   b_first = same(y_first, y_dev);
        if (!(b_eager && b_split && b_host && b_first)) status = 1;
        const size_t w_bytes = be->weight_bytes(), a_bytes = be->activation_bytes();
        const int64_t chunk_rows = be->decoder_chunk_rows();

        std::printf("\nwarm forward, hook-free, device-resident in/out (%s), %d iters after %d warm-up:\n",
                    opts.cuda_graph ? "CUDA graph replay" : "eager launches", iters, warmup);
        std::printf("  median %.2f ms, mean %.2f ms, min %.2f ms\n", median(times), vmean(times), vmin(times));
        std::printf("host predict_step (H2D + forward + D2H, backend graph %s): median %.2f ms\n", graph_s,
                    median(host_ms));
        std::printf("first call %.3f s (eager), load %.2f s\n", first_s, load_s);
        std::printf("bitwise: %s==eager %s, forward_normalised split==fused %s, host==device %s, first call==device %s\n",
                    opts.cuda_graph ? "graph" : "eager", b_eager ? "yes" : "NO", b_split ? "yes" : "NO",
                    b_host ? "yes" : "NO", b_first ? "yes" : "NO");
        std::printf("peak device memory %.2f GB (cudaMemGetInfo, incl. %.2f GB context and %.2f GB bench buffers); "
                    "backend weights %.2f GB, activations %.2f GB\n",
                    peak_gb, context_gb, (2.0 * in_n + 3.0 * out_n) * 4 / 1e9, w_bytes / 1e9, a_bytes / 1e9);

        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        cudaFree(dx); cudaFree(dxn); cudaFree(dy); cudaFree(dyn); cudaFree(dy2);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(s);

        json det;
        if (determinism) {
            using Hashes = std::map<std::string, uint64_t>;
            auto hashed_run = [&](varda::Model& m) {
                Hashes h;
                std::vector<float> y(out_n);
                varda::ActivationHook hook = [&](const std::string& k, const float* d, const std::vector<int64_t>& shp) {
                    int64_t n = 1;
                    for (auto v : shp) n *= v;
                    uint64_t f = 1469598103934665603ull;  // FNV-1a
                    const auto* b = reinterpret_cast<const unsigned char*>(d);
                    for (int64_t i = 0; i < n * 4; ++i) f = (f ^ b[i]) * 1099511628211ull;
                    h[k] = f;
                };
                m.predict_step(x, y.data(), &hook);
                h["<output>"] = same(y, y_dev) ? 0 : 1;
                return h;
            };
            Hashes h1 = hashed_run(*model), h2 = hashed_run(*model);
            model.reset();
            auto m3 = varda::Model::load(model_path, opts);
            Hashes h3 = hashed_run(*m3);
            std::vector<std::string> d_same, d_other;
            for (const auto& kv : h1) {
                if (h2[kv.first] != kv.second) d_same.push_back(kv.first);
                if (h3[kv.first] != kv.second) d_other.push_back(kv.first);
            }
            const bool out_ok = h1["<output>"] == 0 && h3["<output>"] == 0;
            std::printf("determinism: %zu keys; differ between runs of one instance: %zu, between instances: %zu; "
                        "hooked output == device output: %s\n",
                        h1.size() - 1, d_same.size(), d_other.size(), out_ok ? "yes" : "NO");
            for (const auto& k : d_same) std::printf("  same-instance difference: %s\n", k.c_str());
            for (const auto& k : d_other) std::printf("  cross-instance difference: %s\n", k.c_str());
            det = {{"keys", h1.size() - 1}, {"differ_same_instance", d_same}, {"differ_other_instance", d_other},
                   {"hooked_output_equals_device", out_ok}};
            if (!d_same.empty() || !d_other.empty() || !out_ok) status = 1;
        }

        rep = {{"model", model_path}, {"uuid", cfg.uuid}, {"input", input_path}, {"gpu", prop.name},
               {"device", "cuda"}, {"precision", varda::to_string(opts.precision)}, {"cuda_graph", opts.cuda_graph},
               {"measurement", std::string("warm hook-free predict_step_device on device-resident input/output, CUDA "
                                           "events, ") + (opts.cuda_graph ? "CUDA graph replay" : "eager launches")},
               {"median_ms", median(times)}, {"min_ms", vmin(times)}, {"mean_ms", vmean(times)}, {"times_ms", times},
               {"iters", iters}, {"warmup", warmup}, {"first_call_s", first_s}, {"load_s", load_s},
               {"host_predict_step_ms", host_ms}, {"host_predict_step_median_ms", median(host_ms)},
               {"peak_mem_gb", peak_gb}, {"context_mem_gb", context_gb}, {"weight_bytes", w_bytes},
               {"activation_bytes", a_bytes}, {"decoder_chunk_rows", chunk_rows},
               {"bitwise", {{"replay_vs_eager", b_eager}, {"split_vs_fused", b_split}, {"host_vs_device", b_host},
                            {"first_call_vs_device", b_first}}}};
        if (determinism) rep["determinism"] = det;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rep["exception"] = e.what();
        status = 1;
    }
    rep["status"] = status == 0 ? "pass" : "fail";
    if (!json_path.empty()) {
        std::ofstream f(json_path + ".tmp");
        f << rep.dump(1, ' ', false, json::error_handler_t::replace) << "\n";
        f.close();
        std::rename((json_path + ".tmp").c_str(), json_path.c_str());
    }
    std::printf("%s\n", status == 0 ? "PASS" : "FAIL");
    return status;
}
#endif
