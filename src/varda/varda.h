// varda.cpp public API: load a converted anemoi GraphTransformer forecaster
// (docs/format.md) and run its predict_step on the CPU or a CUDA GPU.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "varda/config.h"

namespace varda {

enum class Device { CPU, CUDA };
// FP16 reproduces the reference's fp16 autocast (Linear in fp16, LayerNorm and attention
// softmax in fp32). CUDA only.
enum class Precision { FP32, FP16 };

struct Options {
    Device device = Device::CPU;
    Precision precision = Precision::FP32;
    int num_threads = 0;    // CPU: 0 = OpenMP default
    int cuda_device = 0;
    bool cuda_graph = true; // CUDA: capture the forward pass once, replay afterwards
};

// Receives each contract activation (docs/format.md §2) once per predict_step, as host
// fp32, contiguous, row-major with the given shape. `data` is valid only during the call.
using ActivationHook = std::function<void(const std::string& key, const float* data,
                                          const std::vector<int64_t>& shape)>;

class Backend;
class SafeTensorsFile;

class Model {
public:
    static std::unique_ptr<Model> load(const std::string& path, const Options& opts);
    ~Model();

    const Config& config() const { return cfg_; }
    const Options& options() const { return opts_; }

    // x: host (1, T_in, G, V_in) fp32, physical units, model-input order (contract `in.x`).
    // y: host (1, T_out, 1, G, V_out) fp32, physical units (contract `out.y`).
    // With a hook, every contract activation is reported; the output is unchanged by it.
    void predict_step(const float* x, float* y, const ActivationHook* hook = nullptr);

    Backend& backend() { return *backend_; }

    // Data-grid coordinates in degrees (`grid.latitudes` / `grid.longitudes`, F64 [G]),
    // used for the rollout's computed forcings. Empty if the file predates them.
    const std::vector<double>& latitudes() const { return lat_; }
    const std::vector<double>& longitudes() const { return lon_; }

    // Multi-source (cutout) grids: per source, in data-grid order, the 0/1 mask over that source's
    // own points (`grid.<name>.cutout_mask`, U8); the data grid is the concatenation of every
    // source's kept points (docs/format.md §1.1, docs/spec-io.md §2.6). Empty for a single source.
    struct CutoutSource {
        std::string name;
        std::vector<uint8_t> mask;
    };
    const std::vector<CutoutSource>& cutout() const { return cutout_; }

private:
    Model() = default;
    Config cfg_;
    Options opts_;
    std::vector<double> lat_, lon_;
    std::vector<CutoutSource> cutout_;
    std::unique_ptr<SafeTensorsFile> file_;  // outlives backend_ (declared first)
    std::unique_ptr<Backend> backend_;
};

std::string to_string(Device d);
std::string to_string(Precision p);

}  // namespace varda
