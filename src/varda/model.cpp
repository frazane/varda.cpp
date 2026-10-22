#include <stdexcept>

#include "varda/backend.h"
#include "varda/format.h"
#include "varda/varda.h"

namespace varda {

std::string to_string(Device d) { return d == Device::CPU ? "cpu" : "cuda"; }
std::string to_string(Precision p) { return p == Precision::FP32 ? "fp32" : "fp16"; }

Model::~Model() = default;

std::unique_ptr<Model> Model::load(const std::string& path, const Options& opts) {
    std::unique_ptr<Model> m(new Model());
    m->opts_ = opts;
    m->file_ = std::make_unique<SafeTensorsFile>(path);
    if (m->file_->metadata("varda_format") != "1")
        throw std::runtime_error(path + ": unsupported varda_format " + m->file_->metadata("varda_format"));
    m->cfg_ = parse_config(m->file_->metadata("config"));
    if (m->file_->has("grid.latitudes") || m->file_->has("grid.longitudes")) {
        const std::vector<int64_t> shape{m->cfg_.n_data_nodes};
        const TensorView& la = m->file_->take("grid.latitudes", DType::F64, shape);
        const TensorView& lo = m->file_->take("grid.longitudes", DType::F64, shape);
        m->lat_.assign(la.as<double>(), la.as<double>() + la.numel());
        m->lon_.assign(lo.as<double>(), lo.as<double>() + lo.numel());
    }

    if (opts.device == Device::CPU) {
        if (opts.precision != Precision::FP32)
            throw std::runtime_error("the CPU backend runs fp32 only");
        m->backend_ = make_cpu_backend(m->cfg_, *m->file_, opts);
    } else {
#ifdef VARDA_WITH_CUDA
        m->backend_ = make_cuda_backend(m->cfg_, *m->file_, opts);
#else
        throw std::runtime_error("varda.cpp was built without CUDA (VARDA_CUDA=OFF)");
#endif
    }
    m->file_->require_all_consumed();
    return m;
}

void Model::predict_step(const float* x, float* y, const ActivationHook* hook) {
    backend_->predict_step(x, y, hook);
}

}  // namespace varda
