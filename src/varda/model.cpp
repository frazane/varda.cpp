#include <stdexcept>

#include "nlohmann/json.hpp"

#include "varda/backend.h"
#include "varda/format.h"
#include "varda/varda.h"

namespace varda {

std::string to_string(Device d) {
    switch (d) {
        case Device::CPU: return "cpu";
        case Device::CUDA: return "cuda";
        case Device::Metal: return "metal";
    }
    throw std::runtime_error("unknown compute device");
}
std::string to_string(Precision p) { return p == Precision::FP32 ? "fp32" : "fp16"; }

Model::~Model() = default;

std::unique_ptr<Model> Model::load(const std::string& path, const Options& opts) {
    std::unique_ptr<Model> m(new Model());
    m->opts_ = opts;
    m->file_ = std::make_unique<SafeTensorsFile>(path);
    m->file_->prefetch();  // every byte is read below: one parallel pass beats serial page faults
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
    if (m->file_->metadata().count("cutout")) {
        int64_t kept = 0;
        for (const auto& name : nlohmann::json::parse(m->file_->metadata("cutout"))) {
            Model::CutoutSource src{name.get<std::string>(), {}};
            const TensorView& t = m->file_->take("grid." + src.name + ".cutout_mask");
            if (t.dtype != DType::U8 || t.shape.size() != 1)
                throw std::runtime_error(path + ": grid." + src.name + ".cutout_mask must be U8 [n]");
            src.mask.assign(t.as<uint8_t>(), t.as<uint8_t>() + t.numel());
            for (uint8_t v : src.mask) {
                if (v > 1) throw std::runtime_error(path + ": cutout mask " + src.name + " is not 0/1");
                kept += v;
            }
            m->cutout_.push_back(std::move(src));
        }
        if (kept != m->cfg_.n_data_nodes)
            throw std::runtime_error(path + ": cutout masks keep " + std::to_string(kept) +
                                     " points, the data grid has " + std::to_string(m->cfg_.n_data_nodes));
    }

    if (opts.device == Device::CPU) {
        if (opts.precision != Precision::FP32)
            throw std::runtime_error("the CPU backend runs fp32 only");
        m->backend_ = make_cpu_backend(m->cfg_, *m->file_, opts);
    } else if (opts.device == Device::CUDA) {
#ifdef VARDA_WITH_CUDA
        m->backend_ = make_cuda_backend(m->cfg_, *m->file_, opts);
#else
        throw std::runtime_error("varda.cpp was built without CUDA (VARDA_CUDA=OFF)");
#endif
    } else if (opts.device == Device::Metal) {
#ifdef VARDA_WITH_METAL
        m->backend_ = make_metal_backend(m->cfg_, *m->file_, opts);
#else
        throw std::runtime_error("varda.cpp was built without Metal (VARDA_METAL=OFF)");
#endif
    } else {
        throw std::runtime_error("unknown compute device");
    }
    m->file_->require_all_consumed();
    return m;
}

void Model::predict_step(const float* x, float* y, const ActivationHook* hook) {
    backend_->predict_step(x, y, hook);
}

}  // namespace varda
