// Shared helpers for the apps: contract keys, golden files, tolerance lookup, metric, timing.
#pragma once

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "varda/config.h"
#include "varda/format.h"
#include "varda/varda.h"
#include <algorithm>

#ifndef VARDA_SOURCE_DIR
#define VARDA_SOURCE_DIR "."
#endif

namespace app {

using json = nlohmann::json;

// docs/format.md §2, in production order.
inline std::vector<std::string> contract_keys(int n_proc) {
    std::vector<std::string> k = {"in.x", "in.x_norm", "enc.x_src", "enc.x_dst", "enc.emb_src", "enc.emb_dst", "enc.out"};
    for (int i = 0; i < n_proc; ++i) k.push_back("proc." + std::to_string(i) + ".out");
    for (const char* s : {"latent", "dec.emb_dst", "dec.block_out", "dec.out", "out.pre_bound", "out.norm", "out.y"})
        k.push_back(s);
    return k;
}

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
    return out + "]";
}

// Golden activations (docs/format.md §3).
struct Golden {
    std::unique_ptr<varda::SafeTensorsFile> file;
    std::string path, tag, uuid;
    json manifest;

    explicit Golden(const std::string& p) : file(new varda::SafeTensorsFile(p)), path(p) {
        if (file->metadata("varda_golden") != "1") throw std::runtime_error(p + ": not a varda golden file");
        tag = file->metadata("tag");
        uuid = file->metadata("checkpoint_uuid");
        manifest = json::parse(file->metadata("manifest"));
    }
};

// Tolerance tables (docs/format.md §4): explicit file, or the file in tolerances/ whose
// checkpoint.uuid matches.
inline json load_tolerances(const std::string& explicit_path, const std::string& uuid, std::string* used_path) {
    std::vector<std::string> candidates;
    if (!explicit_path.empty()) {
        candidates.push_back(explicit_path);
    } else {
        const std::string dir = std::string(VARDA_SOURCE_DIR) + "/tolerances";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                std::string n = e->d_name;
                if (n.size() > 5 && n.substr(n.size() - 5) == ".json") candidates.push_back(dir + "/" + n);
            }
            closedir(d);
        }
    }
    if (uuid.empty()) throw std::runtime_error("model has no checkpoint uuid: cannot pair it with tolerances");
    std::sort(candidates.begin(), candidates.end());
    for (const auto& c : candidates) {
        std::ifstream f(c);
        if (!f) throw std::runtime_error("cannot read tolerances " + c);
        json t = json::parse(f);
        const std::string tu = t.contains("checkpoint") ? t["checkpoint"].value("uuid", "") : "";
        if (tu == uuid) {
            if (used_path) *used_path = c;
            return t;
        }
        if (!explicit_path.empty())
            throw std::runtime_error(c + ": checkpoint uuid " + tu + " does not match the model's " + uuid);
    }
    throw std::runtime_error("no tolerance file for checkpoint uuid " + uuid + " in " VARDA_SOURCE_DIR "/tolerances");
}

struct Metric {
    double max_abs = 0, absmax = 0, rel_absmax = 0;
    int64_t n_nan = 0;      // NaN in the engine output
    int64_t n_ref_bad = 0;  // NaN or Inf in the golden tensor
};

// docs/format.md §2: float64, rel_absmax = max|a - r| / max|r|. NaN in `a` (where `r` is
// finite) and any non-finite `r` are counted; either makes the key fail. Inf in `a` gives an
// infinite max_abs.
inline Metric compare(const float* a, const float* r, int64_t n) {
    double max_abs = 0, absmax = 0;
    int64_t nan = 0, ref_bad = 0;
#pragma omp parallel for reduction(max : max_abs, absmax) reduction(+ : nan, ref_bad) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double av = a[i], rv = r[i];
        if (!std::isfinite(rv)) { ++ref_bad; continue; }
        if (std::isnan(av)) { ++nan; continue; }
        max_abs = std::max(max_abs, std::fabs(av - rv));
        absmax = std::max(absmax, std::fabs(rv));
    }
    Metric m;
    m.max_abs = max_abs;
    m.absmax = absmax;
    m.n_nan = nan;
    m.n_ref_bad = ref_bad;
    m.rel_absmax = absmax > 0 ? max_abs / absmax : (max_abs > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    return m;
}

inline varda::Device parse_device(const std::string& s) {
    if (s == "cpu") return varda::Device::CPU;
    if (s == "cuda") return varda::Device::CUDA;
    if (s == "metal") return varda::Device::Metal;
    throw std::runtime_error("unknown --device '" + s + "' (cpu | cuda | metal)");
}

inline varda::Precision parse_precision(const std::string& s) {
    if (s == "fp32") return varda::Precision::FP32;
    if (s == "fp16") return varda::Precision::FP16;
    throw std::runtime_error("unknown --precision '" + s + "' (fp32 | fp16)");
}

}  // namespace app
