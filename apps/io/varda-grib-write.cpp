// varda-grib-write: the encoder-only driver of the GRIB output (docs/io-output.md §6, docs/spec-io.md §5.2 C3-C6).
// Feeds a reference run's own dumped states into GribOutput, so that no model and no input side is involved:
//
//   varda-grib-write --io-config varda-io.json --templates DIR --model MODEL.safetensors
//                    --dumps REF/dumps --out OUT --date 2026-10-22T15 [--steps 4]
//
//   * step 0:  dumps/state0.pre.safetensors  (the 70 fields of the initial state, as the reference read them)
//   * step s:  dumps/step<s-1>.y.safetensors (the model output y [1,1,1,G,67], physical, bounded)
//   * C3:      tp post-processing (accumulate, x1000) recomputed from the dumped state<s>.pre and compared
//              bitwise with the reference's state<s>.1.ForwardTransformFilter tp (when dumped);
//              y vs state<s>.pre fields (they must be the same floats).
//   * the cutout masks come from the converted model file (grid.<source>.cutout_mask), read without loading
//     the model. Writes OUT/outputs/*.grib; compare with varda-grib-compare OUT/outputs REF/outputs.
// Needs VARDA_COSMO_DEFINITIONS (the COSMO ecCodes definitions).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "varda/forcings.h"
#include "varda/format.h"
#include "varda/io/grib.h"
#include "varda/io/io_config.h"
#include "varda/io/output.h"

using namespace varda;
using namespace varda::io;

int main(int argc, char** argv) {
    std::string cfgp, tdir, modelp, dumps, out, date;
    int steps = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--io-config") cfgp = next();
        else if (a == "--templates") tdir = next();
        else if (a == "--model") modelp = next();
        else if (a == "--dumps") dumps = next();
        else if (a == "--out") out = next();
        else if (a == "--date") date = next();
        else if (a == "--steps") steps = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (cfgp.empty() || tdir.empty() || modelp.empty() || dumps.empty() || out.empty() || date.empty()) {
        std::fprintf(stderr, "usage: varda-grib-write --io-config J --templates D --model M --dumps D --out D --date ISO [--steps N]\n");
        return 2;
    }
    try {
        const double t0 = app::now_s();
        IoConfig cfg = IoConfig::load(cfgp);
        const DateTime t = parse_iso8601_utc(date);

        SafeTensorsFile mf(modelp);
        std::vector<GridSource> src;
        for (const auto& name : nlohmann::json::parse(mf.metadata("cutout"))) {
            const TensorView& m = mf.peek("grid." + name.get<std::string>() + ".cutout_mask");
            GridSource s;
            s.name = name.get<std::string>();
            s.mask.assign(m.as<uint8_t>(), m.as<uint8_t>() + m.numel());
            src.push_back(std::move(s));
        }
        GribOutput go(src, cfg.output_variables, cfg, tdir, out, t);
        const int64_t G = go.n_points();
        std::printf("G %lld, io config %s, ecCodes %ld\n", (long long)G, cfgp.c_str(), eccodes_version());

        int fails = 0;
        auto check = [&](bool ok, const std::string& what) {
            std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
            fails += !ok;
        };

        {  // step 0
            SafeTensorsFile f(dumps + "/state0.pre.safetensors");
            const auto names = nlohmann::json::parse(f.metadata("meta")).at("names").get<std::vector<std::string>>();
            check(names == cfg.initial_state_fields, "dumped initial-state fields == io config initial_state_fields (" + std::to_string(names.size()) + ")");
            std::vector<std::vector<float>> vals;
            for (const auto& n : names) {
                const TensorView& v = f.peek("field/" + n);
                if (v.numel() != G) throw std::runtime_error("field " + n + " has " + std::to_string(v.numel()) + " values");
                vals.emplace_back(v.as<float>(), v.as<float>() + G);
            }
            const double a = app::now_s();
            go.write_initial(names, vals);
            std::printf("step 0 written (%zu fields) in %.1f s\n", names.size(), app::now_s() - a);
        }

        std::vector<float> acc;
        for (int s = 1; s <= steps; ++s) {
            SafeTensorsFile fy(dumps + "/step" + std::to_string(s - 1) + ".y.safetensors");
            const TensorView& y = fy.peek("y");
            const size_t V = cfg.output_variables.size();
            if (y.numel() != int64_t(G * V)) throw std::runtime_error("y has the wrong size");
            // C3: the pre-state fields are the same floats as y
            const std::string spre = dumps + "/state" + std::to_string(s) + ".pre.safetensors";
            {
                SafeTensorsFile fp(spre);
                const auto names = nlohmann::json::parse(fp.metadata("meta")).at("names").get<std::vector<std::string>>();
                check(names == cfg.output_variables, "state" + std::to_string(s) + ".pre names == output variables");
                long diff = 0;
                for (size_t j = 0; j < V; ++j) {
                    const float* p = fp.peek("field/" + names[j]).as<float>();
                    for (int64_t g = 0; g < G; ++g)
                        if (std::memcmp(&p[g], &y.as<float>()[size_t(g) * V + j], 4) != 0) ++diff;
                }
                check(diff == 0, "y == state" + std::to_string(s) + ".pre fields bitwise (" + std::to_string(diff) + " differ)");
                // tp post-processing against the reference's ForwardTransformFilter output
                const size_t jt = size_t(std::find(names.begin(), names.end(), "tp") - names.begin());
                const float* tp = fp.peek("field/tp").as<float>();
                accumulate_into(acc, std::vector<float>(tp, tp + G), false);
                const auto r = rescale(acc, 1000, 0);
                const std::string ff = dumps + "/state" + std::to_string(s) + ".1.ForwardTransformFilter.safetensors";
                if (FILE* e = std::fopen(ff.c_str(), "rb")) {
                    std::fclose(e);
                    SafeTensorsFile fr(ff);
                    const float* ref = fr.peek("field/tp").as<float>();
                    long d = 0;
                    for (int64_t g = 0; g < G; ++g) d += std::memcmp(&r[size_t(g)], &ref[g], 4) != 0;
                    check(d == 0, "step " + std::to_string(s) + " tp (accumulated, x1000) == reference bitwise (" + std::to_string(d) + " differ)");
                }
                (void)jt;
            }
            const double a = app::now_s();
            go.write_step(6 * s, y.as<float>());
            std::printf("step %d (+%dh) written in %.1f s\n", s, 6 * s, app::now_s() - a);
        }
        go.close();
        for (const auto& f : go.files()) std::printf("wrote %s\n", f.c_str());
        std::printf("done in %.1f s; %s\n", app::now_s() - t0, fails ? "CHECKS FAILED" : "all checks ok");
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "varda-grib-write: %s\n", e.what());
        return 2;
    }
}
