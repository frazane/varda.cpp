// Real-data input: KENDA-CH1 analyses and ECMWF open-data IFS fields (GRIB) -> the model input
// window x0, exactly as anemoi-inference builds it with the Hugging Face inference_opendata.yaml
// (docs/spec-io.md §1-3; API and verification: docs/io-input.md).
//
//   lam_0  (KENDA, 1 147 980 points): model levels -> pressure levels (log p, extrapolation) and
//          FI from HHL; surface fields as decoded; constants HSURF*g, FR_LAND, SSO_*.
//   global (IFS 0.25°): MIR-linear regrid to N320 (16-bit re-encode), gh*g -> z_*; constants from
//          an explicitly named run.
//   cutout merge with the model's masks, computed forcings, x0 (1, T, G, V_in) in model-input order.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "varda/config.h"
#include "varda/forcings.h"
#include "varda/varda.h"

namespace varda::io {

struct IoConfig;

struct InputFiles {
    std::string date;  // t: "YYYY-MM-DDTHH" (or any form parse_iso8601_utc accepts)
    // KENDA cache: raw/<YYYYmmddTHHMMSS>/<VAR>.grib2 for every input date, and
    // constants/{horizontal,vertical}_constants_icon-ch1-eps.grib2.
    std::string kenda_dir;
    // ECMWF open-data GRIB files covering t-6h and t (any number each; messages are selected by
    // shortName, level and validity date-time).
    std::vector<std::string> ifs_files_prev, ifs_files_t;
    // The run the global constants (z lsm sdor slor, step 0) are taken from (spec §1.5: the
    // reference takes the latest run at start time; here it is explicit).
    std::vector<std::string> ifs_constant_files;
    std::string regrid_matrix;      // python/export_regrid_matrix.py output (.safetensors) or MIR's .mat
    std::string n320_template;      // an N320 grid_ccsds GRIB2 file (HF templates/ifs-levtype=sfc.grib)
    std::string io_config;          // varda-io.json (python/export_io_config.py)
    std::string cosmo_definitions;  // eccodes-cosmo-resources definitions (spec §1.2)
    bool allow_nans = false;        // reference: allow_nans true only reports; default here: stop
    bool keep_stages = false;       // keep the per-stage states (InputState::stages) for dumps
};

// A state as anemoi holds it: name -> float32 [n_dates][n_points], in insertion order.
struct FieldState {
    size_t n_dates = 0, n_points = 0;
    std::vector<std::string> names;
    std::vector<std::vector<float>> values;
    std::vector<double> latitudes, longitudes;  // when known (global: the N320 geometry)

    FieldState() = default;
    FieldState(size_t nd, size_t np) : n_dates(nd), n_points(np) {}
    bool has(const std::string& name) const;
    const std::vector<float>& at(const std::string& name) const;  // throws if absent
    std::vector<float>& at(const std::string& name);
    std::vector<float>& add(const std::string& name);  // NaN-filled; throws on a duplicate
};

// The intermediate states, named like the reference dumps (docs/realdata-reference.md):
//   source.<purpose>.<src>  per cutout source after its pre-processors (global before the mask)
//   cutout.<purpose>        the merged state (2, G)
// plus, per regridded field, MIR's double-decoded output (key "<shortName>@<level>@<YYYYmmddHHMM>+<step>").
struct InputStages {
    std::map<std::string, FieldState> states;
    std::map<std::string, std::vector<double>> regrid;
};

struct InputState {
    // (1, T, G, V_in) physical float32 in model-input order, slot 0 = t - 6h (oldest).
    std::vector<float> x0;
    int64_t T = 0, G = 0, V_in = 0;
    DateTime date0;               // t, the newest slot's date
    std::vector<DateTime> dates;  // every slot's date, oldest first
    // Step-0 output (write_initial_state, spec §4.1): the 66 prognostics then z lsm slor sdor at
    // date t on the merged grid, in io_config.initial_state_fields order.
    std::vector<std::string> initial_names;
    std::vector<std::vector<float>> initial_values;  // [70][G]
    std::vector<std::string> log;  // provenance: files, runs, messages, timings (one line each)
    std::map<std::string, double> seconds;  // per stage
    InputStages stages;  // filled only with InputFiles::keep_stages
};

// What build_input needs from a model: its config, data-grid coordinates and cutout masks.
struct InputModel {
    Config config;
    std::vector<double> latitudes, longitudes;
    std::vector<Model::CutoutSource> cutout;

    static InputModel from(const Model& m);
    // From a converted file without touching the weights (config, grid.*, cutout masks).
    static InputModel load(const std::string& model_path);
};

InputState build_input(const InputModel& model, const InputFiles& files);
InputState build_input(const Model& model, const InputFiles& files);

// "YYYY-MM-DDTHH" (and the forms parse_iso8601_utc accepts).
DateTime parse_input_date(const std::string& s);

// Fills `f` (date unless already set, kenda_dir, the IFS files) from a frozen real-data directory
// written by python/reference/realdata/fetch_realdata.py: DIR/pins.json and DIR/cache/{kenda,ecmwf/files}.
void files_from_realdata(const std::string& dir, InputFiles& f);

// The KENDA model_to_pressure_level configuration of inference_opendata.yaml (spec §2.2).
const std::vector<int>& kenda_interpolate_levels();  // 50 .. 1000 hPa (13)
const std::vector<int>& kenda_extrapolate_levels();  // 500 600 700 850 925 1000

}  // namespace varda::io
