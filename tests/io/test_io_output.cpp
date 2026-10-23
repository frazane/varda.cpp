// Unit tests of the GRIB output side (src/varda/io/output.*, io_config.*). The pure-function parts need
// the exported io config ($VARDA_IO_CONFIG, default $VARDA_SCRATCH/realdata/assets/varda-io.json); the
// end-to-end part also needs the HF templates ($VARDA_TEMPLATES, default $VARDA_SCRATCH/hf/templates)
// and the COSMO definitions ($VARDA_COSMO_DEFINITIONS). Missing inputs skip, they do not fail.
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

#include "varda/io/grib.h"
#include "varda/io/io_config.h"
#include "varda/io/output.h"

using namespace varda;
using namespace varda::io;

static int fails = 0;
static void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    fails += !ok;
}
static bool exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
static std::string env_or(const char* name, const std::string& dflt) {
    const char* v = std::getenv(name);
    return v ? v : dflt;
}
static std::string join(const ScalarMap& m) {
    std::string s;
    for (const auto& [k, v] : m) s += (s.empty() ? "" : ",") + k + "=" + v.str();
    return s;
}

static void test_pure() {
    expect(py_parses_as_int("129") && py_parses_as_int(" -7 ") && py_parses_as_int("1_000"), "py int: digits");
    expect(!py_parses_as_int("T") && !py_parses_as_int("") && !py_parses_as_int("1__0") && !py_parses_as_int("1.5"), "py int: rejects");
    expect(py_parses_as_float("1.5") && py_parses_as_float("nan") && !py_parses_as_float("T_2M"), "py float");
    expect(param_key_name("128") == "paramId" && param_key_name("1.5") == "param" && param_key_name("TOT_PREC") == "shortName",
           "param key name");

    // accumulate: acc = f32(acc + max(0, x)), NaN propagates
    std::vector<float> acc;
    accumulate_into(acc, {1.0f, -2.0f, 0.1f, std::nanf("")}, false);
    accumulate_into(acc, {0.2f, 3.0f, 0.2f, 1.0f}, false);
    expect(acc[0] == 1.0f + 0.2f && acc[1] == 3.0f && acc[2] == 0.1f + 0.2f && std::isnan(acc[3]), "accumulate (negative clamp, float32, NaN kept)");
    std::vector<float> acc2;
    accumulate_into(acc2, {-1.0f}, true);
    expect(acc2[0] == -1.0f, "accumulate allow_negative");
    auto r = rescale({0.1f, 2.5f}, 1000, 0);
    expect(r[0] == 0.1f * 1000.0f && r[1] == 2500.0f, "rescale x1000 in float32");
    auto a = assign_mask({1, 2, 3}, {0, 1, 0, 1, 1});
    expect(std::isnan(a[0]) && a[1] == 1 && std::isnan(a[2]) && a[3] == 2 && a[4] == 3, "assign_mask scatters in order");
    bool threw = false;
    try { assign_mask({1, 2}, {1, 1, 1}); } catch (const std::exception&) { threw = true; }
    expect(threw, "assign_mask size mismatch throws");
    std::vector<float> v = {1.0f, std::nanf(""), INFINITY, -INFINITY};
    bool any = nan_to_missing(v, -9999.0f);
    expect(any && v[1] == -9999.0f && v[2] == std::numeric_limits<float>::max() && v[3] == -std::numeric_limits<float>::max(), "nan_to_missing");
    std::vector<float> clean = {1.0f, INFINITY};
    expect(!nan_to_missing(clean, -9999.0f) && std::isinf(clean[1]), "no NaN: untouched");
    expect(render_path("outputs/global-{dateTime}_{step}.grib", "202610221500", 6) == "outputs/global-202610221500_6.grib", "render_path");
}

static void test_config(const std::string& cfgpath) {
    IoConfig c = IoConfig::load(cfgpath);
    expect(c.variables.size() == 98 && c.model_variables.size() == 80 && c.output_variables.size() == 67, "config sizes 98/80/67");
    expect(c.initial_state_fields.size() == 70 && c.initial_state_fields[0] == "z_100" && c.initial_state_fields[66] == "z" &&
               c.initial_state_fields[69] == "sdor",
           "initial-state order: 66 prognostics then z lsm slor sdor");
    expect(c.output_variables[18] == "tp" && c.output_variables[0] == "10u" && c.output_variables[66] == "z_925", "output order (tp index 18)");
    expect(c.kenda_name("FI", 500).value() == "z_500" && c.kenda_name("T_2M", 0).value() == "2t" && !c.kenda_name("HHL", 1) &&
               c.kenda_name("SSO_STDH", 0).value() == "sdor",
           "KENDA namer");
    expect(c.outputs.size() == 2 && c.outputs[0].name == "lam" && c.outputs[1].name == "global", "two outputs");
    expect(c.outputs[0].path_effective == "outputs/{dateTime}_{step}.grib" &&
               c.outputs[1].path_effective == "outputs/global-{dateTime}_{step}.grib",
           "file name pattern: {step:03} loses its spec");
    expect(c.post_processors.size() == 2 && c.post_processors[0].type == PostProcessorDef::Accumulate &&
               c.post_processors[1].type == PostProcessorDef::Rescale && c.post_processors[1].scale == 1000,
           "top-level post-processors");

    const DateTime t{2026, 10, 22, 15, 0, 0};
    const OutputDef& lam = c.output("lam");
    const OutputDef& glo = c.output("global");
    auto keys = [&](const OutputDef& o, const char* name, int step, std::optional<int> start = std::nullopt) {
        ScalarMap k = build_keys(c.variable(name), o.encoding, 2, t, step, start);
        for (const auto& m : o.modifiers) k = apply_patches(m, name, k);
        return join(k);
    };
    expect(keys(lam, "t_500", 6) ==
               "edition=2,typeOfLevel=isobaricInhPa,stepType=instant,typeOfGeneratingProcess=2,shortName=T,dataType=fc,date=20261022,time=1500,step=6,level=500",
           "keys lam pl: " + keys(lam, "t_500", 6));
    expect(keys(lam, "2t", 12) ==
               "edition=2,stepType=instant,typeOfGeneratingProcess=2,shortName=T_2M,dataType=fc,date=20261022,time=1500,step=12",
           "keys lam sfc");
    expect(keys(lam, "tp", 18, 0) ==
               "edition=2,stepType=accum,productDefinitionTemplateNumber=8,typeOfGeneratingProcess=2,shortName=TOT_PREC,dataType=fc,date=20261022,time=1500,startStep=0,endStep=18",
           "keys lam tp: " + keys(lam, "tp", 18, 0));
    expect(keys(lam, "z", 0) ==
               "edition=2,stepType=instant,typeOfGeneratingProcess=2,shortName=FIS,dataType=fc,date=20261022,time=1500,step=0",
           "keys lam z (null levelist dropped)");
    expect(keys(glo, "z_500", 6) ==
               "edition=2,typeOfLevel=isobaricInhPa,stepType=instant,typeOfGeneratingProcess=2,shortName=z,dataType=fc,date=20261022,time=1500,step=6,level=500,param=129",
           "keys global pl: patch replaces shortName in place, appends param: " + keys(glo, "z_500", 6));
    expect(keys(glo, "tp", 6, 0) ==
               "edition=2,stepType=accum,productDefinitionTemplateNumber=8,typeOfGeneratingProcess=2,shortName=tp,dataType=fc,date=20261022,time=1500,startStep=0,endStep=6,param=228",
           "keys global tp");
    struct P { const char* name; const char* sn; int pid; };
    bool ok = true;
    for (P p : {P{"10u", "10u", 165}, P{"10v", "10v", 166}, P{"2d", "2d", 168}, P{"2t", "2t", 167}, P{"lsm", "lsm", 172},
                P{"msl", "msl", 151}, P{"sp", "sp", 134}, P{"slor", "slor", 163}, P{"sdor", "sdor", 160}, P{"z", "z", 129},
                P{"q_50", "q", 133}, P{"t_925", "t", 130}, P{"u_100", "u", 131}, P{"v_1000", "v", 132}, P{"z_850", "z", 129}}) {
        ScalarMap k = build_keys(c.variable(p.name), glo.encoding, 2, t, 6, std::nullopt);
        k = apply_patches(glo.modifiers[0], p.name, k);
        const Scalar* s = find(k, "shortName");
        const Scalar* i = find(k, "param");
        if (!s || !i || s->s != p.sn || i->i != p.pid) { ok = false; std::printf("   patch %s -> %s\n", p.name, join(k).c_str()); }
    }
    expect(ok, "global patches: 15 names -> shortName + paramId");

    auto tf = [&](const OutputDef& o, const char* name) {
        const TemplateIndexEntry* e = match_template(o, c.variable(name));
        return e ? e->file : std::string("<none>");
    };
    expect(tf(lam, "t_500") == "icon-ch1-typeOfLevel=isobaricInhPa.grib" && tf(lam, "z_50") == "icon-ch1-typeOfLevel=isobaricInhPa.grib" &&
               tf(lam, "2t") == "icon-ch1-typeOfLevel=heightAboveGround.grib" && tf(lam, "10v") == "icon-ch1-typeOfLevel=heightAboveGround.grib" &&
               tf(lam, "z") == "icon-ch1-typeOfLevel=surface.grib" && tf(lam, "lsm") == "icon-ch1-typeOfLevel=surface.grib" &&
               tf(lam, "sp") == "icon-ch1-typeOfLevel=surface.grib" && tf(lam, "sdor") == "icon-ch1-typeOfLevel=surface.grib" &&
               tf(lam, "msl") == "icon-ch1-typeOfLevel=meanSea.grib" && tf(lam, "tp") == "icon-ch1-shortName=TOT_PREC.grib",
           "ICON template matching");
    expect(tf(glo, "q_50") == "ifs-levtype=pl.grib" && tf(glo, "z_925") == "ifs-levtype=pl.grib" && tf(glo, "10u") == "ifs-levtype=sfc.grib" &&
               tf(glo, "tp") == "ifs-levtype=sfc.grib" && tf(glo, "z") == "ifs-levtype=sfc.grib",
           "IFS template matching");
    expect(tf(lam, "cos_latitude") == "<none>", "no template for a computed forcing");
}

// End to end on a synthetic grid with the real templates: counts, names, keys, tp, bitmap.
static void test_end_to_end(const std::string& cfgpath, const std::string& tdir) {
    IoConfig c = IoConfig::load(cfgpath);
    const int64_t n0 = 1147980, n1 = 540670, N320 = 542080;
    std::vector<GridSource> src(2);
    src[0].name = "lam_0";
    src[0].mask.assign(size_t(n0), 1);
    src[1].name = "global";
    src[1].mask.assign(size_t(N320), 1);
    for (int64_t k = 0; k < 1410; ++k) src[1].mask[size_t(62219 + 20 * k)] = 0;  // 1410 overlapping points (spread)
    const int64_t G = n0 + n1;
    namespace fs = std::filesystem;
    const std::string out = env_or("TMPDIR", "/tmp") + "/varda_test_io_output_" + std::to_string(::getpid());
    fs::create_directories(out);
    const DateTime t{2026, 10, 22, 15, 0, 0};
    {
        GribOutput go(src, c.output_variables, c, tdir, out, t);
        std::vector<std::vector<float>> init;
        for (size_t k = 0; k < c.initial_state_fields.size(); ++k) {
            std::vector<float> f(static_cast<size_t>(G));
            for (int64_t g = 0; g < G; ++g) f[size_t(g)] = 100.0f + float(k) + 0.001f * float(g % 1000);
            init.push_back(std::move(f));
        }
        go.write_initial(c.initial_state_fields, init);
        const size_t V = c.output_variables.size();
        std::vector<float> y(size_t(G) * V);
        for (int s = 1; s <= 2; ++s) {
            for (int64_t g = 0; g < G; ++g)
                for (size_t j = 0; j < V; ++j) y[size_t(g) * V + j] = float(j) + 0.5f * float(s) + 0.01f * float(g % 100);
            // tp (index 18): a negative value at g=0 must be clamped
            y[18] = -1.0f;
            go.write_step(6 * s, y.data());
        }
        go.close();
    }
    for (const char* f : {"outputs/202610221500_0.grib", "outputs/202610221500_6.grib", "outputs/202610221500_12.grib",
                          "outputs/global-202610221500_0.grib", "outputs/global-202610221500_6.grib",
                          "outputs/global-202610221500_12.grib"})
        expect(exists(out + "/" + f), std::string("file ") + f);
    auto m0 = read_messages(out + "/outputs/202610221500_0.grib");
    auto m6 = read_messages(out + "/outputs/202610221500_6.grib");
    auto g0 = read_messages(out + "/outputs/global-202610221500_0.grib");
    auto g12 = read_messages(out + "/outputs/global-202610221500_12.grib");
    expect(m0.size() == 70 && m6.size() == 67 && g0.size() == 70 && g12.size() == 67, "message counts 70/67/70/67");
    if (m6.size() == 67 && g12.size() == 67 && m0.size() == 70 && g0.size() == 70) {
        expect(m6[18].get_string("shortName") == "TOT_PREC" && m6[18].get_long("endStep") == 6 && m6[18].get_long("startStep") == 0 &&
                   m6[18].get_long("productDefinitionTemplateNumber") == 8,
               "lam tp: index 18, accum 0-6, PDT 8");
        auto tp6 = m6[18].get_double_array();
        expect(std::fabs(tp6[0] - 0.0) < 1e-9, "tp at point 0 clamped to 0 (value " + std::to_string(tp6[0]) + ")");
        auto tp12 = read_messages(out + "/outputs/202610221500_12.grib")[18].get_double_array();
        // step 1: j=18, s=1: 17.5 + 0.01*(g%100) ; step 2: 18.0 + ...; acc*1000
        const double e12 = (double(float(18.5f + 0.01f * 5)) + double(float(19.0f + 0.01f * 5))) * 1000.0;
        expect(std::fabs(tp12[5] - e12) / e12 < 1e-6, "tp at step 12 = 1000 * (sum of two steps): " + std::to_string(tp12[5]) + " vs " + std::to_string(e12));
        expect(g12[18].get_string("shortName") == "tp" && g12[18].get_long("paramId") == 228 && g12[18].get_long("bitmapPresent") == 1,
               "global tp: tp / 228 / bitmap");
        auto gv = g12[0].get_double_array();
        expect(gv.size() == size_t(N320) && g12[0].get_long("numberOfMissing") == 1410 && g12[0].get_long("bitmapPresent") == 1,
               "global message: 542080 values, bitmap with 1410 missing");
        expect(m6[0].get_long("date") == 20261022 && m6[0].get_long("time") == 1500 && m6[0].get_string("dataType") == "fc" &&
                   m6[0].get_long("typeOfGeneratingProcess") == 2 && m6[0].get_long("generatingProcessIdentifier") == 255,
               "lam header keys");
        expect(m0[66].get_string("shortName") == "FIS" && m0[67].get_string("shortName") == "FR_LAND" && m0[69].get_string("shortName") == "SSO_STDH",
               "step 0: constants last, FIS/FR_LAND/.../SSO_STDH");
        expect(m0[0].get_string("shortName") == "FI" && m0[0].get_long("level") == 100 && m0[0].get_long("step") == 0, "step 0: first message z_100 -> FI 100 hPa");
        expect(m0[0].get_long("bitmapPresent") == 0, "lam: no bitmap without NaN");
    }
    fs::remove_all(out);  // created by this test
}

int main() {
    test_pure();
    const std::string scratch = env_or("VARDA_SCRATCH", "");
    const std::string cfg = env_or("VARDA_IO_CONFIG", scratch + "/realdata/assets/varda-io.json");
    if (scratch.empty() && !std::getenv("VARDA_IO_CONFIG")) {
        std::printf("skip config tests (VARDA_SCRATCH / VARDA_IO_CONFIG unset)\n");
        return fails ? 1 : 0;
    }
    if (!exists(cfg)) {
        std::printf("skip config tests (%s missing)\n", cfg.c_str());
        return fails ? 1 : 0;
    }
    test_config(cfg);
    const std::string tdir = env_or("VARDA_TEMPLATES", scratch + "/hf/templates");
    if (!exists(tdir + "/ifs-levtype=pl.grib") || !std::getenv("VARDA_COSMO_DEFINITIONS")) {
        std::printf("skip end-to-end (templates or VARDA_COSMO_DEFINITIONS missing)\n");
        return fails ? 1 : 0;
    }
    test_end_to_end(cfg, tdir);
    return fails ? 1 : 0;
}
