// The GRIB layer links against the ecCodes the reference uses, and the COSMO definitions give
// KENDA-CH1 messages their COSMO shortNames (docs/spec-io.md §1.2). The real-data part runs
// only when VARDA_KENDA_SAMPLE names a KENDA-CH1 T file (it skips otherwise).
#include <cstdio>
#include <cstdlib>
#include <string>

#include "varda/io/grib.h"

using namespace varda::io;

int main() {
    int fails = 0;
    auto expect = [&](bool ok, const std::string& what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
        fails += !ok;
    };
    expect(eccodes_version() == 24703, "ecCodes 2.47.3 (" + std::to_string(eccodes_version()) + ")");

    const char* defs = std::getenv("VARDA_COSMO_DEFINITIONS");
    const char* sample = std::getenv("VARDA_KENDA_SAMPLE");
    if (!defs || !sample) {
        std::printf("skip real-data checks (VARDA_COSMO_DEFINITIONS / VARDA_KENDA_SAMPLE unset)\n");
        return fails ? 1 : 0;
    }
    set_definitions_path(defs);
    set_definitions_path(defs);  // idempotent
    expect(definitions_path().rfind(defs, 0) == 0, "definitions path " + definitions_path());
    auto msgs = read_messages(sample);
    expect(msgs.size() == 80, std::to_string(msgs.size()) + " messages");
    if (!msgs.empty()) {
        expect(msgs[0].get_string("shortName") == "T", "shortName " + msgs[0].get_string("shortName"));
        auto f = msgs[0].get_float_array();
        auto d = msgs[0].get_double_array();
        bool same = f.size() == d.size() && f.size() == 1147980;
        for (size_t i = 0; same && i < f.size(); ++i) same = f[i] == float(d[i]);
        expect(same, "float decode == float(double decode), 1147980 values");
        GribHandle c = msgs[0].clone();
        c.set_key_vals({{"typeOfGeneratingProcess", "2"}, {"generatingProcessIdentifier", "255"}});
        expect(c.get_long("typeOfGeneratingProcess") == 2, "set_key_vals");
    }
    return fails ? 1 : 0;
}
