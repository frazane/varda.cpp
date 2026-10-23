// varda-grib-compare: compare two GRIB files, or two directories of GRIB files, message by message.
//
//   varda-grib-compare TEST REF [--rel-tol X] [--abs-tol Y] [--require-bytes] [--ignore-key K[,K...]]
//                               [--ignore-data-keys]
//                               [--max-report N] [-q]
//
// Per message, in file order: the name (shortName / paramId / typeOfLevel / level / step), every
// header key (like `grib_compare -H`, including md5Headers; arrays up to 10000 elements; larger ones
// are values or geometry and are skipped, their count is reported), the bitmap (missing positions),
// the values (count differing, bitwise yes/no, max_abs, rel_absmax = max_abs / max|REF| over the
// points present in both; docs/format.md §2) and whether the message bytes are equal.
// Exit status 0 only if: the same files in the same order, the same number of messages, equal names,
// no key differs (except --ignore-key), bitmaps equal, max_abs <= --abs-tol and rel_absmax <=
// --rel-tol (default 0: values must be bitwise), and with --require-bytes the bytes too.
// --ignore-data-keys ignores the keys that depend on the values or on the packing (statistics, reference value,
// binary scale factor, bits per value, lengths, md5Headers): for end-to-end comparisons that are not bitwise.
// The COSMO definitions are taken from $VARDA_COSMO_DEFINITIONS (needed to read ICON messages).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "varda/io/grib.h"

using namespace varda::io;
namespace fs = std::filesystem;

struct Options {
    double rel_tol = 0, abs_tol = 0;
    bool require_bytes = false, quiet = false;
    std::set<std::string> ignore;
    int max_report = 40;  // total number of detail lines printed
};

struct Totals {
    int files = 0, messages = 0, bytes_equal = 0, values_bitwise = 0, keys_mismatch_messages = 0, bad = 0;
    long key_mismatches = 0;
    double worst_abs = 0, worst_rel = 0;
    long skipped_big_keys = 0;
    int reported = 0;
};

static std::string label(const GribHandle& h) {
    auto g = [&](const char* k) {
        try { return h.get_string(k); } catch (...) { return std::string("?"); }
    };
    return g("shortName") + "/" + g("paramId") + "/" + g("typeOfLevel") + "/L" + g("level") + "/step" + g("step");
}

using KeyType = GribHandle::KeyType;

struct KeyValue {
    KeyType type = KeyType::Other;
    bool missing = false;
    std::string s;
    std::vector<long> l;
    std::vector<double> d;
};

static const size_t kMaxArray = 10000;

static bool read_key(const GribHandle& h, const std::string& name, KeyValue& kv, bool& too_big) {
    too_big = false;
    try {
        const KeyType type = h.native_type(name);
        if (h.size(name) > kMaxArray) { too_big = true; return false; }
        if (type == KeyType::Other) return false;  // bytes, sections, labels: covered by md5Headers
        kv.type = type;
        kv.missing = h.is_missing(name);
        if (type == KeyType::Long) kv.l = h.get_long_array(name);
        else if (type == KeyType::Double) kv.d = h.get_double_array(name);
        else kv.s = h.get_string(name);
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

static bool same_key(const KeyValue& a, const KeyValue& b) {
    if (a.type != b.type || a.missing != b.missing) return false;
    if (a.type == KeyType::Long) return a.l == b.l;
    if (a.type == KeyType::String) return a.s == b.s;
    if (a.d.size() != b.d.size()) return false;
    return a.d.empty() || std::memcmp(a.d.data(), b.d.data(), a.d.size() * sizeof(double)) == 0;
}

static std::string show(const KeyValue& k) {
    if (k.missing) return "MISSING";
    if (k.type == KeyType::String) return "'" + k.s + "'";
    std::string s;
    auto add = [&](const std::string& x) { if (s.size() < 60) s += (s.empty() ? "" : ",") + x; };
    if (k.type == KeyType::Long) for (long v : k.l) add(std::to_string(v));
    else for (double v : k.d) { char b[40]; std::snprintf(b, sizeof b, "%.17g", v); add(b); }
    return k.l.size() > 1 || k.d.size() > 1 ? "[" + s + "]" : s;
}

static std::map<std::string, KeyValue> read_all_keys(const GribHandle& h, long& skipped) {
    std::map<std::string, KeyValue> out;
    static const std::set<std::string> never = {"values", "codedValues", "packedValues", "bitmap"};
    for (const std::string& name : h.key_names()) {
        if (never.count(name)) continue;
        KeyValue kv;
        bool big = false;
        if (read_key(h, name, kv, big)) out.emplace(name, std::move(kv));
        else if (big) ++skipped;
    }
    // explicit, as grib_compare -H does
    for (const char* name : {"md5Headers"}) {
        KeyValue kv;
        bool big;
        if (read_key(h, name, kv, big)) out[name] = kv;
    }
    return out;
}

// Compare one pair of messages; returns true if it passes the criteria.
static bool compare_messages(GribHandle& a, GribHandle& b, int idx, const std::string& file, const Options& o, Totals& t) {
    bool ok = true;
    auto report = [&](const std::string& what) {
        if (o.quiet) return;
        if (t.reported < o.max_report) std::printf("  %s msg %d: %s\n", file.c_str(), idx, what.c_str());
        else if (t.reported == o.max_report) std::printf("  ... further differences not listed (--max-report)\n");
        ++t.reported;
    };
    const std::string la = label(a), lb = label(b);
    if (la != lb) { report("name differs: " + la + " vs " + lb); ok = false; }

    auto [pa, na] = a.message();
    auto [pb, nb] = b.message();
    const bool bytes = na == nb && std::memcmp(pa, pb, na) == 0;
    t.bytes_equal += bytes;
    if (!bytes) {
        size_t first = 0;
        const size_t m = std::min(na, nb);
        while (first < m && static_cast<const unsigned char*>(pa)[first] == static_cast<const unsigned char*>(pb)[first]) ++first;
        report("bytes differ (lengths " + std::to_string(na) + " vs " + std::to_string(nb) + ", first difference at byte " + std::to_string(first) + ")");
        if (o.require_bytes) ok = false;
    }

    // header keys
    long skipped = 0;
    auto ka = read_all_keys(a, skipped), kb = read_all_keys(b, skipped);
    t.skipped_big_keys += skipped;
    int nk = 0;
    for (const auto& [k, va] : ka) {
        if (o.ignore.count(k)) continue;
        auto it = kb.find(k);
        if (it == kb.end()) { report("key " + k + " only in TEST"); ++nk; continue; }
        if (!same_key(va, it->second)) { report("key " + k + ": TEST " + show(va) + " REF " + show(it->second)); ++nk; }
    }
    for (const auto& [k, vb] : kb)
        if (!o.ignore.count(k) && !ka.count(k)) { report("key " + k + " only in REF"); ++nk; }
    if (nk) { ok = false; t.key_mismatches += nk; ++t.keys_mismatch_messages; }

    // bitmap and values: missing values decoded as a value no field can have
    constexpr double kMiss = 1.0e300;
    a.set_double("missingValue", kMiss);
    b.set_double("missingValue", kMiss);
    const std::vector<double> va = a.get_double_array(), vb = b.get_double_array();
    if (va.size() != vb.size()) {
        report("values: sizes differ " + std::to_string(va.size()) + " vs " + std::to_string(vb.size()));
        return false;
    }
    long miss_diff = 0, ndiff = 0, nvals = 0;
    double max_abs = 0, absmax_ref = 0;
    for (size_t i = 0; i < va.size(); ++i) {
        const bool ma = va[i] == kMiss, mb = vb[i] == kMiss;
        if (ma != mb) { ++miss_diff; continue; }
        if (ma) continue;
        ++nvals;
        if (std::memcmp(&va[i], &vb[i], sizeof(double)) != 0) {
            // NaN cannot occur in GRIB values; -0.0 vs 0.0 counts as different here (bitwise)
            ++ndiff;
        }
        max_abs = std::max(max_abs, std::fabs(va[i] - vb[i]));
        absmax_ref = std::max(absmax_ref, std::fabs(vb[i]));
    }
    const double rel = max_abs == 0 ? 0.0 : (absmax_ref == 0 ? INFINITY : max_abs / absmax_ref);
    t.values_bitwise += (ndiff == 0 && miss_diff == 0);
    t.worst_abs = std::max(t.worst_abs, max_abs);
    t.worst_rel = std::max(t.worst_rel, rel);
    if (miss_diff) { report("bitmap: " + std::to_string(miss_diff) + " positions missing in only one"); ok = false; }
    if (ndiff || max_abs > 0) {
        char b[200];
        std::snprintf(b, sizeof b, "values: %ld of %ld differ, max_abs %.6g, rel_absmax %.6g", ndiff, nvals, max_abs, rel);
        report(b);
    }
    if (max_abs > o.abs_tol || rel > o.rel_tol) ok = false;
    return ok;
}

static bool compare_files(const std::string& fa, const std::string& fb, const std::string& name, const Options& o, Totals& t) {
    auto ma = read_messages(fa), mb = read_messages(fb);
    ++t.files;
    bool ok = true;
    if (ma.size() != mb.size()) {
        std::printf("  %s: message counts differ: %zu vs %zu\n", name.c_str(), ma.size(), mb.size());
        ok = false;
    }
    const size_t n = std::min(ma.size(), mb.size());
    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        ++t.messages;
        if (!compare_messages(ma[i], mb[i], int(i), name, o, t)) { ++bad; ok = false; }
    }
    t.bad += bad;
    std::printf("%s %s: %zu messages, %d failing\n", ok ? "ok  " : "FAIL", name.c_str(), n, bad);
    return ok;
}

static std::vector<std::string> grib_files(const fs::path& dir) {
    std::vector<std::string> names;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".grib") names.push_back(fs::relative(e.path(), dir).string());
    std::sort(names.begin(), names.end());
    return names;
}

int main(int argc, char** argv) {
    Options o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--rel-tol") o.rel_tol = std::atof(next().c_str());
        else if (a == "--abs-tol") o.abs_tol = std::atof(next().c_str());
        else if (a == "--require-bytes") o.require_bytes = true;
        else if (a == "--ignore-data-keys")
            for (const char* k : {"average", "kurtosis", "skewness", "standardDeviation", "min", "max", "minimum", "maximum",
                                  "referenceValue", "referenceValueError", "binaryScaleFactor", "bitsPerValue", "isConstant",
                                  "totalLength", "section7Length", "md5Headers"})
                o.ignore.insert(k);
        else if (a == "-q") o.quiet = true;
        else if (a == "--max-report") o.max_report = std::atoi(next().c_str());
        else if (a == "--ignore-key") {
            std::string v = next();
            for (size_t p = 0; p <= v.size();) {
                size_t e = v.find(',', p);
                if (e == std::string::npos) e = v.size();
                if (e > p) o.ignore.insert(v.substr(p, e - p));
                p = e + 1;
            }
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else pos.push_back(a);
    }
    if (pos.size() != 2) {
        std::fprintf(stderr, "usage: %s TEST REF [--rel-tol X] [--abs-tol Y] [--require-bytes] [--ignore-key K,...] [--ignore-data-keys] [--max-report N] [-q]\n", argv[0]);
        return 2;
    }
    try {
        if (const char* d = std::getenv("VARDA_COSMO_DEFINITIONS")) set_definitions_path(d);
        Totals t;
        bool ok = true;
        if (fs::is_directory(pos[0]) != fs::is_directory(pos[1])) {
            std::fprintf(stderr, "TEST and REF must both be files or both be directories\n");
            return 2;
        }
        if (fs::is_directory(pos[0])) {
            const auto na = grib_files(pos[0]), nb = grib_files(pos[1]);
            if (na != nb) {
                ok = false;
                for (const auto& n : na) if (!std::binary_search(nb.begin(), nb.end(), n)) std::printf("FAIL file only in TEST: %s\n", n.c_str());
                for (const auto& n : nb) if (!std::binary_search(na.begin(), na.end(), n)) std::printf("FAIL file only in REF: %s\n", n.c_str());
            }
            for (const auto& n : na)
                if (std::binary_search(nb.begin(), nb.end(), n))
                    ok = compare_files((fs::path(pos[0]) / n).string(), (fs::path(pos[1]) / n).string(), n, o, t) && ok;
        } else {
            ok = compare_files(pos[0], pos[1], fs::path(pos[0]).filename().string(), o, t);
        }
        std::printf("summary: %d files, %d messages, %d failing; bytes equal %d, values bitwise %d, messages with key mismatches %d (%ld keys); "
                    "worst max_abs %.6g, worst rel_absmax %.6g; %ld big arrays skipped; tolerances rel %.3g abs %.3g%s -> %s\n",
                    t.files, t.messages, t.bad, t.bytes_equal, t.values_bitwise, t.keys_mismatch_messages, t.key_mismatches,
                    t.worst_abs, t.worst_rel, t.skipped_big_keys, o.rel_tol, o.abs_tol, o.require_bytes ? ", bytes required" : "",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "varda-grib-compare: %s\n", e.what());
        return 2;
    }
}
