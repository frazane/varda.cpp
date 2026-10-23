#include "varda/io/grib.h"

#include <eccodes.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace varda::io {

namespace {

[[noreturn]] void fail(const std::string& what, int err) {
    throw std::runtime_error("ecCodes: " + what + ": " + codes_get_error_message(err));
}

void check(int err, const std::string& what) {
    if (err != CODES_SUCCESS) fail(what, err);
}

std::mutex g_defs_mutex;
std::string g_cosmo;  // the path set_definitions_path installed, empty before

}  // namespace

void set_definitions_path(const std::string& cosmo_definitions) {
    std::lock_guard<std::mutex> lock(g_defs_mutex);
    if (!g_cosmo.empty()) {
        if (g_cosmo != cosmo_definitions)
            throw std::runtime_error("ecCodes definitions already set to " + g_cosmo);
        return;
    }
    const char* prev = codes_definition_path(nullptr);
    const std::string path = cosmo_definitions + (prev && *prev ? ":" + std::string(prev) : "");
    codes_context_set_definitions_path(nullptr, path.c_str());
    g_cosmo = cosmo_definitions;
}

std::string definitions_path() {
    const char* p = codes_definition_path(nullptr);
    return p ? p : "";
}

long eccodes_version() { return codes_get_api_version(); }

GribHandle::GribHandle(grib_handle* h) : h_(h) {}
GribHandle::GribHandle(GribHandle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
GribHandle& GribHandle::operator=(GribHandle&& o) noexcept {
    if (this != &o) {
        if (h_) codes_handle_delete(h_);
        h_ = o.h_;
        o.h_ = nullptr;
    }
    return *this;
}
GribHandle::~GribHandle() {
    if (h_) codes_handle_delete(h_);
}

GribHandle GribHandle::clone() const {
    codes_handle* c = codes_handle_clone(h_);
    if (!c) throw std::runtime_error("ecCodes: codes_handle_clone failed");
    return GribHandle(c);
}

GribHandle GribHandle::from_message(const void* data, size_t size) {
    codes_handle* h = codes_handle_new_from_message_copy(nullptr, data, size);
    if (!h) throw std::runtime_error("ecCodes: cannot parse message");
    return GribHandle(h);
}

bool GribHandle::has(const std::string& key) const { return codes_is_defined(h_, key.c_str()) != 0; }

bool GribHandle::is_missing(const std::string& key) const {
    int err = 0;
    int m = codes_is_missing(h_, key.c_str(), &err);
    check(err, "is_missing " + key);
    return m != 0;
}

long GribHandle::get_long(const std::string& key) const {
    long v = 0;
    check(codes_get_long(h_, key.c_str(), &v), "get_long " + key);
    return v;
}

double GribHandle::get_double(const std::string& key) const {
    double v = 0;
    check(codes_get_double(h_, key.c_str(), &v), "get_double " + key);
    return v;
}

std::string GribHandle::get_string(const std::string& key) const {
    size_t len = 0;
    check(codes_get_length(h_, key.c_str(), &len), "get_length " + key);
    std::string s(len + 1, '\0');
    len = s.size();
    check(codes_get_string(h_, key.c_str(), s.data(), &len), "get_string " + key);
    s.resize(std::strlen(s.c_str()));
    return s;
}

size_t GribHandle::size(const std::string& key) const {
    size_t n = 0;
    check(codes_get_size(h_, key.c_str(), &n), "get_size " + key);
    return n;
}

std::vector<std::string> GribHandle::key_names() const {
    codes_keys_iterator* it = codes_keys_iterator_new(
        h_, CODES_KEYS_ITERATOR_ALL_KEYS | CODES_KEYS_ITERATOR_SKIP_DUPLICATES, nullptr);
    if (!it) throw std::runtime_error("ecCodes: cannot iterate keys");
    std::vector<std::string> out;
    while (codes_keys_iterator_next(it)) out.emplace_back(codes_keys_iterator_get_name(it));
    codes_keys_iterator_delete(it);
    return out;
}

GribHandle::KeyType GribHandle::native_type(const std::string& key) const {
    int t = 0;
    check(codes_get_native_type(h_, key.c_str(), &t), "get_native_type " + key);
    switch (t) {
        case CODES_TYPE_LONG: return KeyType::Long;
        case CODES_TYPE_DOUBLE: return KeyType::Double;
        case CODES_TYPE_STRING: return KeyType::String;
        default: return KeyType::Other;
    }
}

std::vector<long> GribHandle::get_long_array(const std::string& key) const {
    size_t n = size(key);
    std::vector<long> v(n);
    check(codes_get_long_array(h_, key.c_str(), v.data(), &n), "get_long_array " + key);
    v.resize(n);
    return v;
}

std::vector<float> GribHandle::get_float_array(const std::string& key) const {
    size_t n = size(key);
    std::vector<float> v(n);
    check(codes_get_float_array(h_, key.c_str(), v.data(), &n), "get_float_array " + key);
    v.resize(n);
    return v;
}

std::vector<double> GribHandle::get_double_array(const std::string& key) const {
    size_t n = size(key);
    std::vector<double> v(n);
    check(codes_get_double_array(h_, key.c_str(), v.data(), &n), "get_double_array " + key);
    v.resize(n);
    return v;
}

void GribHandle::set_long(const std::string& key, long v) {
    check(codes_set_long(h_, key.c_str(), v), "set_long " + key);
}
void GribHandle::set_double(const std::string& key, double v) {
    check(codes_set_double(h_, key.c_str(), v), "set_double " + key);
}
void GribHandle::set_string(const std::string& key, const std::string& v) {
    size_t len = v.size();
    check(codes_set_string(h_, key.c_str(), v.c_str(), &len), "set_string " + key);
}

void GribHandle::set_key_vals(const KeyVals& kvs) {
    if (kvs.empty()) throw std::runtime_error("set_key_vals: no keys");
    std::string s;
    for (const auto& [k, v] : kvs) {
        if (!s.empty()) s += ",";
        s += k + "=" + v;
    }
    // As gribapi.grib_set_key_vals: room for 1024 values, at most 1000 parsed.
    std::vector<grib_values> values(1024);
    std::memset(values.data(), 0, values.size() * sizeof(grib_values));
    int count = 1000;
    std::vector<char> buf(s.begin(), s.end());
    buf.push_back('\0');
    check(parse_keyval_string(nullptr, buf.data(), 1, GRIB_TYPE_UNDEFINED, values.data(), &count),
          "parse_keyval_string " + s);
    // grib_values point into buf (names and string values): buf outlives the call.
    check(grib_set_values(h_, values.data(), size_t(count)), "set_values " + s);
}

void GribHandle::set_double_array(const std::string& key, const std::vector<double>& v) {
    check(codes_set_double_array(h_, key.c_str(), v.data(), v.size()), "set_double_array " + key);
}

std::pair<const void*, size_t> GribHandle::message() const {
    const void* p = nullptr;
    size_t n = 0;
    check(codes_get_message(h_, &p, &n), "get_message");
    return {p, n};
}

std::vector<GribHandle> read_messages(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<GribHandle> out;
    try {
        for (;;) {
            int err = 0;
            codes_handle* h = codes_handle_new_from_file(nullptr, f, PRODUCT_GRIB, &err);
            if (!h) {
                if (err != CODES_SUCCESS) fail(path + ": message " + std::to_string(out.size()), err);
                break;
            }
            out.emplace_back(h);
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return out;
}

GribHandle read_first_message(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    int err = 0;
    codes_handle* h = codes_handle_new_from_file(nullptr, f, PRODUCT_GRIB, &err);
    std::fclose(f);
    if (!h) {
        if (err != CODES_SUCCESS) fail(path, err);
        throw std::runtime_error(path + ": no GRIB message");
    }
    return GribHandle(h);
}

GribFileWriter::GribFileWriter(const std::string& path) : path_(path) {
    f_ = std::fopen((path + ".part").c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot create " + path + ".part");
}

GribFileWriter::~GribFileWriter() {
    if (f_) {
        std::fclose(static_cast<FILE*>(f_));
        std::remove((path_ + ".part").c_str());
    }
}

void GribFileWriter::write(const GribHandle& h) {
    auto [p, n] = h.message();
    FILE* f = static_cast<FILE*>(f_);
    if (!f) throw std::runtime_error(path_ + ": writer is closed");
    if (std::fwrite(p, 1, n, f) != n || std::fflush(f) != 0)
        throw std::runtime_error("write failed: " + path_);
    ++count_;
}

void GribFileWriter::close() {
    FILE* f = static_cast<FILE*>(f_);
    if (!f) return;
    f_ = nullptr;
    if (std::fclose(f) != 0) throw std::runtime_error("close failed: " + path_);
    if (std::rename((path_ + ".part").c_str(), path_.c_str()) != 0)
        throw std::runtime_error("rename failed: " + path_);
}

}  // namespace varda::io
