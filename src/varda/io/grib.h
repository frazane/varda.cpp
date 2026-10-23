// Thin RAII layer over the ecCodes C API: the only place that includes eccodes.h.
// Shared by the input (docs/spec-io.md §1-3) and output (§4) sides; owned by the integrator.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct grib_handle;

namespace varda::io {

// Puts `cosmo_definitions` in front of ecCodes' default definitions path, as importing
// anemoi_plugins_meteoswiss does (docs/spec-io.md §1.2). Must run before the first handle is
// created; calling it again with the same path is a no-op, a different path throws.
void set_definitions_path(const std::string& cosmo_definitions);
// The path in effect (for logs and manifests).
std::string definitions_path();
long eccodes_version();  // e.g. 24703 for 2.47.3

// "key=value" pairs for GribHandle::set_key_vals, values already spelled as Python's str()
// spells them (anemoi-inference passes a dict through eccodes' grib_set_key_vals).
using KeyVals = std::vector<std::pair<std::string, std::string>>;

class GribHandle {
public:
    GribHandle() = default;
    explicit GribHandle(grib_handle* h);  // takes ownership
    GribHandle(GribHandle&&) noexcept;
    GribHandle& operator=(GribHandle&&) noexcept;
    GribHandle(const GribHandle&) = delete;
    GribHandle& operator=(const GribHandle&) = delete;
    ~GribHandle();

    explicit operator bool() const { return h_ != nullptr; }
    grib_handle* raw() const { return h_; }
    GribHandle clone() const;
    // A handle on a copy of the message bytes.
    static GribHandle from_message(const void* data, size_t size);

    bool has(const std::string& key) const;
    bool is_missing(const std::string& key) const;
    long get_long(const std::string& key) const;
    double get_double(const std::string& key) const;
    std::string get_string(const std::string& key) const;
    size_t size(const std::string& key = "values") const;

    // Decode as anemoi does: codes_get_float_array / codes_get_double_array (docs/spec-io.md §2.0).
    std::vector<float> get_float_array(const std::string& key = "values") const;
    std::vector<double> get_double_array(const std::string& key = "values") const;

    void set_long(const std::string& key, long v);
    void set_double(const std::string& key, double v);
    void set_string(const std::string& key, const std::string& v);
    // eccodes-python's grib_set_key_vals: joins "k=v" with ",", types inferred from the text
    // by ecCodes' parse_keyval_string, then one grib_set_values call (docs/spec-io.md §4.3).
    void set_key_vals(const KeyVals& kvs);
    void set_double_array(const std::string& key, const std::vector<double>& v);

    // The encoded message (valid until the handle changes).
    std::pair<const void*, size_t> message() const;

private:
    grib_handle* h_ = nullptr;
};

// Every message of a GRIB file, in file order. Throws if the file can't be opened or a
// message fails to parse.
std::vector<GribHandle> read_messages(const std::string& path);
// The first message of a file (anemoi's template rule, docs/spec-io.md §4.3).
GribHandle read_first_message(const std::string& path);

// Writes messages to a new file (opened "wb" once), flushing after each message, as
// anemoi-inference does. Written under `path + ".part"`, renamed to `path` by close().
class GribFileWriter {
public:
    explicit GribFileWriter(const std::string& path);
    ~GribFileWriter();  // without close(): removes the partial file
    GribFileWriter(const GribFileWriter&) = delete;
    GribFileWriter& operator=(const GribFileWriter&) = delete;
    void write(const GribHandle& h);
    void close();
    size_t count() const { return count_; }

private:
    std::string path_;
    void* f_ = nullptr;  // FILE*
    size_t count_ = 0;
};

}  // namespace varda::io
