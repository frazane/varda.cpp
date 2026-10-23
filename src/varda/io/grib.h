// Thin RAII layer over the ecCodes C API: the only place that includes eccodes.h.
// Shared by the input (docs/spec-io.md §1-3) and output (§4) sides; owned by the integrator.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
    // A handle on a message inside memory that `owner` keeps alive (released after the handle).
    GribHandle(grib_handle* h, std::shared_ptr<const void> owner);
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

    // Key introspection (comparison tools): every key name, duplicates skipped, in ecCodes'
    // iteration order; a key's native type.
    enum class KeyType { Long, Double, String, Other };
    std::vector<std::string> key_names() const;
    KeyType native_type(const std::string& key) const;
    std::vector<long> get_long_array(const std::string& key) const;

    // Decode as anemoi does: codes_get_float_array / codes_get_double_array (docs/spec-io.md §2.0).
    std::vector<float> get_float_array(const std::string& key = "values") const;
    std::vector<double> get_double_array(const std::string& key = "values") const;
    // The same, into a caller's buffer of exactly size(key) values (throws otherwise).
    void get_float_array(const std::string& key, float* dst, size_t n) const;
    void get_double_array(const std::string& key, double* dst, size_t n) const;

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
    std::shared_ptr<const void> owner_;  // the message bytes, when the handle does not own a copy
};

// Every message of a GRIB file, in file order. Throws if the file can't be opened or a
// message fails to parse.
std::vector<GribHandle> read_messages(const std::string& path);
// The first message of a file (anemoi's template rule, docs/spec-io.md §4.3).
GribHandle read_first_message(const std::string& path);
// Every message of a GRIB file held in memory (the bytes of the file `what`), as read_messages
// gives them: the same messages, so the same keys and decoded values. A file that is a plain
// sequence of GRIB2 messages (each "GRIB" ... "7777" of its section-0 length, nothing between or
// after them) is split here and each message is wrapped without copying, its bytes kept alive by
// `owner` (null: each message is copied); anything else goes through ecCodes' own file reader
// (on a memory FILE), with read_messages' errors.
std::vector<GribHandle> read_messages_from_memory(const char* data, size_t size, const std::string& what,
                                                  std::shared_ptr<const void> owner = nullptr);

// Reads [offset, offset + size) of a file into `dst` with up to `threads` parallel preads of
// `chunk_bytes` each (a parallel file system serves concurrent requests far faster than one
// sequential stream). size == UINT64_MAX: to the end of the file; returns the bytes read.
uint64_t read_file_range(const std::string& path, uint64_t offset, uint64_t size, char* dst, int threads = 8,
                         size_t chunk_bytes = size_t(8) << 20);
uint64_t file_size(const std::string& path);  // throws "cannot open <path>"
// Reads the given byte ranges of a file once (in parallel) so that the page cache holds them, e.g.
// before a memory map is touched: page faults on a cold mapping fetch it one small serial request
// at a time. Contents are not changed. Empty `ranges`: the whole file.
void warm_file_cache(const std::string& path, const std::vector<std::pair<uint64_t, uint64_t>>& ranges = {},
                     int threads = 8);

// Reads and parses GRIB files ahead of their use: a pool of reader threads reads each file whole
// with parallel chunked preads (several files at once), and one parser thread splits the files
// that have been read into handles (read_messages_from_memory), the earliest listed first: ecCodes
// serialises handle creation internally, so more parsers only contend. Files are started in the
// order given; take() blocks until a file is ready. The handles are those read_messages gives.
class GribPrefetch {
public:
    struct Options {
        int threads = 16;                       // reader threads (I/O-bound; parsing is light)
        size_t chunk_bytes = size_t(32) << 20;  // largest pread; a file is split into >= ~8 of them (>= 1 MiB each)
    };
    explicit GribPrefetch(const std::vector<std::string>& paths) : GribPrefetch(paths, Options()) {}
    GribPrefetch(const std::vector<std::string>& paths, Options opts);
    ~GribPrefetch();  // stops the readers (files not started are skipped) and joins them
    GribPrefetch(const GribPrefetch&) = delete;
    GribPrefetch& operator=(const GribPrefetch&) = delete;

    // The messages of `path`, as read_messages(path) gives them (and its errors, rethrown). A path
    // listed k times can be taken k times; the pool drops its reference after the last take. A path
    // not listed is read by the caller with read_messages.
    std::shared_ptr<const std::vector<GribHandle>> take(const std::string& path);
    // Seconds the takers spent blocked on unfinished files (an I/O-bound pipeline's wait).
    double wait_seconds() const;
    // One line per file: size, messages, and when (seconds since construction) it was opened,
    // read, parsed and first taken. For profiling.
    std::vector<std::string> trace() const;

private:
    struct File;
    struct Task {
        size_t file;
        int64_t chunk;  // -1: open the file and split it into chunks
    };
    void worker();
    void parser();
    void run(const Task& t);
    void finish_chunk(size_t f);  // the last chunk of a file hands it to the parser

    Options opts_;
    std::vector<std::unique_ptr<File>> files_;
    mutable std::mutex mu_;
    std::condition_variable cv_work_, cv_done_, cv_parse_;
    std::deque<Task> queue_;
    int opening_ = 0;  // open tasks in progress (they may add chunk tasks)
    bool stop_ = false;
    size_t n_finished_ = 0;  // files parsed or failed
    double wait_s_ = 0;
    double t0_ = 0;
    std::vector<std::thread> threads_;
    std::thread parser_;
};

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
