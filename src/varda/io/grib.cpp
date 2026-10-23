#include "varda/io/grib.h"

#include <eccodes.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
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
GribHandle::GribHandle(grib_handle* h, std::shared_ptr<const void> owner) : h_(h), owner_(std::move(owner)) {}
GribHandle::GribHandle(GribHandle&& o) noexcept : h_(o.h_), owner_(std::move(o.owner_)) { o.h_ = nullptr; }
GribHandle& GribHandle::operator=(GribHandle&& o) noexcept {
    if (this != &o) {
        if (h_) codes_handle_delete(h_);  // before its bytes (owner_) may go
        h_ = o.h_;
        o.h_ = nullptr;
        owner_ = std::move(o.owner_);
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

void GribHandle::get_float_array(const std::string& key, float* dst, size_t n) const {
    if (size(key) != n) throw std::runtime_error("get_float_array " + key + ": " + std::to_string(size(key)) +
                                                 " values, buffer " + std::to_string(n));
    size_t len = n;
    check(codes_get_float_array(h_, key.c_str(), dst, &len), "get_float_array " + key);
    if (len != n) throw std::runtime_error("get_float_array " + key + ": decoded " + std::to_string(len) + " values");
}

void GribHandle::get_double_array(const std::string& key, double* dst, size_t n) const {
    if (size(key) != n) throw std::runtime_error("get_double_array " + key + ": " + std::to_string(size(key)) +
                                                 " values, buffer " + std::to_string(n));
    size_t len = n;
    check(codes_get_double_array(h_, key.c_str(), dst, &len), "get_double_array " + key);
    if (len != n) throw std::runtime_error("get_double_array " + key + ": decoded " + std::to_string(len) + " values");
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

namespace {

// The loop of read_messages on an open stream (closed here).
std::vector<GribHandle> read_stream(FILE* f, const std::string& path) {
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

}  // namespace

std::vector<GribHandle> read_messages(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    return read_stream(f, path);
}

namespace {

// The (offset, length) of each message when `d` is exactly a sequence of GRIB2 messages; empty
// otherwise (then ecCodes' reader decides, as for a file).
std::vector<std::pair<size_t, size_t>> split_grib2(const char* d, size_t n) {
    std::vector<std::pair<size_t, size_t>> out;
    const auto* p = reinterpret_cast<const unsigned char*>(d);
    size_t o = 0;
    while (o < n) {
        if (n - o < 16 || std::memcmp(p + o, "GRIB", 4) != 0 || p[o + 7] != 2) return {};
        uint64_t len = 0;
        for (int i = 8; i < 16; ++i) len = (len << 8) | p[o + size_t(i)];
        if (len < 20 || len > n - o || std::memcmp(p + o + len - 4, "7777", 4) != 0) return {};
        out.push_back({o, size_t(len)});
        o += size_t(len);
    }
    return out;
}

}  // namespace

std::vector<GribHandle> read_messages_from_memory(const char* data, size_t size, const std::string& what,
                                                  std::shared_ptr<const void> owner) {
    if (size == 0) return {};
    const std::vector<std::pair<size_t, size_t>> msgs = split_grib2(data, size);
    if (!msgs.empty()) {
        std::vector<GribHandle> out;
        out.reserve(msgs.size());
        bool ok = true;
        for (const auto& [o, len] : msgs) {
            codes_handle* h = owner ? codes_handle_new_from_message(nullptr, data + o, len)
                                    : codes_handle_new_from_message_copy(nullptr, data + o, len);
            if (!h) {  // ecCodes' file reader reports it as read_messages does
                ok = false;
                break;
            }
            out.push_back(owner ? GribHandle(h, owner) : GribHandle(h));
        }
        if (ok) return out;
    }
    FILE* f = ::fmemopen(const_cast<char*>(data), size, "rb");
    if (!f) throw std::runtime_error("fmemopen failed for " + what + ": " + std::strerror(errno));
    return read_stream(f, what);
}

// --------------------------------------------------------------------------------------- parallel reads

namespace {

struct Fd {
    int fd = -1;
    explicit Fd(const std::string& path) : fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (fd < 0) throw std::runtime_error("cannot open " + path);
    }
    ~Fd() {
        if (fd >= 0) ::close(fd);
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    uint64_t size(const std::string& path) const {
        struct stat st {};
        if (::fstat(fd, &st) != 0) throw std::runtime_error("cannot stat " + path);
        return uint64_t(st.st_size);
    }
};

// Reads exactly n bytes at off (dst null: into a scratch buffer, to fill the page cache only).
void pread_all(int fd, uint64_t off, uint64_t n, char* dst, std::vector<char>& scratch, const std::string& path) {
    while (n > 0) {
        char* p = dst;
        size_t want = size_t(std::min<uint64_t>(n, uint64_t(1) << 30));
        if (!p) {
            want = std::min(want, size_t(8) << 20);
            if (scratch.size() < want) scratch.resize(want);
            p = scratch.data();
        }
        const ssize_t r = ::pread(fd, p, want, off_t(off));
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("read failed: " + path + ": " + std::strerror(errno));
        }
        if (r == 0) throw std::runtime_error("read failed: " + path + ": unexpected end of file");
        off += uint64_t(r);
        n -= uint64_t(r);
        if (dst) dst += r;
    }
}

struct Range {
    uint64_t off, len;
    char* dst;
};

// The ranges split into chunks, read by up to `threads` threads (the caller is one of them).
void parallel_pread(int fd, const std::vector<Range>& ranges, int threads, size_t chunk, const std::string& path) {
    std::vector<Range> parts;
    for (const Range& r : ranges)
        for (uint64_t o = 0; o < r.len; o += chunk)
            parts.push_back({r.off + o, std::min<uint64_t>(chunk, r.len - o), r.dst ? r.dst + o : nullptr});
    std::atomic<size_t> next{0};
    std::mutex mu;
    std::string err;
    auto work = [&] {
        std::vector<char> scratch;
        for (size_t i; (i = next.fetch_add(1)) < parts.size();) {
            try {
                pread_all(fd, parts[i].off, parts[i].len, parts[i].dst, scratch, path);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(mu);
                if (err.empty()) err = e.what();
                next = parts.size();
            }
        }
    };
    const size_t n_threads = std::min(parts.size(), size_t(std::max(threads, 1)));
    std::vector<std::thread> pool;
    for (size_t t = 1; t < n_threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (!err.empty()) throw std::runtime_error(err);
}

}  // namespace

uint64_t file_size(const std::string& path) {
    Fd fd(path);
    return fd.size(path);
}

uint64_t read_file_range(const std::string& path, uint64_t offset, uint64_t size, char* dst, int threads,
                         size_t chunk_bytes) {
    Fd fd(path);
    if (size == UINT64_MAX) {
        const uint64_t n = fd.size(path);
        size = n > offset ? n - offset : 0;
    }
    parallel_pread(fd.fd, {{offset, size, dst}}, threads, std::max<size_t>(chunk_bytes, 1), path);
    return size;
}

void warm_file_cache(const std::string& path, const std::vector<std::pair<uint64_t, uint64_t>>& ranges, int threads) {
    Fd fd(path);
    const uint64_t n = fd.size(path);
    std::vector<Range> rs;
    if (ranges.empty()) rs.push_back({0, n, nullptr});
    for (const auto& [o, l] : ranges)
        if (o < n) rs.push_back({o, std::min(l, n - o), nullptr});
    parallel_pread(fd.fd, rs, threads, size_t(4) << 20, path);
}

// -------------------------------------------------------------------------------------------- prefetch

struct GribPrefetch::File {
    std::string path;
    int uses = 0, taken = 0;
    int fd = -1;
    uint64_t size = 0, chunk = 0;
    std::shared_ptr<char[]> buf;  // shared with the handles (they wrap the messages in place)
    int64_t chunks_left = 0;
    bool read = false;  // read whole, waiting for the parser
    bool done = false;
    std::exception_ptr err;
    std::shared_ptr<const std::vector<GribHandle>> msgs;
    double t_open = -1, t_read = -1, t_parsed = -1, t_take = -1;
    size_t n_msgs = 0;
    ~File() {
        if (fd >= 0) ::close(fd);
    }
};

namespace {
double mono_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

GribPrefetch::GribPrefetch(const std::vector<std::string>& paths, Options opts) : opts_(opts), t0_(mono_s()) {
    if (opts_.chunk_bytes == 0) opts_.chunk_bytes = size_t(32) << 20;
    std::map<std::string, size_t> index;
    for (const auto& p : paths) {
        auto it = index.find(p);
        if (it == index.end()) {
            it = index.emplace(p, files_.size()).first;
            files_.push_back(std::make_unique<File>());
            files_.back()->path = p;
            queue_.push_back({files_.size() - 1, -1});
        }
        ++files_[it->second]->uses;
    }
    const size_t n = std::min(files_.size(), size_t(std::max(opts_.threads, 1)));
    for (size_t t = 0; t < n; ++t) threads_.emplace_back([this] { worker(); });
    if (!files_.empty()) parser_ = std::thread([this] { parser(); });
}

GribPrefetch::~GribPrefetch() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        queue_.clear();
    }
    cv_work_.notify_all();
    cv_parse_.notify_all();
    for (auto& t : threads_) t.join();
    if (parser_.joinable()) parser_.join();
}

void GribPrefetch::parser() {
    for (;;) {
        size_t i = files_.size();
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_parse_.wait(lk, [&] {
                if (stop_ || n_finished_ == files_.size()) return true;
                for (const auto& f : files_)
                    if (f->read) return true;
                return false;
            });
            if (stop_ || n_finished_ == files_.size()) return;
            for (i = 0; !files_[i]->read; ++i) {
            }
            files_[i]->read = false;
        }
        File& f = *files_[i];
        std::shared_ptr<const std::vector<GribHandle>> msgs;
        std::exception_ptr err;
        try {
            msgs = std::make_shared<const std::vector<GribHandle>>(
                read_messages_from_memory(f.buf.get(), f.size, f.path, std::shared_ptr<const void>(f.buf, f.buf.get())));
        } catch (...) {
            err = std::current_exception();
        }
        f.buf.reset();
        f.t_parsed = mono_s() - t0_;
        {
            std::lock_guard<std::mutex> lk(mu_);
            f.n_msgs = msgs ? msgs->size() : 0;
            f.msgs = std::move(msgs);
            f.err = err;
            f.done = true;
            ++n_finished_;
        }
        cv_done_.notify_all();
    }
}

void GribPrefetch::worker() {
    for (;;) {
        Task t{};
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stop_ || !queue_.empty() || opening_ == 0; });
            if (stop_ || queue_.empty()) return;  // nothing queued and no open in progress: done
            t = queue_.front();
            queue_.pop_front();
            if (t.chunk < 0) ++opening_;
        }
        run(t);
    }
}

void GribPrefetch::run(const Task& t) {
    File& f = *files_[t.file];
    if (t.chunk < 0) {
        int64_t n = 0;
        f.t_open = mono_s() - t0_;
        try {
            f.fd = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC);
            if (f.fd < 0) throw std::runtime_error("cannot open " + f.path);
            struct stat st {};
            if (::fstat(f.fd, &st) != 0) throw std::runtime_error("cannot stat " + f.path);
            f.size = uint64_t(st.st_size);
            f.buf.reset(new char[std::max<uint64_t>(f.size, 1)]);
            // At least ~8 requests in flight per file (of 1 MiB or more): a single object storage
            // target can be slow per request, and parallel requests hide that.
            f.chunk = std::max<uint64_t>(std::min<uint64_t>(opts_.chunk_bytes, (f.size + 7) / 8), uint64_t(1) << 20);
            n = int64_t((f.size + f.chunk - 1) / f.chunk);
        } catch (...) {
            f.err = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            --opening_;
            f.chunks_left = std::max<int64_t>(n, 1);
            if (!stop_)  // the other chunks go first, so idle readers help finish the files in flight
                for (int64_t c = n - 1; c >= 1; --c) queue_.push_front({t.file, c});
        }
        cv_work_.notify_all();
        if (f.err || n == 0) {
            finish_chunk(t.file);
            return;
        }
        run({t.file, 0});
        return;
    }
    try {
        const uint64_t off = uint64_t(t.chunk) * f.chunk;
        std::vector<char> scratch;
        pread_all(f.fd, off, std::min<uint64_t>(f.chunk, f.size - off), f.buf.get() + off, scratch, f.path);
    } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!f.err) f.err = std::current_exception();
    }
    finish_chunk(t.file);
}

void GribPrefetch::finish_chunk(size_t i) {
    File& f = *files_[i];
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (--f.chunks_left > 0) return;
    }
    if (f.fd >= 0) {
        ::close(f.fd);
        f.fd = -1;
    }
    f.t_read = mono_s() - t0_;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (f.err) {  // nothing to parse: the error goes to the takers
            f.buf.reset();
            f.done = true;
            ++n_finished_;
        } else {
            f.read = true;
        }
    }
    cv_done_.notify_all();
    cv_parse_.notify_all();
}

std::shared_ptr<const std::vector<GribHandle>> GribPrefetch::take(const std::string& path) {
    File* f = nullptr;
    for (const auto& p : files_)
        if (p->path == path) f = p.get();
    std::unique_lock<std::mutex> lk(mu_);
    if (!f || (f->done && !f->err && !f->msgs)) {  // not listed, or taken more often than listed
        lk.unlock();
        return std::make_shared<const std::vector<GribHandle>>(read_messages(path));
    }
    if (f->t_take < 0) f->t_take = mono_s() - t0_;
    if (!f->done) {
        const auto t0 = std::chrono::steady_clock::now();
        cv_done_.wait(lk, [&] { return f->done; });
        wait_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    if (f->err) std::rethrow_exception(f->err);
    std::shared_ptr<const std::vector<GribHandle>> r = f->msgs;
    if (++f->taken >= f->uses) f->msgs.reset();
    return r;
}

std::vector<std::string> GribPrefetch::trace() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (const auto& f : files_) {
        char b[160];
        std::snprintf(b, sizeof b, "%8.1f MB %4zu msgs: open %6.3f read %6.3f parsed %6.3f taken %6.3f  ",
                      double(f->size) / 1e6, f->n_msgs, f->t_open, f->t_read, f->t_parsed, f->t_take);
        out.push_back(b + f->path);
    }
    return out;
}

double GribPrefetch::wait_seconds() const {
    std::lock_guard<std::mutex> lk(mu_);
    return wait_s_;
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
