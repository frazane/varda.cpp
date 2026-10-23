#include "varda/format.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "nlohmann/json.hpp"

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22  // Linux >= 5.14; older kernels reject it with EINVAL
#endif

namespace varda {

using json = nlohmann::json;

namespace {
struct DTypeInfo { DType t; const char* name; size_t size; };
constexpr DTypeInfo kDTypes[] = {
    {DType::F32, "F32", 4}, {DType::F16, "F16", 2},  {DType::BF16, "BF16", 2},
    {DType::F64, "F64", 8}, {DType::I64, "I64", 8},  {DType::I32, "I32", 4},
    {DType::I16, "I16", 2}, {DType::I8, "I8", 1},    {DType::U8, "U8", 1},
    {DType::BOOL, "BOOL", 1},
};

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error(path + ": " + msg);
}
}  // namespace

const char* dtype_name(DType t) {
    for (const auto& d : kDTypes) if (d.t == t) return d.name;
    return "?";
}

DType dtype_from_name(const std::string& s) {
    for (const auto& d : kDTypes) if (s == d.name) return d.t;
    throw std::runtime_error("unsupported safetensors dtype " + s);
}

size_t dtype_size(DType t) {
    for (const auto& d : kDTypes) if (d.t == t) return d.size;
    return 0;
}

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

std::string TensorView::shape_str() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
    try {
        open_and_parse();
    } catch (...) {
        if (map_) ::munmap(map_, map_size_);
        if (fd_ >= 0) ::close(fd_);
        map_ = nullptr;
        fd_ = -1;
        throw;
    }
}

void SafeTensorsFile::open_and_parse() {
    const std::string& path = path_;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) fail(path, std::string("cannot open: ") + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd_, &st) != 0) fail(path, "fstat failed");
    map_size_ = static_cast<size_t>(st.st_size);
    if (map_size_ < 8) fail(path, "file too small for a safetensors header");
    map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (map_ == MAP_FAILED) { map_ = nullptr; fail(path, "mmap failed"); }

    const auto* base = static_cast<const unsigned char*>(map_);
    uint64_t n = 0;
    std::memcpy(&n, base, 8);  // little-endian host assumed (x86-64, aarch64)
    if (n > map_size_ - 8) fail(path, "header length exceeds file size");
    json header = json::parse(base + 8, base + 8 + n);
    const size_t data_begin = 8 + n;
    const size_t data_size = map_size_ - data_begin;

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") {
            for (auto m = it.value().begin(); m != it.value().end(); ++m)
                metadata_[m.key()] = m.value().get<std::string>();
            continue;
        }
        TensorView v;
        v.name = it.key();
        v.dtype = dtype_from_name(it.value().at("dtype").get<std::string>());
        v.shape = it.value().at("shape").get<std::vector<int64_t>>();
        auto off = it.value().at("data_offsets").get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[0] > off[1] || off[1] > data_size)
            fail(path, "bad data_offsets for " + v.name);
        v.nbytes = off[1] - off[0];
        uint64_t n_elem = 1;
        for (int64_t d : v.shape) {
            if (d < 0) fail(path, "negative dimension in " + v.name);
            if (d != 0 && n_elem > UINT64_MAX / static_cast<uint64_t>(d)) fail(path, "shape overflow in " + v.name);
            n_elem *= static_cast<uint64_t>(d);
        }
        if (n_elem > v.nbytes || n_elem * dtype_size(v.dtype) != v.nbytes)
            fail(path, "size mismatch for " + v.name);
        v.data = base + data_begin + off[0];
        if (reinterpret_cast<uintptr_t>(v.data) % dtype_size(v.dtype) != 0)
            fail(path, "tensor " + v.name + " is not aligned to its element size");
        tensors_.emplace(v.name, std::move(v));
    }
}

// Left to page faults, a cold mapping on Lustre is read as one serial read-ahead stream at
// ~100 MB/s (13 s for Varda's 1.3 GB), and even a cached one costs a fault per page. Here
// kThreads threads take kChunk-sized pieces of the mapping in file order and populate them with
// MADV_POPULATE_READ (or by touching every page on kernels without it): the reads run in
// parallel (~3 GB/s on a cold A100 node) and the page tables are filled, so the backend's reads
// fault on nothing. Chosen by measurement (16 or 32 threads, 8 or 32 MB chunks were within noise;
// a parallel pread into the page cache alone left the per-page faults and was 1 s slower).
// VARDA_PREFETCH_THREADS overrides the thread count; 0 disables the prefetch.
void SafeTensorsFile::prefetch(int threads) const {
    constexpr int kThreads = 32;
    constexpr size_t kChunk = size_t{8} << 20;
    if (const char* e = std::getenv("VARDA_PREFETCH_THREADS"); e && *e) threads = std::atoi(e);
    else if (threads <= 0) threads = kThreads;
    if (threads <= 0 || !map_) return;
    const size_t n_chunks = (map_size_ + kChunk - 1) / kChunk;
    threads = static_cast<int>(std::min<size_t>(static_cast<size_t>(threads), n_chunks));
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    std::atomic<size_t> next{0};
    std::atomic<bool> populate{true};
    auto worker = [&] {
        unsigned sum = 0;
        for (size_t c; (c = next.fetch_add(1)) < n_chunks;) {
            const size_t lo = c * kChunk, hi = std::min(map_size_, lo + kChunk);
            char* p = static_cast<char*>(map_) + lo;  // lo is a multiple of kChunk: page aligned
            if (populate.load(std::memory_order_relaxed)) {
                if (::madvise(p, hi - lo, MADV_POPULATE_READ) == 0) continue;
                if (errno != EINVAL) continue;  // EINTR, or an I/O error that surfaces where the bytes are read, as before
                populate.store(false, std::memory_order_relaxed);  // kernel without it: touch the pages
            }
            const volatile unsigned char* q = reinterpret_cast<const unsigned char*>(p);
            for (size_t o = 0; o < hi - lo; o += page) sum += q[o];
        }
        (void)sum;
    };
    std::vector<std::thread> pool;
    try {
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    } catch (const std::system_error&) {
        // fewer threads than asked: the ones started (and this one) still cover every chunk
    }
    worker();
    for (auto& t : pool) t.join();
}

SafeTensorsFile::~SafeTensorsFile() {
    if (map_) ::munmap(map_, map_size_);
    if (fd_ >= 0) ::close(fd_);
}

std::string SafeTensorsFile::metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) fail(path_, "missing __metadata__ entry '" + key + "'");
    return it->second;
}

const TensorView& SafeTensorsFile::peek(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail(path_, "missing tensor '" + name + "'");
    return it->second;
}

const TensorView& SafeTensorsFile::take(const std::string& name) {
    const TensorView& v = peek(name);
    consumed_.insert(name);
    return v;
}

const TensorView& SafeTensorsFile::take(const std::string& name, DType dtype,
                                        const std::vector<int64_t>& shape) {
    const TensorView& v = take(name);
    if (v.dtype != dtype)
        fail(path_, "tensor '" + name + "' has dtype " + dtype_name(v.dtype) + ", expected " +
                        dtype_name(dtype));
    if (v.shape != shape) {
        TensorView want; want.shape = shape;
        fail(path_, "tensor '" + name + "' has shape " + v.shape_str() + ", expected " +
                        want.shape_str());
    }
    return v;
}

std::vector<std::string> SafeTensorsFile::names() const {
    std::vector<std::string> out;
    for (const auto& kv : tensors_) out.push_back(kv.first);
    return out;
}

std::vector<std::string> SafeTensorsFile::unconsumed() const {
    std::vector<std::string> out;
    for (const auto& kv : tensors_)
        if (!consumed_.count(kv.first)) out.push_back(kv.first);
    return out;
}

void SafeTensorsFile::require_all_consumed() const {
    auto left = unconsumed();
    if (left.empty()) return;
    std::string msg = std::to_string(left.size()) + " tensor(s) not consumed by the engine:";
    for (size_t i = 0; i < left.size() && i < 20; ++i) msg += " " + left[i];
    fail(path_, msg);
}

void write_safetensors(const std::string& path, const std::vector<TensorOut>& tensors,
                       const std::map<std::string, std::string>& metadata) {
    json header = json::object();
    if (!metadata.empty()) header["__metadata__"] = metadata;
    uint64_t off = 0;
    for (const auto& t : tensors) {
        uint64_t n = dtype_size(t.dtype);
        for (int64_t s : t.shape) n *= static_cast<uint64_t>(s);
        header[t.name] = {{"dtype", dtype_name(t.dtype)}, {"shape", t.shape},
                          {"data_offsets", {off, off + n}}};
        off += n;
    }
    std::string h = header.dump();
    while ((8 + h.size()) % 8) h.push_back(' ');

    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) fail(tmp, "cannot open for writing");
    uint64_t n = h.size();
    bool ok = std::fwrite(&n, 8, 1, f) == 1 && std::fwrite(h.data(), 1, h.size(), f) == h.size();
    for (const auto& t : tensors) {
        uint64_t nb = dtype_size(t.dtype);
        for (int64_t s : t.shape) nb *= static_cast<uint64_t>(s);
        ok = ok && std::fwrite(t.data, 1, nb, f) == nb;
    }
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        fail(path, "write failed");
    }
}

}  // namespace varda
