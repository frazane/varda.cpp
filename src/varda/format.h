// safetensors container: read-only mmap reader with strict consumption tracking, and a
// small writer (docs/format.md §1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace varda {

enum class DType { F32, F16, BF16, F64, I64, I32, I16, I8, U8, BOOL };

const char* dtype_name(DType t);
DType dtype_from_name(const std::string& s);
size_t dtype_size(DType t);

struct TensorView {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const void* data = nullptr;  // points into the mapped file
    size_t nbytes = 0;

    int64_t numel() const;
    int64_t dim(int i) const { return shape.at(i < 0 ? shape.size() + i : i); }
    std::string shape_str() const;
    template <class T> const T* as() const { return static_cast<const T*>(data); }
};

class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& path);
    ~SafeTensorsFile();
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    const std::string& path() const { return path_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    std::string metadata(const std::string& key) const;  // throws if missing

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    // Look up without marking as consumed.
    const TensorView& peek(const std::string& name) const;
    // Look up and mark as consumed; optionally check dtype and shape.
    const TensorView& take(const std::string& name);
    const TensorView& take(const std::string& name, DType dtype, const std::vector<int64_t>& shape);
    std::vector<std::string> names() const;
    // Names never passed to take().
    std::vector<std::string> unconsumed() const;
    // Throws, listing the names, unless every tensor was taken.
    void require_all_consumed() const;

private:
    std::string path_;
    int fd_ = -1;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    std::map<std::string, std::string> metadata_;
    std::map<std::string, TensorView> tensors_;
    std::set<std::string> consumed_;
};

// Writes a safetensors file atomically (temporary file in the same directory, then rename).
struct TensorOut {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    const void* data;
};
void write_safetensors(const std::string& path, const std::vector<TensorOut>& tensors,
                       const std::map<std::string, std::string>& metadata);

}  // namespace varda
