#include "varda/io/regrid.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "varda/format.h"

namespace varda::io {

using json = nlohmann::json;

// ------------------------------------------------------------------------------------------- geometry

RegridGeometry RegridGeometry::of(const GribHandle& h) {
    RegridGeometry g;
    g.gridType = h.get_string("gridType");
    g.Ni = h.get_long("Ni");
    g.Nj = h.get_long("Nj");
    g.numberOfDataPoints = h.get_long("numberOfDataPoints");
    g.latitudeOfFirstGridPointInDegrees = h.get_double("latitudeOfFirstGridPointInDegrees");
    g.longitudeOfFirstGridPointInDegrees = h.get_double("longitudeOfFirstGridPointInDegrees");
    g.latitudeOfLastGridPointInDegrees = h.get_double("latitudeOfLastGridPointInDegrees");
    g.longitudeOfLastGridPointInDegrees = h.get_double("longitudeOfLastGridPointInDegrees");
    g.iDirectionIncrementInDegrees = h.get_double("iDirectionIncrementInDegrees");
    g.jDirectionIncrementInDegrees = h.get_double("jDirectionIncrementInDegrees");
    g.iScansNegatively = h.get_long("iScansNegatively");
    g.jScansPositively = h.get_long("jScansPositively");
    g.jPointsAreConsecutive = h.get_long("jPointsAreConsecutive");
    return g;
}

RegridGeometry RegridGeometry::ifs_open_data_0p25() {
    RegridGeometry g;
    g.gridType = "regular_ll";
    g.Ni = 1440;
    g.Nj = 721;
    g.numberOfDataPoints = 1440 * 721;
    g.latitudeOfFirstGridPointInDegrees = 90;
    g.longitudeOfFirstGridPointInDegrees = 180;
    g.latitudeOfLastGridPointInDegrees = -90;
    g.longitudeOfLastGridPointInDegrees = 179.75;
    g.iDirectionIncrementInDegrees = 0.25;
    g.jDirectionIncrementInDegrees = 0.25;
    return g;
}

std::string RegridGeometry::diff(const RegridGeometry& o) const {
    std::ostringstream s;
    s.precision(17);
    auto cmp = [&](const char* k, auto a, auto b) {
        if (!(a == b)) s << k << ": message " << a << " != matrix " << b << "; ";
    };
    cmp("gridType", gridType, o.gridType);
    cmp("Ni", Ni, o.Ni);
    cmp("Nj", Nj, o.Nj);
    cmp("numberOfDataPoints", numberOfDataPoints, o.numberOfDataPoints);
    cmp("latitudeOfFirstGridPointInDegrees", latitudeOfFirstGridPointInDegrees, o.latitudeOfFirstGridPointInDegrees);
    cmp("longitudeOfFirstGridPointInDegrees", longitudeOfFirstGridPointInDegrees, o.longitudeOfFirstGridPointInDegrees);
    cmp("latitudeOfLastGridPointInDegrees", latitudeOfLastGridPointInDegrees, o.latitudeOfLastGridPointInDegrees);
    cmp("longitudeOfLastGridPointInDegrees", longitudeOfLastGridPointInDegrees, o.longitudeOfLastGridPointInDegrees);
    cmp("iDirectionIncrementInDegrees", iDirectionIncrementInDegrees, o.iDirectionIncrementInDegrees);
    cmp("jDirectionIncrementInDegrees", jDirectionIncrementInDegrees, o.jDirectionIncrementInDegrees);
    cmp("iScansNegatively", iScansNegatively, o.iScansNegatively);
    cmp("jScansPositively", jScansPositively, o.jScansPositively);
    cmp("jPointsAreConsecutive", jPointsAreConsecutive, o.jPointsAreConsecutive);
    return s.str();
}

// --------------------------------------------------------------------------------------------- matrix

namespace {

RegridGeometry geometry_from_json(const json& g) {
    RegridGeometry r;
    r.gridType = g.at("gridType").get<std::string>();
    r.Ni = g.at("Ni").get<long>();
    r.Nj = g.at("Nj").get<long>();
    r.numberOfDataPoints = g.at("numberOfDataPoints").get<long>();
    r.latitudeOfFirstGridPointInDegrees = g.at("latitudeOfFirstGridPointInDegrees").get<double>();
    r.longitudeOfFirstGridPointInDegrees = g.at("longitudeOfFirstGridPointInDegrees").get<double>();
    r.latitudeOfLastGridPointInDegrees = g.at("latitudeOfLastGridPointInDegrees").get<double>();
    r.longitudeOfLastGridPointInDegrees = g.at("longitudeOfLastGridPointInDegrees").get<double>();
    r.iDirectionIncrementInDegrees = g.at("iDirectionIncrementInDegrees").get<double>();
    r.jDirectionIncrementInDegrees = g.at("jDirectionIncrementInDegrees").get<double>();
    r.iScansNegatively = g.at("iScansNegatively").get<long>();
    r.jScansPositively = g.at("jScansPositively").get<long>();
    r.jPointsAreConsecutive = g.at("jPointsAreConsecutive").get<long>();
    if (g.contains("bitmapPresent") && !g["bitmapPresent"].is_null() && g["bitmapPresent"].get<long>() != 0)
        throw std::runtime_error("regrid matrix: source geometry records a bitmap");
    return r;
}

void check_csr(const std::string& path, const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices,
               size_t nnz, int64_t n_source) {
    if (indptr.empty() || indptr.front() != 0 || indptr.back() != int64_t(nnz))
        throw std::runtime_error(path + ": inconsistent CSR row pointers");
    for (size_t i = 0; i + 1 < indptr.size(); ++i)
        if (indptr[i + 1] < indptr[i]) throw std::runtime_error(path + ": decreasing CSR row pointers");
    for (int64_t c : indices)
        if (c < 0 || c >= n_source) throw std::runtime_error(path + ": column index out of range");
}

// eckit::Stream, as SP/eckit/stream.py reads it: a tag byte (end-of-object tags skipped), then
// the payload in network byte order; unsigned long and int are 4 bytes, a large blob's length 8.
class EckitStream {
public:
    explicit EckitStream(const std::string& path) : path_(path), f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("cannot open " + path);
    }
    uint32_t read_u32(int tag) {
        read_tag(tag);
        unsigned char b[4];
        read(b, 4);
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
    }
    std::vector<unsigned char> read_large_blob() {
        read_tag(21);
        unsigned char b[8];
        read(b, 8);
        uint64_t n = 0;
        for (int i = 0; i < 8; ++i) n = (n << 8) | b[i];
        std::vector<unsigned char> out(n);
        read(out.data(), n);
        return out;
    }

private:
    void read(void* p, size_t n) {
        f_.read(static_cast<char*>(p), std::streamsize(n));
        if (size_t(f_.gcount()) != n) throw std::runtime_error(path_ + ": truncated eckit stream");
    }
    void read_tag(int expected) {
        unsigned char t = 2;
        while (t == 2) read(&t, 1);  // TAG_END_OBJ
        if (t != expected)
            throw std::runtime_error(path_ + ": eckit stream tag " + std::to_string(t) + ", expected " +
                                     std::to_string(expected));
    }
    std::string path_;
    std::ifstream f_;
};

template <class T>
T load_le(const unsigned char* p, bool little) {
    T v;
    unsigned char b[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i) b[i] = little ? p[i] : p[sizeof(T) - 1 - i];
    std::memcpy(&v, b, sizeof(T));
    return v;
}

}  // namespace

RegridMatrix RegridMatrix::load(const std::string& path) {
    SafeTensorsFile f(path);
    RegridMatrix m;
    m.path_ = path;
    m.meta_ = f.metadata("regrid");
    const json meta = json::parse(m.meta_);
    m.source_ = geometry_from_json(meta.at("source_geometry"));
    m.n_source_ = meta.at("n_source").get<int64_t>();
    const int64_t n_target = meta.at("n_target").get<int64_t>();
    const TensorView& ip = f.take("regrid.indptr", DType::I64, {n_target + 1});
    const TensorView& ix = f.take("regrid.indices");
    const TensorView& w = f.take("regrid.weights");
    if (ix.dtype != DType::I32 || w.dtype != DType::F64 || ix.shape.size() != 1 || w.shape != ix.shape)
        throw std::runtime_error(path + ": regrid.indices must be I32 [nnz] and regrid.weights F64 [nnz]");
    f.require_all_consumed();
    const size_t nnz = size_t(ix.dim(0));
    m.indptr_.assign(ip.as<int64_t>(), ip.as<int64_t>() + n_target + 1);
    m.indices_.assign(ix.as<int32_t>(), ix.as<int32_t>() + nnz);
    m.weights_.assign(w.as<double>(), w.as<double>() + nnz);
    if (m.source_.numberOfDataPoints != m.n_source_)
        throw std::runtime_error(path + ": n_source differs from the source geometry's numberOfDataPoints");
    check_csr(path, m.indptr_, m.indices_, nnz, m.n_source_);
    return m;
}

RegridMatrix RegridMatrix::load_mir_mat(const std::string& path, const RegridGeometry& source) {
    EckitStream s(path);
    const uint32_t rows = s.read_u32(10), cols = s.read_u32(10), nnz = s.read_u32(10);
    const bool little = s.read_u32(5) != 0;
    const uint32_t isz = s.read_u32(10), ssz = s.read_u32(10);
    s.read_u32(10);  // sizeof(size)
    if ((isz != 4 && isz != 8) || (ssz != 4 && ssz != 8)) throw std::runtime_error(path + ": unsupported item sizes");
    const auto outer = s.read_large_blob(), inner = s.read_large_blob(), data = s.read_large_blob();
    if (outer.size() != size_t(rows + 1) * isz || inner.size() != size_t(nnz) * isz || data.size() != size_t(nnz) * ssz)
        throw std::runtime_error(path + ": inconsistent CSR blob sizes");
    RegridMatrix m;
    m.path_ = path;
    m.source_ = source;
    m.n_source_ = cols;
    m.indptr_.resize(rows + 1);
    m.indices_.resize(nnz);
    m.weights_.resize(nnz);
    for (size_t i = 0; i <= rows; ++i)
        m.indptr_[i] = isz == 4 ? int64_t(load_le<uint32_t>(&outer[i * 4], little))
                                : int64_t(load_le<uint64_t>(&outer[i * 8], little));
    for (size_t i = 0; i < nnz; ++i) {
        m.indices_[i] = isz == 4 ? int64_t(load_le<uint32_t>(&inner[i * 4], little))
                                 : int64_t(load_le<uint64_t>(&inner[i * 8], little));
        m.weights_[i] = ssz == 8 ? load_le<double>(&data[i * 8], little) : double(load_le<float>(&data[i * 4], little));
    }
    if (source.numberOfDataPoints != int64_t(cols))
        throw std::runtime_error(path + ": matrix columns differ from the source geometry's numberOfDataPoints");
    check_csr(path, m.indptr_, m.indices_, nnz, cols);
    return m;
}

void RegridMatrix::check_message(const GribHandle& h) const {
    const RegridGeometry g = RegridGeometry::of(h);
    const std::string d = g.diff(source_);
    if (!d.empty()) throw std::runtime_error("regrid: message geometry differs from the matrix's source: " + d);
    if (h.get_long("bitmapPresent") != 0)
        throw std::runtime_error("regrid: message has a bitmap (MIR's missing-value handling is not W x)");
    if (int64_t(h.size("values")) != n_source_) throw std::runtime_error("regrid: message size differs from the matrix");
}

void RegridMatrix::apply(const double* x, double* y) const {
    const int64_t n = n_target();
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t j = indptr_[size_t(i)]; j < indptr_[size_t(i) + 1]; ++j) {
            const double p = weights_[size_t(j)] * x[indices_[size_t(j)]];
            s = s + p;
        }
        y[i] = s;
    }
}

bool RegridMatrix::same_matrix(const RegridMatrix& o) const {
    return n_source_ == o.n_source_ && indptr_ == o.indptr_ && indices_ == o.indices_ &&
           weights_.size() == o.weights_.size() &&
           std::memcmp(weights_.data(), o.weights_.data(), weights_.size() * sizeof(double)) == 0;
}

// ------------------------------------------------------------------------------------------- regridder

Regridder::Regridder(const RegridMatrix& m, GribHandle n320_template) : m_(m), tmpl_(std::move(n320_template)) {
    if (!tmpl_) throw std::runtime_error("regrid: no N320 template handle");
    if (int64_t(tmpl_.size("values")) != m.n_target())
        throw std::runtime_error("regrid: template has " + std::to_string(tmpl_.size("values")) +
                                 " points, the matrix " + std::to_string(m.n_target()) + " rows");
    if (tmpl_.get_string("packingType") != "grid_ccsds")
        throw std::runtime_error("regrid: template packing is " + tmpl_.get_string("packingType") +
                                 ", expected grid_ccsds");
    if (tmpl_.get_long("bitmapPresent") != 0) throw std::runtime_error("regrid: template has a bitmap");
}

GribHandle Regridder::encode(const std::vector<double>& y) const {
    GribHandle h = tmpl_.clone();
    h.set_long("decimalScaleFactor", 0);
    h.set_long("bitsPerValue", 16);
    h.set_double_array("values", y);
    return h;
}

void Regridder::regrid(const GribHandle& src, std::vector<float>* out_f, std::vector<double>* out_d) const {
    m_.check_message(src);
    const std::vector<double> x = src.get_double_array("values");
    std::vector<double> y(size_t(m_.n_target()));
    m_.apply(x.data(), y.data());
    const GribHandle h = encode(y);
    if (out_f) *out_f = h.get_float_array("values");
    if (out_d) *out_d = h.get_double_array("values");
}

void Regridder::target_coordinates(std::vector<double>* lat, std::vector<double>* lon) const {
    if (lat) *lat = tmpl_.get_double_array("latitudes");
    if (lon) *lon = tmpl_.get_double_array("longitudes");
}

}  // namespace varda::io
