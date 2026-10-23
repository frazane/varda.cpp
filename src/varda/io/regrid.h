// IFS open data 0.25° -> N320, as MIR 1.29 `linear` does it for anemoi (docs/spec-io.md §2.4):
// y = W x in double with MIR's own weight matrix (rows summed sequentially in MIR's stored column
// order), then a 16-bit grid_ccsds GRIB2 re-encode and decode through ecCodes, which reproduces
// the quantisation of MIR's output message.
//
// The matrix cannot be derived in C++ (MIR's triangulation); it is a data file exported by
// python/export_regrid_matrix.py from MIR's weight cache:
//   regrid.indptr I64 [n_target + 1], regrid.indices I32 [nnz] (stored order), regrid.weights F64 [nnz],
//   metadata "regrid" = JSON with "source_geometry" (keys of a real open-data message),
//   "n_source", "n_target", "mat_sha256", ...
// MIR's cache file (.mat, an eckit Stream) can be read directly too (load_mir_mat), for C1a.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "varda/io/grib.h"

namespace varda::io {

// The source-grid keys a message must match (python/export_regrid_matrix.py GEOMETRY_KEYS).
struct RegridGeometry {
    std::string gridType;
    long Ni = 0, Nj = 0, numberOfDataPoints = 0;
    double latitudeOfFirstGridPointInDegrees = 0, longitudeOfFirstGridPointInDegrees = 0;
    double latitudeOfLastGridPointInDegrees = 0, longitudeOfLastGridPointInDegrees = 0;
    double iDirectionIncrementInDegrees = 0, jDirectionIncrementInDegrees = 0;
    long iScansNegatively = 0, jScansPositively = 0, jPointsAreConsecutive = 0;

    static RegridGeometry of(const GribHandle& h);
    // The real open-data layout (docs/spec-io.md §1.5): 1440 x 721, lat 90 -> -90, lon 180 -> 179.75.
    static RegridGeometry ifs_open_data_0p25();
    // Empty if equal, else "key: message value != matrix value; ...".
    std::string diff(const RegridGeometry& o) const;
};

class RegridMatrix {
public:
    // The exported safetensors file (geometry from its metadata).
    static RegridMatrix load(const std::string& path);
    // MIR's cache file, read raw (rows, cols, nnz, endianness, item sizes, blobs; SP/mir/weight_matrix.py).
    static RegridMatrix load_mir_mat(const std::string& path, const RegridGeometry& source);

    int64_t n_source() const { return n_source_; }
    int64_t n_target() const { return int64_t(indptr_.size()) - 1; }
    int64_t nnz() const { return int64_t(weights_.size()); }
    const RegridGeometry& source() const { return source_; }
    const std::string& path() const { return path_; }
    const std::string& metadata_json() const { return meta_; }  // "" for a .mat
    const std::vector<int64_t>& indptr() const { return indptr_; }
    const std::vector<int64_t>& indices() const { return indices_; }
    const std::vector<double>& weights() const { return weights_; }

    // Throws (naming the key) if the message's geometry differs from the matrix's source
    // geometry, if it has a bitmap (MIR's missing-value handling is not W x) or the wrong size.
    void check_message(const GribHandle& h) const;

    // y[i] = ((0 + w[i,0] x[c0]) + w[i,1] x[c1]) + ... in double, stored order.
    void apply(const double* x, double* y) const;

    // True if both hold the same rows, columns, stored indices and weights (bitwise).
    bool same_matrix(const RegridMatrix& o) const;

private:
    std::string path_, meta_;
    int64_t n_source_ = 0;
    RegridGeometry source_;
    std::vector<int64_t> indptr_, indices_;
    std::vector<double> weights_;
};

// The handle the regridded values are encoded into: an N320 GRIB2 message with grid_ccsds
// packing (e.g. the first message of the HF template `ifs-levtype=sfc.grib`). Its values are
// replaced; bitsPerValue is set to 16 and decimalScaleFactor to 0 (MIR's accuracy=16 output).
class Regridder {
public:
    Regridder(const RegridMatrix& m, GribHandle n320_template);

    // One field: check geometry, x = codes_get_double_array, y = W x, encode (16-bit ccsds) and
    // decode back. Returns the decoded values; `as_double` selects codes_get_double_array (for
    // gh -> z, §2.4) instead of codes_get_float_array (stored into `out_f`).
    void regrid(const GribHandle& src, std::vector<float>* out_f, std::vector<double>* out_d) const;
    // Encode y (N320, double) as MIR's output would be and return the encoded handle.
    GribHandle encode(const std::vector<double>& y) const;
    // The target coordinates of the template (latitudes / longitudes, ecCodes geometry).
    void target_coordinates(std::vector<double>* lat, std::vector<double>* lon) const;

    const RegridMatrix& matrix() const { return m_; }

private:
    const RegridMatrix& m_;
    GribHandle tmpl_;
};

}  // namespace varda::io
