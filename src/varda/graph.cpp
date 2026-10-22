#include "varda/graph.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace varda {

int32_t CsrGraph::max_degree() const {
    int32_t m = 0;
    for (int64_t d = 0; d < n_dst; ++d) m = std::max(m, degree(d));
    return m;
}

CsrGraph build_csr_by_dst(const TensorView& edge_index, const TensorView& edge_attr,
                          int64_t n_src, int64_t n_dst) {
    if (edge_index.shape.size() != 2 || edge_index.shape[0] != 2)
        throw std::runtime_error(edge_index.name + ": expected shape [2, E], got " + edge_index.shape_str());
    const int64_t E = edge_index.shape[1];
    if (edge_attr.dtype != DType::F32 || edge_attr.shape.size() != 2 || edge_attr.shape[0] != E)
        throw std::runtime_error(edge_attr.name + ": expected F32 [E, A] with E = " + std::to_string(E));
    if (E >= std::numeric_limits<int32_t>::max() || n_src >= std::numeric_limits<int32_t>::max() ||
        n_dst >= std::numeric_limits<int32_t>::max())
        throw std::runtime_error(edge_index.name + ": graph too large for int32 indices");

    auto get = [&](int row, int64_t e) -> int64_t {
        if (edge_index.dtype == DType::I64) return edge_index.as<int64_t>()[row * E + e];
        if (edge_index.dtype == DType::I32) return edge_index.as<int32_t>()[row * E + e];
        throw std::runtime_error(edge_index.name + ": edge_index must be I64 or I32");
    };

    CsrGraph g;
    g.n_src = n_src;
    g.n_dst = n_dst;
    g.n_edges = E;
    g.attr_dim = static_cast<int>(edge_attr.shape[1]);
    g.row_ptr.assign(n_dst + 1, 0);
    for (int64_t e = 0; e < E; ++e) {
        const int64_t s = get(0, e), d = get(1, e);
        if (s < 0 || s >= n_src || d < 0 || d >= n_dst)
            throw std::runtime_error(edge_index.name + ": edge " + std::to_string(e) + " out of range");
        ++g.row_ptr[d + 1];
    }
    for (int64_t d = 0; d < n_dst; ++d) g.row_ptr[d + 1] += g.row_ptr[d];

    // Stable counting sort by destination.
    std::vector<int32_t> fill(g.row_ptr.begin(), g.row_ptr.end() - 1);
    g.src.resize(E);
    g.edge_id.resize(E);
    g.attr.resize(static_cast<size_t>(E) * g.attr_dim);
    const float* a = edge_attr.as<float>();
    for (int64_t e = 0; e < E; ++e) {
        const int32_t slot = fill[get(1, e)]++;
        g.src[slot] = static_cast<int32_t>(get(0, e));
        g.edge_id[slot] = static_cast<int32_t>(e);
        std::memcpy(&g.attr[static_cast<size_t>(slot) * g.attr_dim], a + e * g.attr_dim,
                    sizeof(float) * g.attr_dim);
    }
    return g;
}

}  // namespace varda
