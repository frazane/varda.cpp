// Static graphs in destination-major (CSR) layout, built once at load time.
#pragma once

#include <cstdint>
#include <vector>

#include "varda/format.h"

namespace varda {

// Edges grouped by destination node. Within a destination, edges keep their stored order
// (stable sort), so per-destination sums visit edges in the same order as a sequential
// scatter over the stored edge list.
struct CsrGraph {
    int64_t n_src = 0, n_dst = 0, n_edges = 0;
    int attr_dim = 0;
    std::vector<int32_t> row_ptr;  // [n_dst + 1]
    std::vector<int32_t> src;      // [n_edges] source node of each CSR slot
    std::vector<int32_t> edge_id;  // [n_edges] stored edge index of each CSR slot
    std::vector<float> attr;       // [n_edges, attr_dim] edge attributes in CSR order

    int32_t degree(int64_t d) const { return row_ptr[d + 1] - row_ptr[d]; }
    int32_t max_degree() const;
};

// edge_index: [2, E] I64 or I32 (row 0 source, row 1 destination); edge_attr: [E, A] F32.
// Throws if an index is out of range or E does not fit in int32.
CsrGraph build_csr_by_dst(const TensorView& edge_index, const TensorView& edge_attr,
                          int64_t n_src, int64_t n_dst);

}  // namespace varda
