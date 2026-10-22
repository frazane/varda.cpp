// Prints a converted model's identity, dimensions and tensor table, and checks that the
// config parses and the graphs are well formed.
//   varda-info MODEL.safetensors [--tensors]
#include <cstdio>
#include <cstring>
#include <exception>

#include "varda/config.h"
#include "varda/format.h"
#include "varda/graph.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.safetensors [--tensors]\n", argv[0]);
        return 2;
    }
    try {
        varda::SafeTensorsFile f(argv[1]);
        varda::Config c = varda::parse_config(f.metadata("config"));
        std::printf("checkpoint  %s\nuuid        %s\nrun_id      %s\nsha256      %s\n",
                    c.checkpoint.c_str(), c.uuid.c_str(), c.run_id.c_str(), c.sha256.c_str());
        std::printf("dims        C=%d heads=%d x %d mlp=%d proc_layers=%d T_in=%d\n", c.num_channels,
                    c.num_heads, c.head_dim, c.mlp_hidden, c.num_processor_layers, c.n_input_steps);
        std::printf("vars        data=%d input=%d output=%d prognostic=%zu\n", c.n_data_vars,
                    c.n_input_vars, c.n_output_vars, c.residual_input.size());
        std::printf("nodes       data=%lld hidden=%lld\n", (long long)c.n_data_nodes,
                    (long long)c.n_hidden_nodes);
        for (const auto& b : c.bounding) std::printf("bounding    %s (%zu vars)\n", b.type.c_str(), b.index.size());
        std::printf("mappers     enc chunks=%d dec chunks=%d\n", c.enc_chunks, c.dec_chunks);
        const char* gname[] = {"enc", "proc", "dec"};
        const int64_t nsrc[] = {c.n_data_nodes, c.n_hidden_nodes, c.n_hidden_nodes};
        const int64_t ndst[] = {c.n_hidden_nodes, c.n_hidden_nodes, c.n_data_nodes};
        for (int i = 0; i < 3; ++i) {
            std::string p = std::string("graph.") + gname[i];
            auto g = varda::build_csr_by_dst(f.peek(p + ".edge_index"), f.peek(p + ".edge_attr"), nsrc[i], ndst[i]);
            int32_t mn = g.n_dst ? g.degree(0) : 0;
            for (int64_t d = 0; d < g.n_dst; ++d) mn = std::min(mn, g.degree(d));
            std::printf("graph %-5s edges=%lld in-degree %d..%d\n", gname[i], (long long)g.n_edges, mn, g.max_degree());
        }
        size_t total = 0;
        for (const auto& n : f.names()) total += f.peek(n).nbytes;
        std::printf("tensors     %zu (%.1f MB)\n", f.names().size(), total / 1e6);
        if (argc > 2 && std::strcmp(argv[2], "--tensors") == 0)
            for (const auto& n : f.names()) {
                const auto& t = f.peek(n);
                std::printf("  %-28s %-4s %s\n", n.c_str(), varda::dtype_name(t.dtype), t.shape_str().c_str());
            }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
