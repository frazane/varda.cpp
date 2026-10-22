// Model configuration, parsed from the `config` metadata of a converted file
// (docs/format.md §1.2). Every size and setting the engine uses comes from here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace varda {

struct Bounding {
    std::string type;              // "relu" | "hardtanh" | "fraction"
    std::vector<int> index;        // model-output indices
    std::vector<int> total_index;  // fraction only
    float min = 0.0f, max = 0.0f;  // hardtanh, fraction
};

struct BlockInfo {
    std::string name;  // "enc.block", "proc.<i>", "dec.block"
    std::string kind;  // "graph_transformer_mapper_block" | "graph_transformer_processor_block"
};

struct Config {
    std::string json_text;  // the raw config JSON
    int format_version = 0;

    // source / identity
    std::string checkpoint, uuid, run_id, sha256;

    // dims
    int num_channels = 0, num_heads = 0, head_dim = 0, mlp_hidden = 0;
    int num_processor_layers = 0, n_input_steps = 0, n_output_steps = 0;
    int n_data_vars = 0, n_input_vars = 0, n_output_vars = 0;
    int64_t n_data_nodes = 0, n_hidden_nodes = 0;
    int node_attr_dim = 0, edge_attr_dim = 0;
    float layer_norm_eps = 0.0f;
    bool gelu_tanh = false;  // dims.gelu_approximate: "none" -> erf (false), "tanh" -> true
    bool latent_skip = false;

    std::vector<std::string> var_data, var_input, var_output;
    std::vector<int> norm_input;       // data index of each model input
    std::vector<int> norm_output;      // data index of each model output
    std::vector<int> residual_input;   // model-input index of each prognostic variable
    std::vector<int> residual_output;  // model-output index of each prognostic variable
    int residual_step = -1;            // residual: x_norm[:, residual_step] (skip)

    std::vector<Bounding> bounding;    // applied in this order, normalised space
    int enc_chunks = 1, dec_chunks = 1;  // anemoi mapper destination chunks (GEMM rows only)
    std::vector<BlockInfo> blocks;     // execution order
    std::vector<std::string> absent;   // weight names with no tensor (bias=False)

    // Encoder source features: n_input_steps * n_input_vars + node_attr_dim.
    int src_features() const { return n_input_steps * n_input_vars + node_attr_dim; }
    bool is_absent(const std::string& name) const;
};

// Parses and validates; throws std::runtime_error on unknown kinds or inconsistent sizes.
Config parse_config(const std::string& json_text);

}  // namespace varda
