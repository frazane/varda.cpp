#include "varda/config.h"

#include <algorithm>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace varda {

using json = nlohmann::json;

namespace {

constexpr const char* kNewKindDoc = "see docs/design.md (adding a component kind)";

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("unsupported config: " + what + "; " + kNewKindDoc);
}

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("inconsistent config: " + what);
}

void check_range(const std::vector<int>& idx, int n, const std::string& what) {
    for (int i : idx) require(i >= 0 && i < n, what + " index " + std::to_string(i) + " out of range");
}

}  // namespace

bool Config::is_absent(const std::string& name) const {
    return std::find(absent.begin(), absent.end(), name) != absent.end();
}

Config parse_config(const std::string& json_text) {
    const json j = json::parse(json_text);
    Config c;
    c.json_text = json_text;
    c.format_version = j.at("format_version").get<int>();
    if (c.format_version != 3)
        unsupported("format_version " + std::to_string(c.format_version) + " (this engine reads 3)");

    const json& src = j.at("source");
    c.checkpoint = src.value("checkpoint", "");
    c.uuid = src.value("uuid", "");
    c.run_id = src.value("run_id", "");
    c.sha256 = src.value("sha256", "");

    const json& arch = j.at("architecture");
    auto want_kind = [&](const char* part, const char* kind) {
        std::string k = arch.at(part).at("kind").get<std::string>();
        if (k != kind) unsupported(std::string("architecture.") + part + ".kind '" + k + "'");
    };
    want_kind("model", "enc_proc_dec");
    want_kind("encoder", "graph_transformer_forward_mapper");
    want_kind("processor", "graph_transformer_processor");
    want_kind("decoder", "graph_transformer_backward_mapper");

    const json& d = j.at("dims");
    c.num_channels = d.at("num_channels");
    c.num_heads = d.at("num_heads");
    c.head_dim = d.at("head_dim");
    c.mlp_hidden = d.at("mlp_hidden");
    c.num_processor_layers = d.at("num_processor_layers");
    c.n_input_steps = d.at("n_input_steps");
    c.n_output_steps = d.value("n_output_steps", 1);
    c.n_data_vars = d.at("n_data_vars");
    c.n_input_vars = d.at("n_input_vars");
    c.n_output_vars = d.at("n_output_vars");
    c.n_data_nodes = d.at("n_data_nodes");
    c.n_hidden_nodes = d.at("n_hidden_nodes");
    c.node_attr_dim = d.at("node_attr_dim");
    c.edge_attr_dim = d.at("edge_attr_dim");
    c.layer_norm_eps = d.at("layer_norm_eps").get<float>();
    const std::string gelu = d.at("gelu_approximate");
    if (gelu == "none") c.gelu_tanh = false;
    else if (gelu == "tanh") c.gelu_tanh = true;
    else unsupported("dims.gelu_approximate '" + gelu + "'");
    c.latent_skip = d.at("latent_skip");

    require(c.num_heads * c.head_dim == c.num_channels, "num_heads * head_dim != num_channels");
    if (c.n_output_steps != 1) unsupported("n_output_steps != 1");

    const json& v = j.at("variables");
    c.var_data = v.at("data").get<std::vector<std::string>>();
    c.var_input = v.at("input").get<std::vector<std::string>>();
    c.var_output = v.at("output").get<std::vector<std::string>>();
    require((int)c.var_data.size() == c.n_data_vars, "variables.data size");
    require((int)c.var_input.size() == c.n_input_vars, "variables.input size");
    require((int)c.var_output.size() == c.n_output_vars, "variables.output size");

    const json& ix = j.at("indices");
    c.norm_input = ix.at("norm_input").get<std::vector<int>>();
    c.norm_output = ix.at("norm_output").get<std::vector<int>>();
    c.residual_input = ix.at("residual_input").get<std::vector<int>>();
    c.residual_output = ix.at("residual_output").get<std::vector<int>>();
    require((int)c.norm_input.size() == c.n_input_vars, "indices.norm_input size");
    require((int)c.norm_output.size() == c.n_output_vars, "indices.norm_output size");
    require(c.residual_input.size() == c.residual_output.size(), "residual index sizes differ");
    check_range(c.norm_input, c.n_data_vars, "norm_input");
    check_range(c.norm_output, c.n_data_vars, "norm_output");
    check_range(c.residual_input, c.n_input_vars, "residual_input");
    check_range(c.residual_output, c.n_output_vars, "residual_output");

    const json& r = j.at("residual");
    if (r.at("type") != "skip") unsupported("residual.type '" + r.at("type").get<std::string>() + "'");
    c.residual_step = r.at("step");
    if (c.residual_step < 0) c.residual_step += c.n_input_steps;
    require(c.residual_step >= 0 && c.residual_step < c.n_input_steps, "residual.step");

    for (const json& b : j.at("bounding")) {
        Bounding bd;
        bd.type = b.at("type");
        bd.index = b.at("index").get<std::vector<int>>();
        check_range(bd.index, c.n_output_vars, "bounding");
        if (bd.type == "relu") {
        } else if (bd.type == "hardtanh" || bd.type == "fraction") {
            bd.min = b.at("min").get<float>();
            bd.max = b.at("max").get<float>();
            if (bd.type == "fraction") {
                bd.total_index = b.at("total_index").get<std::vector<int>>();
                check_range(bd.total_index, c.n_output_vars, "bounding total_index");
                require(bd.total_index.size() == 1 || bd.total_index.size() == bd.index.size(),
                        "fraction total_index size");
            }
        } else {
            unsupported("bounding type '" + bd.type + "'");
        }
        c.bounding.push_back(bd);
    }

    if (j.contains("mappers")) {
        c.enc_chunks = j.at("mappers").at("enc").at("num_chunks");
        c.dec_chunks = j.at("mappers").at("dec").at("num_chunks");
    }

    // Blocks in execution order: enc.block, proc.0 .. proc.{n-1}, dec.block.
    const json& blocks = j.at("blocks");
    std::vector<std::string> order{"enc.block"};
    for (int i = 0; i < c.num_processor_layers; ++i) order.push_back("proc." + std::to_string(i));
    order.push_back("dec.block");
    require(blocks.size() == order.size(), "blocks count");
    for (const auto& name : order) {
        std::string kind = blocks.at(name).at("kind");
        const bool mapper = name == "enc.block" || name == "dec.block";
        const char* want = mapper ? "graph_transformer_mapper_block" : "graph_transformer_processor_block";
        if (kind != want) unsupported("block " + name + " kind '" + kind + "'");
        c.blocks.push_back({name, kind});
    }

    c.absent = j.value("absent", std::vector<std::string>{});
    return c;
}

}  // namespace varda
