// Native Metal fp32 inference. MPS handles dense GEMMs; graph attention and all
// elementwise operations run as Metal compute kernels. No CPU compute fallback.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "varda/backend.h"
#include "varda/graph.h"
#include "varda/metal/metal_backend.h"
#include "varda/metal/kernels_source.h"

namespace varda::metal {
bool available() {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d && d.hasUnifiedMemory;
    }
}
DeviceInfo device_info() {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d || !d.hasUnifiedMemory) throw std::runtime_error("no unified-memory Metal GPU is available");
        return {d.name.UTF8String, d.maxBufferLength, d.recommendedMaxWorkingSetSize};
    }
}
}  // namespace varda::metal

namespace varda {
namespace {

uint32_t u(int64_t n) {
    if (n < 0 || uint64_t(n) > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Metal tensor dimensions exceed the 32-bit indexing limit");
    return static_cast<uint32_t>(n);
}
struct Args {
    uint32_t n = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0;
    float x = 0, y = 0;
};
static_assert(sizeof(Args) == 40, "Metal parameter layout");

struct Tensor {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    Tensor sub(int64_t floats) const {
        Tensor t = *this;
        t.offset += static_cast<NSUInteger>(floats) * sizeof(float);
        return t;
    }
    float* data() const { return reinterpret_cast<float*>(static_cast<char*>(buffer.contents) + offset); }
};
struct Linear { Tensor w, b; int64_t out = 0, in = 0; };
struct Norm { Tensor w, b; };
struct Block {
    Linear q, k, v, self, edge, proj, mlp1, mlp2;
    Norm src, dst, mlp;
};
struct Graph {
    CsrGraph host;
    Tensor rows, sources, destinations, attrs;
};
struct Bound { Bounding cfg; Tensor indices, totals; };

class MetalBackend final : public Backend {
public:
    MetalBackend(const Config& cfg, SafeTensorsFile& file, const Options& opts);
    void predict_step(const float* x, float* y, const ActivationHook* hook) override;

private:
    Tensor bytes(size_t n, const void* source = nullptr);
    Tensor floats(int64_t n) { u(n); return bytes(static_cast<size_t>(n) * sizeof(float)); }
    template<class T> Tensor upload(const std::vector<T>& v) { return bytes(v.size() * sizeof(T), v.data()); }
    Tensor arena(const std::string& name, int64_t n);
    Linear load_linear(SafeTensorsFile& f, const std::string& name, int64_t out, int64_t in);
    Norm load_norm(SafeTensorsFile& f, const std::string& name);
    Block load_block(SafeTensorsFile& f, const std::string& name, bool mapper);
    Graph load_graph(SafeTensorsFile& f, const std::string& name, int64_t src, int64_t dst);
    id<MTLCommandBuffer> command();
    void flush();
    void finish_op();
    void dispatch(const char* name, const Args& args, std::initializer_list<Tensor> buffers);
    void linear(const Linear& l, const Tensor& x, int64_t rows, const Tensor& out);
    void norm(const Norm& l, const Tensor& x, int64_t rows, const Tensor& out);
    void copy(const Tensor& x, const Tensor& out, int64_t n);
    void add(const Tensor& x, const Tensor& y, const Tensor& out, int64_t n);
    void source(const Block& b, const Tensor& x, int64_t rows, int64_t dest_row);
    void destination(const Block& b, const Graph& g, int64_t row, int64_t rows,
                     const Tensor& x, const Tensor& out);
    void report(const ActivationHook* hook, const std::string& key, const Tensor& t,
                std::vector<int64_t> shape);
    void run(const float* x, float* y, const ActivationHook* hook);

    Config cfg_;
    int64_t C_, G_, H_, F_, chunk_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLCommandBuffer> command_;
    size_t encoded_ops_ = 0;
    std::map<std::string, id<MTLComputePipelineState>> kernels_;
    std::map<std::tuple<int64_t, int64_t, int64_t>, MPSMatrixMultiplication*> gemms_;
    std::map<std::string, Tensor> arena_;
    Tensor mul_in_, add_in_, mul_out_, add_out_, data_attr_, hidden_attr_, residual_in_, residual_out_;
    Tensor keys_, values_, scores_, ln_, q_, self_, attention_, residual_, hidden_;
    Linear enc_src_, enc_dst_, dec_dst_, out_proj_;
    Norm out_norm_;
    Block enc_, dec_;
    std::vector<Block> processors_;
    Graph enc_graph_, proc_graph_, dec_graph_;
    std::vector<Bound> bounds_;
};

Tensor MetalBackend::bytes(size_t n, const void* source) {
    u(n / sizeof(float));  // every GPU buffer stores 32-bit floats or indices
    const size_t size = std::max(n, sizeof(float));
    if (size > device_.maxBufferLength)
        throw std::runtime_error("Metal buffer exceeds this GPU's maxBufferLength");
    const uint64_t used = device_.currentAllocatedSize, limit = device_.recommendedMaxWorkingSetSize;
    if (used > limit || size > limit - used)
        throw std::runtime_error("Metal model exceeds this GPU's recommended working-set budget (" +
                                 std::to_string(limit >> 20) + " MiB); use a Mac with more unified memory");
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal buffer allocation failed (" + std::to_string(size) + " bytes)");
    if (source && n) std::memcpy(buffer.contents, source, n);
    return {buffer, 0};
}

Tensor MetalBackend::arena(const std::string& name, int64_t n) {
    Tensor& t = arena_[name];
    u(n);
    if (!t.buffer || t.buffer.length < static_cast<size_t>(n) * sizeof(float)) t = floats(n);
    return t;
}

Linear MetalBackend::load_linear(SafeTensorsFile& f, const std::string& name, int64_t out, int64_t in) {
    const TensorView& w = f.take(name + ".weight", DType::F32, {out, in});
    Linear l{bytes(w.nbytes, w.data), {}, out, in};
    if (!cfg_.is_absent(name + ".bias")) {
        const TensorView& b = f.take(name + ".bias", DType::F32, {out});
        l.b = bytes(b.nbytes, b.data);
    }
    return l;
}

Norm MetalBackend::load_norm(SafeTensorsFile& f, const std::string& name) {
    const TensorView& w = f.take(name + ".weight", DType::F32, {C_});
    Norm l{bytes(w.nbytes, w.data), {}};
    if (!cfg_.is_absent(name + ".bias")) {
        const TensorView& b = f.take(name + ".bias", DType::F32, {C_});
        l.b = bytes(b.nbytes, b.data);
    }
    return l;
}

Block MetalBackend::load_block(SafeTensorsFile& f, const std::string& name, bool mapper) {
    Block b;
    b.q = load_linear(f, name + ".q", C_, C_);
    b.k = load_linear(f, name + ".k", C_, C_);
    b.v = load_linear(f, name + ".v", C_, C_);
    b.self = load_linear(f, name + ".self", C_, C_);
    b.edge = load_linear(f, name + ".edge", C_, cfg_.edge_attr_dim);
    b.proj = load_linear(f, name + ".proj", C_, C_);
    b.mlp1 = load_linear(f, name + ".mlp1", cfg_.mlp_hidden, C_);
    b.mlp2 = load_linear(f, name + ".mlp2", C_, cfg_.mlp_hidden);
    b.dst = load_norm(f, name + ".ln_dst");
    b.mlp = load_norm(f, name + ".ln_mlp");
    b.src = mapper ? load_norm(f, name + ".ln_src") : b.dst;
    return b;
}

Graph MetalBackend::load_graph(SafeTensorsFile& f, const std::string& name, int64_t src, int64_t dst) {
    const TensorView& ei = f.take("graph." + name + ".edge_index");
    const TensorView& ea = f.take("graph." + name + ".edge_attr");
    if (ea.shape.size() != 2 || ea.shape[1] != cfg_.edge_attr_dim)
        throw std::runtime_error(ea.name + ": edge attribute shape does not match the config");
    Graph g;
    g.host = build_csr_by_dst(ei, ea, src, dst);
    std::vector<int32_t> dest(static_cast<size_t>(g.host.n_edges));
    for (int64_t row = 0; row < dst; ++row)
        for (int32_t e = g.host.row_ptr[row]; e < g.host.row_ptr[row + 1]; ++e) dest[e] = u(row);
    g.rows = upload(g.host.row_ptr);
    g.sources = upload(g.host.src);
    g.destinations = upload(dest);
    g.attrs = upload(g.host.attr);
    return g;
}

MetalBackend::MetalBackend(const Config& cfg, SafeTensorsFile& f, const Options& opts)
    : cfg_(cfg), C_(cfg.num_channels), G_(cfg.n_data_nodes), H_(cfg.n_hidden_nodes), F_(cfg.src_features()),
      chunk_(8192) {
    @autoreleasepool {
        if (opts.precision != Precision::FP32) throw std::runtime_error("the Metal backend runs fp32 only");
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_ || !device_.hasUnifiedMemory)
            throw std::runtime_error("no unified-memory Metal GPU is available; use --device cpu");
        queue_ = [device_ newCommandQueue];
        if (!queue_) throw std::runtime_error("cannot create a Metal command queue");
        MTLCompileOptions* compile = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            compile.mathMode = MTLMathModeSafe;
            compile.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            compile.fastMathEnabled = NO;  // compatibility with macOS before 15
#pragma clang diagnostic pop
        }
#else
        compile.fastMathEnabled = NO;
#endif
        compile.languageVersion = MTLLanguageVersion2_3;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:metal::kernel_source]
                                                    options:compile error:&error];
        if (!library) throw std::runtime_error("Metal shader compilation: " + std::string(error.localizedDescription.UTF8String));
        for (const char* name : {"normalise", "assemble", "bias_add", "add_arrays", "copy_array", "layernorm",
                                 "gelu", "attention_scores", "attention_softmax", "attention_values", "residual",
                                 "bound_clamp", "bound_product", "bound_assign", "denormalise"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error("Metal pipeline " + std::string(name) + ": " + error.localizedDescription.UTF8String);
            kernels_[name] = pipeline;
        }
        if (const char* value = std::getenv("VARDA_METAL_CHUNK_ROWS")) {
            char* end = nullptr;
            const long long n = std::strtoll(value, &end, 10);
            if (!*value || *end || n < 1 || n > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("VARDA_METAL_CHUNK_ROWS must be a positive 32-bit integer");
            chunk_ = n;
        }
        chunk_ = std::min(chunk_, std::max(G_, H_));
        const float* mul = f.take("norm.mul", DType::F32, {cfg.n_data_vars}).as<float>();
        const float* add = f.take("norm.add", DType::F32, {cfg.n_data_vars}).as<float>();
        std::vector<float> mi, ai, mo, ao;
        for (int i : cfg.norm_input) mi.push_back(mul[i]), ai.push_back(add[i]);
        for (int i : cfg.norm_output) mo.push_back(mul[i]), ao.push_back(add[i]);
        mul_in_ = upload(mi); add_in_ = upload(ai); mul_out_ = upload(mo); add_out_ = upload(ao);
        auto attr = [&](const char* name, int64_t nodes) {
            const TensorView& t = f.take(name, DType::F32, {nodes, cfg.node_attr_dim});
            return bytes(t.nbytes, t.data);
        };
        data_attr_ = attr("nodes.data.attr", G_);
        hidden_attr_ = attr("nodes.hidden.attr", H_);
        enc_graph_ = load_graph(f, "enc", G_, H_);
        proc_graph_ = load_graph(f, "proc", H_, H_);
        dec_graph_ = load_graph(f, "dec", H_, G_);
        enc_src_ = load_linear(f, "enc.emb_src", C_, F_);
        enc_dst_ = load_linear(f, "enc.emb_dst", C_, cfg.node_attr_dim);
        dec_dst_ = load_linear(f, "dec.emb_dst", C_, F_);
        out_norm_ = load_norm(f, "dec.out_norm");
        out_proj_ = load_linear(f, "dec.out_proj", cfg.n_output_vars, C_);
        for (const BlockInfo& bi : cfg.blocks) {
            const bool mapper = bi.kind == "graph_transformer_mapper_block";
            if (bi.name == "enc.block") enc_ = load_block(f, bi.name, mapper);
            else if (bi.name == "dec.block") dec_ = load_block(f, bi.name, mapper);
            else processors_.push_back(load_block(f, bi.name, mapper));
        }
        residual_in_ = upload(cfg.residual_input); residual_out_ = upload(cfg.residual_output);
        int64_t max_bound = 0;
        for (const Bounding& b : cfg.bounding) {
            if (std::set<int>(b.index.begin(), b.index.end()).size() != b.index.size())
                throw std::runtime_error("Metal bounding requires unique output indices");
            bounds_.push_back({b, upload(b.index), upload(b.total_index)});
            max_bound = std::max(max_bound, static_cast<int64_t>(b.index.size()));
        }
        const int64_t max_nodes = std::max(G_, H_);
        keys_ = arena("keys", max_nodes * C_); values_ = arena("values", max_nodes * C_);
        scores_ = arena("scores", std::max({enc_graph_.host.n_edges, proc_graph_.host.n_edges,
                                           dec_graph_.host.n_edges}) * cfg.num_heads);
        ln_ = arena("ln", chunk_ * C_); q_ = arena("q", chunk_ * C_);
        self_ = arena("self", chunk_ * C_); attention_ = arena("attention", chunk_ * C_);
        residual_ = arena("residual", chunk_ * C_); hidden_ = arena("hidden", chunk_ * cfg.mlp_hidden);
        arena("input", cfg.n_input_steps * G_ * cfg.n_input_vars);
        arena("xn", cfg.n_input_steps * G_ * cfg.n_input_vars);
        arena("features", G_ * F_);
        for (const char* name : {"z", "latent", "proc.a", "proc.b", "encoder.dest"}) arena(name, H_ * C_);
        for (const char* name : {"embed", "block"}) arena(name, chunk_ * C_);
        for (const char* name : {"decoder", "output"}) arena(name, G_ * cfg.n_output_vars);
        arena("products", G_ * max_bound);
    }
}

id<MTLCommandBuffer> MetalBackend::command() {
    if (!command_) command_ = [queue_ commandBuffer];
    if (!command_) throw std::runtime_error("cannot create a Metal command buffer");
    return command_;
}
void MetalBackend::flush() {
    encoded_ops_ = 0;
    if (!command_) return;
    id<MTLCommandBuffer> pending = command_;
    command_ = nil;
    [pending commit];
    [pending waitUntilCompleted];
    if (pending.status == MTLCommandBufferStatusError)
        throw std::runtime_error("Metal execution: " + std::string(pending.error.localizedDescription.UTF8String));
}
void MetalBackend::finish_op() {
    // Bound queued API operations and retained MPS resources without stalling at
    // every model phase. This is an operation-count bound, not a GPU time limit.
    if (++encoded_ops_ >= 64) flush();
}
void MetalBackend::dispatch(const char* name, const Args& args, std::initializer_list<Tensor> buffers) {
    if (!args.n) return;
    id<MTLComputePipelineState> pipeline = kernels_.at(name);
    id<MTLComputeCommandEncoder> encoder = [command() computeCommandEncoder];
    if (!encoder) throw std::runtime_error("cannot create a Metal compute encoder");
    [encoder setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (const Tensor& t : buffers) {
        [encoder setBuffer:t.buffer offset:t.offset atIndex:index++];
    }
    [encoder setBytes:&args length:sizeof(args) atIndex:8];
    const NSUInteger simd = pipeline.threadExecutionWidth;
    const NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup) / simd * simd;
    const NSUInteger threads = std::strcmp(name, "layernorm") == 0 ? u(int64_t(args.n) * simd) : args.n;
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    finish_op();
}

void MetalBackend::linear(const Linear& l, const Tensor& x, int64_t rows, const Tensor& out) {
    auto matrix = [](const Tensor& t, int64_t r, int64_t c) {
        MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c
                                               rowBytes:c * sizeof(float) dataType:MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:t.buffer offset:t.offset descriptor:descriptor];
    };
    const auto key = std::make_tuple(rows, l.out, l.in);
    auto& gemm = gemms_[key];
    if (!gemm) gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device_ transposeLeft:NO transposeRight:YES
                           resultRows:rows resultColumns:l.out interiorColumns:l.in alpha:1.0 beta:0.0];
    if (!gemm) throw std::runtime_error("cannot create an MPS matrix multiplication");
    [gemm encodeToCommandBuffer:command() leftMatrix:matrix(x, rows, l.in)
                   rightMatrix:matrix(l.w, l.out, l.in) resultMatrix:matrix(out, rows, l.out)];
    finish_op();
    if (l.b.buffer) dispatch("bias_add", {u(rows * l.out), u(l.out)}, {out, l.b});
}
void MetalBackend::norm(const Norm& l, const Tensor& x, int64_t rows, const Tensor& out) {
    Args p{u(rows), u(C_), l.b.buffer ? 1u : 0u}; p.x = cfg_.layer_norm_eps;
    dispatch("layernorm", p, {x, l.w, l.b.buffer ? l.b : l.w, out});
}
void MetalBackend::copy(const Tensor& x, const Tensor& out, int64_t n) { dispatch("copy_array", {u(n)}, {x, out}); }
void MetalBackend::add(const Tensor& x, const Tensor& y, const Tensor& out, int64_t n) {
    dispatch("add_arrays", {u(n)}, {x, y, out});
}

void MetalBackend::source(const Block& b, const Tensor& x, int64_t rows, int64_t dest_row) {
    for (int64_t r = 0; r < rows; r += chunk_) {
        const int64_t n = std::min(chunk_, rows - r);
        norm(b.src, x.sub(r * C_), n, ln_);
        linear(b.k, ln_, n, keys_.sub((dest_row + r) * C_));
        linear(b.v, ln_, n, values_.sub((dest_row + r) * C_));
    }
}
void MetalBackend::destination(const Block& b, const Graph& g, int64_t row, int64_t rows,
                               const Tensor& x, const Tensor& out) {
    norm(b.dst, x, rows, ln_);
    linear(b.q, ln_, rows, q_); linear(b.self, ln_, rows, self_);
    Args p{0, u(C_), u(cfg_.num_heads), u(cfg_.head_dim), u(cfg_.edge_attr_dim),
           b.edge.b.buffer ? 1u : 0u, u(row), u(g.host.row_ptr[row])};
    Tensor bias = b.edge.b.buffer ? b.edge.b : b.edge.w;
    p.n = u(int64_t(g.host.row_ptr[row + rows] - g.host.row_ptr[row]) * cfg_.num_heads);
    dispatch("attention_scores", p, {q_, keys_, g.sources, g.destinations, g.attrs, b.edge.w, bias, scores_});
    p.n = u(rows * cfg_.num_heads);
    dispatch("attention_softmax", p, {g.rows, scores_});
    p.n = u(rows * C_);
    dispatch("attention_values", p, {values_, g.rows, g.sources, g.attrs, b.edge.w, bias, scores_, attention_});
    add(attention_, self_, attention_, rows * C_);
    linear(b.proj, attention_, rows, residual_);
    add(residual_, x, residual_, rows * C_);
    norm(b.mlp, residual_, rows, attention_);
    linear(b.mlp1, attention_, rows, hidden_);
    dispatch("gelu", {u(rows * cfg_.mlp_hidden), cfg_.gelu_tanh ? 1u : 0u}, {hidden_});
    linear(b.mlp2, hidden_, rows, out);
    add(out, residual_, out, rows * C_);
}

void MetalBackend::report(const ActivationHook* hook, const std::string& key, const Tensor& t,
                          std::vector<int64_t> shape) {
    if (!hook) return;
    flush();
    (*hook)(key, t.data(), shape);
}

void MetalBackend::run(const float* x, float* y, const ActivationHook* hook) {
    const int64_t T = cfg_.n_input_steps, Vi = cfg_.n_input_vars, Vo = cfg_.n_output_vars;
    Tensor input = arena_.at("input"), xn = arena_.at("xn"), features = arena_.at("features");
    std::memcpy(input.data(), x, static_cast<size_t>(T * G_ * Vi) * sizeof(float));
    if (hook) (*hook)("in.x", x, {1, T, G_, Vi});
    dispatch("normalise", {u(T * G_ * Vi), u(Vi)}, {input, mul_in_, add_in_, xn});
    report(hook, "in.x_norm", xn, {1, T, 1, G_, Vi});
    dispatch("assemble", {u(G_ * F_), u(F_), u(Vi), u(T), u(G_), u(cfg_.node_attr_dim)},
             {xn, data_attr_, features});
    report(hook, "enc.x_src", features, {G_, F_});
    report(hook, "enc.x_dst", hidden_attr_, {H_, cfg_.node_attr_dim});

    Tensor embed = arena_.at("embed"), block = arena_.at("block"), z = arena_.at("z");
    std::vector<float> enc_full(hook ? static_cast<size_t>(G_ * C_) : 0);
    for (int64_t row = 0; row < G_; row += chunk_) {
        const int64_t n = std::min(chunk_, G_ - row);
        linear(enc_src_, features.sub(row * F_), n, embed);
        if (hook) { flush(); std::memcpy(enc_full.data() + row * C_, embed.data(), n * C_ * sizeof(float)); }
        source(enc_, embed, n, row);
        if (hook) flush();
    }
    if (hook) (*hook)("enc.emb_src", enc_full.data(), {G_, C_});
    Tensor dst = arena_.at("encoder.dest");
    linear(enc_dst_, hidden_attr_, H_, dst);
    report(hook, "enc.emb_dst", dst, {H_, C_});
    for (int64_t row = 0; row < H_; row += chunk_) {
        const int64_t n = std::min(chunk_, H_ - row);
        destination(enc_, enc_graph_, row, n, dst.sub(row * C_), z.sub(row * C_));
    }
    report(hook, "enc.out", z, {H_, C_});
    Tensor state = z;
    for (int i = 0; i < cfg_.num_processor_layers; ++i) {
        Tensor next = arena_.at(i % 2 ? "proc.b" : "proc.a");
        source(processors_[i], state, H_, 0);
        for (int64_t row = 0; row < H_; row += chunk_) {
            const int64_t n = std::min(chunk_, H_ - row);
            destination(processors_[i], proc_graph_, row, n, state.sub(row * C_), next.sub(row * C_));
        }
        report(hook, "proc." + std::to_string(i) + ".out", next, {H_, C_});
        state = next;
    }
    Tensor latent = arena_.at("latent");
    if (cfg_.latent_skip) add(state, z, latent, H_ * C_); else copy(state, latent, H_ * C_);
    report(hook, "latent", latent, {H_, C_});
    source(dec_, latent, H_, 0);
    Tensor decoded = arena_.at("decoder");
    std::vector<float> dec_embed(hook ? static_cast<size_t>(G_ * C_) : 0), dec_block(dec_embed.size());
    for (int64_t row = 0; row < G_; row += chunk_) {
        const int64_t n = std::min(chunk_, G_ - row);
        linear(dec_dst_, features.sub(row * F_), n, embed);
        destination(dec_, dec_graph_, row, n, embed, block);
        norm(out_norm_, block, n, ln_);
        linear(out_proj_, ln_, n, decoded.sub(row * Vo));
        if (hook) flush();
        if (hook) {
            std::memcpy(dec_embed.data() + row * C_, embed.data(), n * C_ * sizeof(float));
            std::memcpy(dec_block.data() + row * C_, block.data(), n * C_ * sizeof(float));
        }
    }
    if (hook) {
        (*hook)("dec.emb_dst", dec_embed.data(), {G_, C_});
        (*hook)("dec.block_out", dec_block.data(), {G_, C_});
    }
    report(hook, "dec.out", decoded, {G_, Vo});
    Tensor out = arena_.at("output");
    dispatch("residual", {u(G_), u(Vo), u(Vi), u(cfg_.residual_input.size()), u(cfg_.residual_step), u(G_)},
             {xn, decoded, residual_in_, residual_out_, out});
    report(hook, "out.pre_bound", out, {1, 1, 1, G_, Vo});
    for (const Bound& b : bounds_) {
        Args p{u(G_ * b.cfg.index.size()), u(Vo), u(b.cfg.index.size()), b.cfg.type == "relu" ? 0u : 1u};
        p.x = b.cfg.min; p.y = b.cfg.max;
        dispatch("bound_clamp", p, {out, b.indices});
        if (b.cfg.type == "fraction") {
            p.c = u(b.cfg.total_index.size());
            Tensor products = arena_.at("products");
            dispatch("bound_product", p, {out, b.indices, b.totals, products});
            dispatch("bound_assign", p, {products, b.indices, out});
        }
    }
    report(hook, "out.norm", out, {1, 1, 1, G_, Vo});
    dispatch("denormalise", {u(G_ * Vo), u(Vo)}, {out, mul_out_, add_out_});
    report(hook, "out.y", out, {1, 1, 1, G_, Vo});
    flush();
    std::memcpy(y, out.data(), static_cast<size_t>(G_ * Vo) * sizeof(float));
}

void MetalBackend::predict_step(const float* x, float* y, const ActivationHook* hook) {
    @autoreleasepool {
        try { run(x, y, hook); }
        catch (...) { command_ = nil; encoded_ops_ = 0; throw; }
    }
}
}  // namespace

std::unique_ptr<Backend> make_metal_backend(const Config& cfg, SafeTensorsFile& file, const Options& opts) {
    return std::make_unique<MetalBackend>(cfg, file, opts);
}
}  // namespace varda
