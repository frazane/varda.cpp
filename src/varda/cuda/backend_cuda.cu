// CUDA backend: fp32 and fp16 (autocast-faithful) forward pass of the GraphTransformer
// encoder-processor-decoder (docs/cuda.md).
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "varda/backend.h"
#include "varda/cuda/cuda_backend.h"
#include "varda/cuda/gemm.h"
#include "varda/cuda/kernels.h"
#include "varda/graph.h"

namespace varda {
namespace cuda {

namespace {

template <class T> constexpr Elt elt_of() { return std::is_same<T, float>::value ? Elt::F32 : Elt::F16; }

int64_t env_int(const char* name, int64_t dflt) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoll(v) : dflt;
}

// Linear layer on the device: W [N, ldw] (K used columns, zero padded), bias [N] or null.
template <class T> struct Lin {
    DeviceBuffer w, b;
    int64_t N = 0, K = 0, ldw = 0;
    const T* W() const { return w.template as<T>(); }
    const T* B() const { return b.get() ? b.template as<T>() : nullptr; }
};

struct Norm {
    DeviceBuffer w, b;  // fp32 [C]
    const float* W() const { return w.as<float>(); }
    const float* B() const { return b.as<float>(); }
};

template <class T> struct Block {
    bool mapper = false;
    Norm ln_src, ln_dst, ln_mlp;
    Lin<T> qkvs;    // processor: [q | k | v | self]
    Lin<T> qs, kv;  // mapper: [q | self] on destinations, [k | v] on sources
    Lin<T> proj, mlp1, mlp2;
    DeviceBuffer w_e;  // fp32 [4, C]: W_e columns, then b_e
};

struct GraphBuf {
    DeviceBuffer row_ptr, edges;
    DeviceGraph g;
};

}  // namespace

template <class T>
class Engine final : public CudaBackend {
public:
    Engine(const Config& cfg, SafeTensorsFile& file, const Options& opts);
    ~Engine() override;

    void predict_step(const float* x, float* y, const ActivationHook* hook) override;

    int64_t input_numel() const override { return int64_t(T_in_) * G_ * V_in_; }
    int64_t output_numel() const override { return G_ * V_out_; }
    void normalise(const float* x, float* xn, cudaStream_t s) override {
        cuda::normalise(x, mul_in_.as<float>(), add_in_.as<float>(), xn, input_numel(), V_in_, s);
    }
    void forward_normalised(const float* xn, float* yn, cudaStream_t s) override { forward(xn, yn, s, nullptr); }
    void denormalise(const float* yn, float* y, cudaStream_t s) override {
        cuda::denormalise(yn, add_out_.as<float>(), mul_out_.as<float>(), y, output_numel(), V_out_, s);
    }
    void predict_step_device(const float* x, float* y, cudaStream_t s) override {
        normalise(x, xn_, s);
        forward(xn_, yn_, s, nullptr);
        denormalise(yn_, y, s);
    }
    void prepare() override;
    int device() const override { return dev_; }
    size_t weight_bytes() const override { return weight_bytes_; }
    size_t activation_bytes() const override { return act_bytes_; }
    int64_t decoder_chunk_rows() const override { return dec_rows_; }

private:
    struct Hook;

    // loading
    Lin<T> load_linear(SafeTensorsFile& f, const std::vector<std::string>& names, int64_t K_pad, int64_t N_pad = 0);
    Norm load_norm(SafeTensorsFile& f, const std::string& name);
    Block<T> load_block(SafeTensorsFile& f, const std::string& pre, bool mapper);
    GraphBuf load_graph(SafeTensorsFile& f, const std::string& g, int64_t n_src, int64_t n_dst);
    DeviceBuffer upload(const void* host, size_t bytes);
    template <class U> DeviceBuffer upload_vec(const std::vector<U>& v) { return upload(v.data(), v.size() * sizeof(U)); }
    T* act(size_t elems);

    void linear(const Lin<T>& L, const T* x, int64_t ldx, T* y, int64_t ldy, int64_t M, cudaStream_t s) {
        gemm_->linear(x, ldx, L.W(), L.ldw, L.B(), y, ldy, M, L.N, L.K, s);
    }
    void attention(const GraphBuf& G, const Block<T>& blk, int64_t lo, int64_t n, const T* q, int64_t ldq,
                   const T* k, const T* v, int64_t ldkv, const T* xr, int64_t ldxr, T* out, cudaStream_t s);
    void forward(const float* xn, float* yn, cudaStream_t s, Hook* hk);

    Config cfg_;
    Options opts_;
    int dev_ = 0;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<Gemm> gemm_;

    int C_, heads_, hd_, mlp_, n_proc_, T_in_, V_in_, V_out_, A_node_, F_, ldF_, ldA_, ldVo_;
    int64_t G_, H_;
    float eps_;
    bool gelu_tanh_;

    // weights and graphs
    Lin<T> enc_emb_src_, enc_emb_dst_, dec_emb_dst_, dec_out_proj_;
    Norm dec_out_norm_;
    Block<T> enc_, dec_;
    std::vector<Block<T>> proc_;
    GraphBuf g_enc_, g_proc_, g_dec_;
    DeviceBuffer mul_in_, add_in_, mul_out_, add_out_, res_src_, bounds_, bidx_;
    int n_bounds_ = 0;
    DeviceBuffer d_attr_, h_attr_;  // fp32 [G, A] (assembly), T [H, ldA] (encoder emb_dst input)
    const float* h_attr_host_ = nullptr;
    size_t weight_bytes_ = 0;

    // activations
    std::vector<DeviceBuffer> acts_;
    size_t act_bytes_ = 0;
    float *x_dev_, *xn_, *yn_, *y_dev_;
    T* x_data_;
    T *r1_;  // shared region: encoder source side, then decoder destination chunks
    T *z_enc_, *zA_, *zB_, *xh_, *qkvs_, *sbuf_, *ybuf_, *out1_, *hbuf_;
    T* dec_out_;
    int64_t dec_rows_ = 0;
    int att_vpl_ = static_cast<int>(env_int("VARDA_ATT_VPL", 1));

    // hook scratch (lazily allocated, hooked runs only)
    DeviceBuffer hook_stage_, hook_xsrc_, hook_prebound_;

    // CUDA graph of predict_step_device on x_dev_ -> y_dev_
    bool ran_eager_ = false;
    cudaGraphExec_t graph_exec_ = nullptr;
};

// ====================================================================== hook

// Copies activations to the host as fp32 (synchronising the stream) and reports them. Hooked runs
// launch exactly the kernels of hook-free runs, plus copies.
template <class T>
struct Engine<T>::Hook {
    Engine& e;
    const ActivationHook& fn;
    cudaStream_t s;

    void host(const std::string& key, const float* data, std::vector<int64_t> shape) { fn(key, data, shape); }

    // Device [rows, cols] (leading dimension ld) of type U -> host rows [r0, r0 + rows) of dst (cols wide).
    template <class U> void fetch(const U* d, int64_t ld, int64_t rows, int64_t cols, float* dst, int64_t r0 = 0) {
        VARDA_CUDA_CHECK(cudaStreamSynchronize(s));
        if (std::is_same<U, float>::value) {
            VARDA_CUDA_CHECK(cudaMemcpy2D(dst + r0 * cols, cols * sizeof(float), d, ld * sizeof(float),
                                          cols * sizeof(float), rows, cudaMemcpyDeviceToHost));
            return;
        }
        const int64_t stage_elems = int64_t(1) << 25;  // 128 MB
        if (!e.hook_stage_.get()) e.hook_stage_.alloc(stage_elems * sizeof(float));
        const int64_t blk = std::max<int64_t>(1, stage_elems / cols);
        for (int64_t a = 0; a < rows; a += blk) {
            const int64_t n = std::min(blk, rows - a);
            convert_2d<U, float>(d + a * ld, ld, e.hook_stage_.as<float>(), cols, n, cols, s);
            VARDA_CUDA_CHECK(cudaMemcpyAsync(dst + (r0 + a) * cols, e.hook_stage_.get(), n * cols * sizeof(float),
                                             cudaMemcpyDeviceToHost, s));
            VARDA_CUDA_CHECK(cudaStreamSynchronize(s));
        }
    }
    template <class U> void dev(const std::string& key, const U* d, int64_t ld, int64_t rows, int64_t cols,
                                std::vector<int64_t> shape) {
        std::unique_ptr<float[]> h(new float[rows * cols]);
        fetch(d, ld, rows, cols, h.get());
        host(key, h.get(), shape);
    }
};

// ====================================================================== loading

template <class T>
DeviceBuffer Engine<T>::upload(const void* host, size_t bytes) {
    DeviceBuffer b(bytes);
    if (bytes) VARDA_CUDA_CHECK(cudaMemcpy(b.get(), host, bytes, cudaMemcpyHostToDevice));
    weight_bytes_ += bytes;
    return b;
}

// Rows of several Linear weights stacked ([N_total, K]), stored as T with K padded to K_pad (zero
// columns) and N to N_pad (zero rows).
template <class T>
Lin<T> Engine<T>::load_linear(SafeTensorsFile& f, const std::vector<std::string>& names, int64_t K_pad,
                              int64_t N_pad) {
    Lin<T> L;
    std::vector<const TensorView*> ws;
    for (const auto& n : names) {
        const TensorView& w = f.take(n + ".weight");
        if (w.dtype != DType::F32 || w.shape.size() != 2) throw std::runtime_error(n + ".weight: expected F32 [out, in]");
        if (!ws.empty() && w.dim(1) != ws[0]->dim(1)) throw std::runtime_error(n + ".weight: input size mismatch");
        ws.push_back(&w);
        L.N += w.dim(0);
    }
    L.K = ws[0]->dim(1);
    L.ldw = std::max<int64_t>(K_pad, L.K);
    L.N = std::max<int64_t>(L.N, N_pad);  // extra output rows: zero weights and bias
    std::vector<float> host(L.N * L.ldw, 0.0f), bias(L.N, 0.0f);
    bool any_bias = false;
    int64_t row = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const TensorView& w = *ws[i];
        for (int64_t r = 0; r < w.dim(0); ++r)
            std::memcpy(&host[(row + r) * L.ldw], w.as<float>() + r * L.K, L.K * sizeof(float));
        const std::string bn = names[i] + ".bias";
        if (!cfg_.is_absent(bn)) {
            const TensorView& b = f.take(bn, DType::F32, {w.dim(0)});
            std::memcpy(&bias[row], b.as<float>(), w.dim(0) * sizeof(float));
            any_bias = true;
        }
        row += w.dim(0);
    }
    L.K = L.ldw;  // the GEMM runs over the zero-padded K (exact: zero products)
    DeviceBuffer tmp(host.size() * sizeof(float));
    VARDA_CUDA_CHECK(cudaMemcpy(tmp.get(), host.data(), tmp.bytes(), cudaMemcpyHostToDevice));
    L.w.alloc(host.size() * sizeof(T));
    convert_2d<float, T>(tmp.as<float>(), L.ldw, L.w.template as<T>(), L.ldw, L.N, L.ldw, nullptr);
    weight_bytes_ += L.w.bytes();
    if (any_bias) {
        VARDA_CUDA_CHECK(cudaMemcpy(tmp.get(), bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
        L.b.alloc(bias.size() * sizeof(T));
        convert_2d<float, T>(tmp.as<float>(), L.N, L.b.template as<T>(), L.N, 1, L.N, nullptr);
        weight_bytes_ += L.b.bytes();
    }
    VARDA_CUDA_CHECK(cudaDeviceSynchronize());
    return L;
}

template <class T>
Norm Engine<T>::load_norm(SafeTensorsFile& f, const std::string& name) {
    Norm n;
    const TensorView& w = f.take(name + ".weight", DType::F32, {C_});
    n.w = upload(w.data, w.nbytes);
    std::vector<float> b(C_, 0.0f);
    if (!cfg_.is_absent(name + ".bias")) {
        const TensorView& bv = f.take(name + ".bias", DType::F32, {C_});
        std::memcpy(b.data(), bv.data, bv.nbytes);
    }
    n.b = upload_vec(b);
    return n;
}

namespace {
// What autocast does to the edge Linear's operands in fp16 mode: round to fp16 (identity in fp32).
template <class T> float round_op(float x) {
    if (std::is_same<T, float>::value) return x;
    return __half2float(__float2half_rn(x));
}
}  // namespace

template <class T>
Block<T> Engine<T>::load_block(SafeTensorsFile& f, const std::string& pre, bool mapper) {
    Block<T> b;
    b.mapper = mapper;
    if (mapper) b.ln_src = load_norm(f, pre + ".ln_src");
    b.ln_dst = load_norm(f, pre + ".ln_dst");
    b.ln_mlp = load_norm(f, pre + ".ln_mlp");
    auto P = [&](const char* n) { return pre + "." + n; };
    if (mapper) {
        b.qs = load_linear(f, {P("q"), P("self")}, C_);
        b.kv = load_linear(f, {P("k"), P("v")}, C_);
    } else {
        b.qkvs = load_linear(f, {P("q"), P("k"), P("v"), P("self")}, C_);
    }
    b.proj = load_linear(f, {P("proj")}, C_);
    b.mlp1 = load_linear(f, {P("mlp1")}, C_);
    b.mlp2 = load_linear(f, {P("mlp2")}, mlp_);
    const int A = cfg_.edge_attr_dim;
    const TensorView& we = f.take(P("edge") + ".weight", DType::F32, {C_, A});
    // [4, C]: W_e[:, 0..2] (attribute-major, zero padded to 3 attributes), then b_e
    std::vector<float> w4(size_t(C_) * 4, 0.0f);
    for (int c = 0; c < C_; ++c)
        for (int a = 0; a < A; ++a) w4[size_t(a) * C_ + c] = round_op<T>(we.as<float>()[c * A + a]);
    if (!cfg_.is_absent(P("edge") + ".bias")) {
        const TensorView& bv = f.take(P("edge") + ".bias", DType::F32, {C_});
        for (int c = 0; c < C_; ++c) w4[size_t(3) * C_ + c] = round_op<T>(bv.as<float>()[c]);
    }
    b.w_e = upload_vec(w4);
    if (b.mlp1.N != mlp_ || b.mlp2.N != C_ || b.proj.N != C_)
        throw std::runtime_error(pre + ": unexpected block shapes");
    return b;
}

template <class T>
GraphBuf Engine<T>::load_graph(SafeTensorsFile& f, const std::string& g, int64_t n_src, int64_t n_dst) {
    const TensorView& ei = f.take("graph." + g + ".edge_index");
    const TensorView& ea = f.take("graph." + g + ".edge_attr");
    CsrGraph csr = build_csr_by_dst(ei, ea, n_src, n_dst);
    if (csr.attr_dim != cfg_.edge_attr_dim) throw std::runtime_error("graph." + g + ": edge_attr width != edge_attr_dim");
    std::vector<float4> packed(csr.n_edges);
    for (int64_t e = 0; e < csr.n_edges; ++e) {
        float a[3] = {0, 0, 0};
        for (int k = 0; k < csr.attr_dim; ++k) a[k] = round_op<T>(csr.attr[e * csr.attr_dim + k]);
        float sbits;
        std::memcpy(&sbits, &csr.src[e], sizeof(float));
        packed[e] = make_float4(a[0], a[1], a[2], sbits);
    }
    GraphBuf b;
    b.row_ptr = upload_vec(csr.row_ptr);
    b.edges = upload_vec(packed);
    b.g.n_src = n_src;
    b.g.n_dst = n_dst;
    b.g.n_edges = csr.n_edges;
    b.g.attr_dim = csr.attr_dim;
    b.g.max_degree = csr.max_degree();
    b.g.row_ptr = b.row_ptr.as<int32_t>();
    b.g.edges = b.edges.as<float4>();
    return b;
}

template <class T>
T* Engine<T>::act(size_t elems) {
    acts_.emplace_back(align_up(elems * sizeof(T), 256));
    act_bytes_ += acts_.back().bytes();
    return acts_.back().as<T>();
}

template <class T>
Engine<T>::Engine(const Config& cfg, SafeTensorsFile& file, const Options& opts) : cfg_(cfg), opts_(opts) {
    dev_ = opts.cuda_device;
    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    VARDA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    C_ = cfg.num_channels;
    heads_ = cfg.num_heads;
    hd_ = cfg.head_dim;
    mlp_ = cfg.mlp_hidden;
    n_proc_ = cfg.num_processor_layers;
    T_in_ = cfg.n_input_steps;
    V_in_ = cfg.n_input_vars;
    V_out_ = cfg.n_output_vars;
    A_node_ = cfg.node_attr_dim;
    G_ = cfg.n_data_nodes;
    H_ = cfg.n_hidden_nodes;
    eps_ = cfg.layer_norm_eps;
    gelu_tanh_ = cfg.gelu_tanh;
    F_ = cfg.src_features();
    ldF_ = static_cast<int>(align_up(F_, env_int("VARDA_KPAD", 8)));
    ldA_ = static_cast<int>(align_up(A_node_, 8));
    ldVo_ = static_cast<int>(align_up(V_out_, 8));

    if (C_ % 256 || C_ > 2048) throw std::runtime_error("CUDA backend: num_channels must be a multiple of 256, <= 2048");
    if (hd_ % 8 || (hd_ / 8) & (hd_ / 8 - 1) || hd_ > 256)
        throw std::runtime_error("CUDA backend: head_dim must be 8 * 2^k <= 256");
    if (mlp_ % 8) throw std::runtime_error("CUDA backend: mlp_hidden must be a multiple of 8");
    if (cfg.edge_attr_dim < 1 || cfg.edge_attr_dim > 3)
        throw std::runtime_error("CUDA backend: edge_attr_dim must be 1..3 (packed edge records)");
    if (n_proc_ < 1) throw std::runtime_error("CUDA backend: needs at least one processor block");
    if (cfg.n_output_steps != 1) throw std::runtime_error("CUDA backend: one output step only");

    gemm_ = std::make_unique<Gemm>(elt_of<T>(), size_t(32) << 20, env_int("VARDA_GEMM_AUTOTUNE", 0) != 0);

    // ---- normaliser, residual, bounding
    const TensorView& mul = file.take("norm.mul", DType::F32, {cfg.n_data_vars});
    const TensorView& add = file.take("norm.add", DType::F32, {cfg.n_data_vars});
    std::vector<float> mi(V_in_), ai(V_in_), mo(V_out_), ao(V_out_);
    for (int v = 0; v < V_in_; ++v) mi[v] = mul.as<float>()[cfg.norm_input[v]], ai[v] = add.as<float>()[cfg.norm_input[v]];
    for (int v = 0; v < V_out_; ++v) mo[v] = mul.as<float>()[cfg.norm_output[v]], ao[v] = add.as<float>()[cfg.norm_output[v]];
    mul_in_ = upload_vec(mi);
    add_in_ = upload_vec(ai);
    mul_out_ = upload_vec(mo);
    add_out_ = upload_vec(ao);
    std::vector<int> rs(V_out_, -1);
    for (size_t k = 0; k < cfg.residual_output.size(); ++k) rs[cfg.residual_output[k]] = cfg.residual_input[k];
    res_src_ = upload_vec(rs);
    std::vector<BoundingDev> bd;
    std::vector<int> bidx;
    for (const Bounding& b : cfg.bounding) {
        BoundingDev d;
        d.type = b.type == "relu" ? 0 : b.type == "hardtanh" ? 1 : b.type == "fraction" ? 2 : -1;
        if (d.type < 0) throw std::runtime_error("CUDA backend: unsupported bounding " + b.type);
        d.n = static_cast<int>(b.index.size());
        d.idx_off = static_cast<int>(bidx.size());
        bidx.insert(bidx.end(), b.index.begin(), b.index.end());
        d.tot_off = static_cast<int>(bidx.size());
        d.n_tot = static_cast<int>(b.total_index.size());
        bidx.insert(bidx.end(), b.total_index.begin(), b.total_index.end());
        d.lo = b.min;
        d.hi = b.max;
        for (int i : b.index)
            for (int t : b.total_index)
                if (i == t) throw std::runtime_error("CUDA backend: fraction bounding index overlaps total_index");
        bd.push_back(d);
    }
    n_bounds_ = static_cast<int>(bd.size());
    if (bidx.empty()) bidx.push_back(0);
    if (bd.empty()) bd.push_back(BoundingDev{});
    bounds_ = upload_vec(bd);
    bidx_ = upload_vec(bidx);

    // ---- node attributes
    const TensorView& da = file.take("nodes.data.attr", DType::F32, {G_, A_node_});
    const TensorView& ha = file.take("nodes.hidden.attr", DType::F32, {H_, A_node_});
    d_attr_ = upload(da.data, da.nbytes);
    h_attr_host_ = ha.as<float>();
    {
        DeviceBuffer tmp = upload(ha.data, ha.nbytes);
        h_attr_.alloc(H_ * ldA_ * sizeof(T));
        VARDA_CUDA_CHECK(cudaMemset(h_attr_.get(), 0, h_attr_.bytes()));
        convert_2d<float, T>(tmp.as<float>(), A_node_, h_attr_.as<T>(), ldA_, H_, A_node_, nullptr);
        VARDA_CUDA_CHECK(cudaDeviceSynchronize());
        weight_bytes_ += h_attr_.bytes();
        weight_bytes_ -= tmp.bytes();
    }

    // ---- weights
    enc_emb_src_ = load_linear(file, {"enc.emb_src"}, ldF_);
    enc_emb_dst_ = load_linear(file, {"enc.emb_dst"}, ldA_);
    dec_emb_dst_ = load_linear(file, {"dec.emb_dst"}, ldF_);
    dec_out_norm_ = load_norm(file, "dec.out_norm");
    dec_out_proj_ = load_linear(file, {"dec.out_proj"}, C_, ldVo_);  // N padded: aligned GEMM kernels
    if (enc_emb_src_.K != ldF_ || dec_emb_dst_.K != ldF_ || enc_emb_dst_.K != ldA_ || dec_out_proj_.N != ldVo_ ||
        file.peek("dec.out_proj.weight").dim(0) != V_out_)
        throw std::runtime_error("CUDA backend: embedding shapes do not match the config");
    enc_ = load_block(file, "enc.block", true);
    for (int i = 0; i < n_proc_; ++i) proc_.push_back(load_block(file, "proc." + std::to_string(i), false));
    dec_ = load_block(file, "dec.block", true);

    // ---- graphs (destination-major CSR, stable)
    g_enc_ = load_graph(file, "enc", G_, H_);
    g_proc_ = load_graph(file, "proc", H_, H_);
    g_dec_ = load_graph(file, "dec", H_, G_);

    // ---- activations (docs/cuda.md §5)
    const int64_t in_n = input_numel(), out_n = output_numel();
    x_dev_ = reinterpret_cast<float*>(act(in_n * sizeof(float) / sizeof(T)));
    xn_ = reinterpret_cast<float*>(act(in_n * sizeof(float) / sizeof(T)));
    yn_ = reinterpret_cast<float*>(act(out_n * sizeof(float) / sizeof(T)));
    y_dev_ = reinterpret_cast<float*>(act(out_n * sizeof(float) / sizeof(T)));
    x_data_ = act(G_ * ldF_);
    dec_out_ = act(G_ * ldVo_);
    z_enc_ = act(H_ * C_);
    zA_ = act(H_ * C_);
    zB_ = act(H_ * C_);
    xh_ = act(H_ * C_);
    qkvs_ = act(H_ * 4 * C_);
    sbuf_ = act(H_ * C_);
    ybuf_ = act(H_ * C_);
    out1_ = act(H_ * C_);
    hbuf_ = act(H_ * mlp_);
    // Shared region: the encoder's source side ([G, C] embedding + [G, 2C] k|v), then the decoder's
    // destination chunks (7C + mlp per row). Chunk rows: what fits in the encoder's region, at least
    // 2^18 (or VARDA_DEC_CHUNK_ROWS), multiple of 128 (aligned row offsets).
    const int64_t enc_elems = G_ * 3 * C_;
    const int64_t per_row = 7 * C_ + mlp_;
    int64_t rows = env_int("VARDA_DEC_CHUNK_ROWS", std::max<int64_t>(enc_elems / per_row, int64_t(1) << 18));
    rows = std::min<int64_t>(std::max<int64_t>(rows, 128), G_);
    const int64_t nchunks = (G_ + rows - 1) / rows;
    rows = std::min<int64_t>(int64_t(align_up((G_ + nchunks - 1) / nchunks, 128)), G_);
    dec_rows_ = rows;
    r1_ = act(std::max(enc_elems, rows * per_row));
}

template <class T>
Engine<T>::~Engine() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (stream_) cudaStreamDestroy(stream_);
}

// ====================================================================== forward

template <class T>
void Engine<T>::attention(const GraphBuf& G, const Block<T>& blk, int64_t lo, int64_t n, const T* q, int64_t ldq,
                          const T* k, const T* v, int64_t ldkv, const T* xr, int64_t ldxr, T* out, cudaStream_t s) {
    AttentionArgs a;
    a.g = G.g;
    a.dst_lo = lo;
    a.n_rows = n;
    a.q = q; a.ldq = ldq;
    a.k = k; a.ldk = ldkv;
    a.v = v; a.ldv = ldkv;
    a.xr = xr; a.ldxr = ldxr;
    a.out = out; a.ldo = C_;
    a.w_e = blk.w_e.template as<float>();
    a.C = C_;
    a.heads = heads_;
    a.head_dim = hd_;
    cuda::attention<T>(a, s, att_vpl_);
}

template <class T>
void Engine<T>::forward(const float* xn, float* yn, cudaStream_t s, Hook* hk) {
    const int64_t G = G_, H = H_, C = C_;

    // ---- input assembly: x_data = [xn t0 | xn t1 | node attr | 0 pad]
    assemble_x_data<T>(xn, d_attr_.as<float>(), T_in_, G, V_in_, A_node_, x_data_, ldF_, s);
    if (hk) {
        if (std::is_same<T, float>::value) {
            hk->dev("enc.x_src", x_data_, ldF_, G, F_, {G, F_});
        } else {  // the reference's x_data is fp32; the fp16 cast happens inside the Linear
            if (!hook_xsrc_.get()) hook_xsrc_.alloc(G * F_ * sizeof(float));
            assemble_x_data<float>(xn, d_attr_.as<float>(), T_in_, G, V_in_, A_node_, hook_xsrc_.as<float>(), F_, s);
            hk->dev("enc.x_src", hook_xsrc_.as<float>(), F_, G, F_, {G, F_});
        }
        hk->host("enc.x_dst", h_attr_host_, {H, A_node_});
    }

    // ---- encoder, source side (all data nodes): emb_src, LN_src (in place), [k | v]
    T* emb_src = r1_;
    T* kv_enc = r1_ + G * C;
    linear(enc_emb_src_, x_data_, ldF_, emb_src, C, G, s);
    if (hk) hk->dev("enc.emb_src", emb_src, C, G, C, {G, C});
    layernorm<T>(emb_src, C, enc_.ln_src.W(), enc_.ln_src.B(), emb_src, C, G, C_, eps_, s);
    linear(enc_.kv, emb_src, C, kv_enc, 2 * C, G, s);

    // ---- encoder, destination side (hidden nodes)
    T* emb_dst = zA_;
    linear(enc_emb_dst_, h_attr_.as<T>(), ldA_, emb_dst, C, H, s);
    if (hk) hk->dev("enc.emb_dst", emb_dst, C, H, C, {H, C});
    layernorm<T>(emb_dst, C, enc_.ln_dst.W(), enc_.ln_dst.B(), xh_, C, H, C_, eps_, s);
    linear(enc_.qs, xh_, C, qkvs_, 2 * C, H, s);
    attention(g_enc_, enc_, 0, H, qkvs_, 2 * C, kv_enc, kv_enc + C, 2 * C, qkvs_ + C, 2 * C, sbuf_, s);
    linear(enc_.proj, sbuf_, C, ybuf_, C, H, s);
    add_layernorm<T>(ybuf_, emb_dst, out1_, nullptr, nullptr, enc_.ln_mlp.W(), enc_.ln_mlp.B(), xh_, C, H, C_, eps_, s);
    linear(enc_.mlp1, xh_, C, hbuf_, mlp_, H, s);
    gelu_inplace<T>(hbuf_, H * mlp_, gelu_tanh_, s);
    linear(enc_.mlp2, hbuf_, mlp_, ybuf_, C, H, s);
    add_layernorm<T>(ybuf_, out1_, z_enc_, nullptr, nullptr, proc_[0].ln_dst.W(), proc_[0].ln_dst.B(), xh_, C, H, C_,
                     eps_, s);
    if (hk) hk->dev("enc.out", z_enc_, C, H, C, {H, C});

    // ---- processor (xh_ holds LN_dst of the block input)
    T* z_in = z_enc_;
    T* latent = nullptr;
    for (int i = 0; i < n_proc_; ++i) {
        const Block<T>& b = proc_[i];
        linear(b.qkvs, xh_, C, qkvs_, 4 * C, H, s);
        attention(g_proc_, b, 0, H, qkvs_, 4 * C, qkvs_ + C, qkvs_ + 2 * C, 4 * C, qkvs_ + 3 * C, 4 * C, sbuf_, s);
        linear(b.proj, sbuf_, C, ybuf_, C, H, s);
        add_layernorm<T>(ybuf_, z_in, out1_, nullptr, nullptr, b.ln_mlp.W(), b.ln_mlp.B(), xh_, C, H, C_, eps_, s);
        linear(b.mlp1, xh_, C, hbuf_, mlp_, H, s);
        gelu_inplace<T>(hbuf_, H * mlp_, gelu_tanh_, s);
        linear(b.mlp2, hbuf_, mlp_, ybuf_, C, H, s);
        T* z_out = z_in == zA_ ? zB_ : zA_;
        if (i + 1 < n_proc_) {
            add_layernorm<T>(ybuf_, out1_, z_out, nullptr, nullptr, proc_[i + 1].ln_dst.W(), proc_[i + 1].ln_dst.B(),
                             xh_, C, H, C_, eps_, s);
        } else if (cfg_.latent_skip) {
            // latent = proc out + enc.out, written over the block input (dead), then the decoder's LN_src
            latent = z_in;
            add_layernorm<T>(ybuf_, out1_, z_out, z_enc_, latent, dec_.ln_src.W(), dec_.ln_src.B(), xh_, C, H, C_,
                             eps_, s);
        } else {
            latent = z_out;
            add_layernorm<T>(ybuf_, out1_, z_out, nullptr, nullptr, dec_.ln_src.W(), dec_.ln_src.B(), xh_, C, H, C_,
                             eps_, s);
        }
        if (hk) hk->dev("proc." + std::to_string(i) + ".out", z_out, C, H, C, {H, C});
        z_in = z_out;
    }
    if (hk) hk->dev("latent", latent, C, H, C, {H, C});

    // ---- decoder, source side: [k | v] of LN_src(latent)
    T* kv_dec = qkvs_;
    linear(dec_.kv, xh_, C, kv_dec, 2 * C, H, s);

    // ---- decoder, destination chunks (data nodes)
    std::unique_ptr<float[]> h_emb, h_bo;
    if (hk) {
        h_emb.reset(new float[G * C]);
        h_bo.reset(new float[G * C]);
    }
    const int64_t R = dec_rows_;
    T* e = r1_;
    T* t = e + R * C;
    T* qs = t + R * C;
    T* sb = qs + R * 2 * C;  // attention output, later the block output
    T* y = sb + R * C;
    T* o1 = y + R * C;
    T* h = o1 + R * C;
    for (int64_t lo = 0; lo < G; lo += R) {
        const int64_t n = std::min(R, G - lo);
        linear(dec_emb_dst_, x_data_ + lo * ldF_, ldF_, e, C, n, s);
        if (hk) hk->fetch(e, C, n, C, h_emb.get(), lo);
        layernorm<T>(e, C, dec_.ln_dst.W(), dec_.ln_dst.B(), t, C, n, C_, eps_, s);
        linear(dec_.qs, t, C, qs, 2 * C, n, s);
        attention(g_dec_, dec_, lo, n, qs, 2 * C, kv_dec, kv_dec + C, 2 * C, qs + C, 2 * C, sb, s);
        linear(dec_.proj, sb, C, y, C, n, s);
        add_layernorm<T>(y, e, o1, nullptr, nullptr, dec_.ln_mlp.W(), dec_.ln_mlp.B(), t, C, n, C_, eps_, s);
        linear(dec_.mlp1, t, C, h, mlp_, n, s);
        gelu_inplace<T>(h, n * mlp_, gelu_tanh_, s);
        linear(dec_.mlp2, h, mlp_, y, C, n, s);
        add_layernorm<T>(y, o1, sb, nullptr, nullptr, dec_out_norm_.W(), dec_out_norm_.B(), t, C, n, C_, eps_, s);
        if (hk) hk->fetch(sb, C, n, C, h_bo.get(), lo);
        linear(dec_out_proj_, t, C, dec_out_ + lo * ldVo_, ldVo_, n, s);
    }
    if (hk) {
        hk->host("dec.emb_dst", h_emb.get(), {G, C});
        h_emb.reset();
        hk->host("dec.block_out", h_bo.get(), {G, C});
        h_bo.reset();
        hk->dev("dec.out", dec_out_, ldVo_, G, V_out_, {G, int64_t(V_out_)});
    }

    // ---- output: dec.out -> fp32, + residual (normalised input step), bounding
    OutputArgs o;
    o.dec = dec_out_;
    o.ld_dec = ldVo_;
    o.xn_step = xn + int64_t(cfg_.residual_step) * G * V_in_;
    o.V_in = V_in_;
    o.V_out = V_out_;
    o.G = G;
    o.res_src = res_src_.as<int>();
    o.bounds = bounds_.as<BoundingDev>();
    o.n_bounds = n_bounds_;
    o.bidx = bidx_.as<int>();
    if (hk) {
        if (!hook_prebound_.get()) hook_prebound_.alloc(G * V_out_ * sizeof(float));
        o.pre_bound = hook_prebound_.as<float>();
    }
    o.y = yn;
    output_assemble<T>(o, s);
    if (hk) {
        const std::vector<int64_t> shp = {1, 1, 1, G, V_out_};
        hk->dev("out.pre_bound", hook_prebound_.as<float>(), V_out_, G, V_out_, shp);
        hk->dev("out.norm", yn, V_out_, G, V_out_, shp);
    }
}

template <class T>
void Engine<T>::prepare() {
    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    if (ran_eager_) return;
    VARDA_CUDA_CHECK(cudaMemsetAsync(x_dev_, 0, input_numel() * sizeof(float), stream_));
    predict_step_device(x_dev_, y_dev_, stream_);
    VARDA_CUDA_CHECK(cudaStreamSynchronize(stream_));
    ran_eager_ = true;
}

template <class T>
void Engine<T>::predict_step(const float* x, float* y, const ActivationHook* hook) {
    VARDA_CUDA_CHECK(cudaSetDevice(dev_));
    cudaStream_t s = stream_;
    const int64_t in_n = input_numel(), out_n = output_numel();
    VARDA_CUDA_CHECK(cudaMemcpyAsync(x_dev_, x, in_n * sizeof(float), cudaMemcpyHostToDevice, s));
    if (hook) {
        Hook hk{*this, *hook, s};
        hk.host("in.x", x, {1, T_in_, G_, V_in_});
        normalise(x_dev_, xn_, s);
        hk.dev("in.x_norm", xn_, in_n, 1, in_n, {1, T_in_, 1, G_, V_in_});
        forward(xn_, yn_, s, &hk);
        denormalise(yn_, y_dev_, s);
        VARDA_CUDA_CHECK(cudaMemcpyAsync(y, y_dev_, out_n * sizeof(float), cudaMemcpyDeviceToHost, s));
        VARDA_CUDA_CHECK(cudaStreamSynchronize(s));
        hk.host("out.y", y, {1, 1, 1, G_, V_out_});
        ran_eager_ = true;
        return;
    }
    if (!opts_.cuda_graph || !ran_eager_) {
        predict_step_device(x_dev_, y_dev_, s);
        ran_eager_ = true;
    } else {
        if (!graph_exec_) {
            cudaGraph_t g;
            VARDA_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            predict_step_device(x_dev_, y_dev_, s);
            VARDA_CUDA_CHECK(cudaStreamEndCapture(s, &g));
            VARDA_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, g, 0));
            cudaGraphDestroy(g);
        }
        VARDA_CUDA_CHECK(cudaGraphLaunch(graph_exec_, s));
    }
    VARDA_CUDA_CHECK(cudaMemcpyAsync(y, y_dev_, out_n * sizeof(float), cudaMemcpyDeviceToHost, s));
    VARDA_CUDA_CHECK(cudaStreamSynchronize(s));
}

template class Engine<float>;
template class Engine<__half>;

CudaBackend* as_cuda(Model& m) { return dynamic_cast<CudaBackend*>(&m.backend()); }

}  // namespace cuda

std::unique_ptr<Backend> make_cuda_backend(const Config& cfg, SafeTensorsFile& file, const Options& opts) {
    if (opts.precision == Precision::FP32) return std::make_unique<cuda::Engine<float>>(cfg, file, opts);
    return std::make_unique<cuda::Engine<__half>>(cfg, file, opts);
}

}  // namespace varda
