// CPU fp32 backend: the readable oracle (docs/cpu.md).
//
// predict_step, per docs/format.md §2 and $IE_ROOT/docs/spec-{interface,model}.md:
//   xn     = x * mul[norm_input] + add[norm_input]          two fp32 roundings
//   x_src  = [xn[t0] | xn[t1] | ... | data attr]             (G, T*V_in + A)
//   z      = encoder(x_src, hidden attr)                     (H, C)
//   z      = proc_{n-1}(...proc_0(z)) (+ z if latent_skip)   (H, C)
//   d      = decoder(z, x_src)                               (G, V_out): block + LN + Linear
//   y      = d, + xn[residual_step] on the prognostic outputs; bounding (config order)
//   Y      = (y - add[norm_output]) / mul[norm_output]
//
// Every block (encoder, processor, decoder) runs as a source side (k|v for all source
// rows) and a destination side processed in chunks of complete destination rows (each
// chunk holds all incoming edges of its destinations, so softmax groups are whole). The
// lin_edge projections are recomputed per destination inside the attention kernel and
// never stored for the whole graph.
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "varda/backend.h"
#include "varda/cpu/kernels.h"
#include "varda/graph.h"

namespace varda {
namespace {

using cpu::EdgeLinear;
using cpu::LayerNorm;
using cpu::Linear;

// Uninitialised fp32 buffer: pages are first touched by the (parallel) kernel that fills it.
std::unique_ptr<float[]> alloc(int64_t n) { return std::unique_ptr<float[]>(new float[static_cast<size_t>(n)]); }

// Named buffers kept across predict_step calls. Allocating the ~20 GB of Varda activations
// afresh in every call made the kernel page-fault (and zero) them again each time, which
// cost seconds and, on a node under memory pressure, tens of seconds (docs/cpu.md). A buffer
// grows to the largest size requested under its name; buffers whose lifetimes do not
// overlap share a name.
class Arena {
public:
    float* get(const std::string& name, int64_t n) {
        Slot& s = slots_[name];
        if (s.n < n) {
            s.p.reset();
            s.p = alloc(n);
            s.n = n;
        }
        return s.p.get();
    }
private:
    struct Slot {
        std::unique_ptr<float[]> p;
        int64_t n = 0;
    };
    std::map<std::string, Slot> slots_;
};

struct Block {
    Linear q, k, v, self, proj, mlp1, mlp2;
    EdgeLinear edge;
    std::vector<float> edge_wt;  // lin_edge weight transposed to [n_in, n_out]
    LayerNorm ln_src, ln_dst, ln_mlp;  // processor blocks: ln_src is ln_dst
};

// Destination-side scratch for up to `rows` destination rows.
struct Workspace {
    int64_t rows = 0;
    float *ln = nullptr, *qs = nullptr, *att = nullptr, *hid = nullptr;
    Workspace(Arena& a, int64_t r, int64_t C, int64_t mlp)
        : rows(r), ln(a.get("ws.ln", r * C)), qs(a.get("ws.qs", r * 2 * C)), att(a.get("ws.att", r * C)),
          hid(a.get("ws.hid", r * mlp)) {}
};

// Optional wall-clock profile (VARDA_CPU_PROFILE=1): seconds per phase, printed per call.
struct Profile {
    bool on = false;
    std::vector<std::pair<std::string, double>> t;
    void add(const std::string& name, double s) {
        if (!on) return;
        for (auto& p : t)
            if (p.first == name) { p.second += s; return; }
        t.emplace_back(name, s);
    }
};

class Timer {
public:
    Timer(Profile& p, std::string name) : p_(p), name_(std::move(name)), t0_(omp_get_wtime()) {}
    ~Timer() { p_.add(name_, omp_get_wtime() - t0_); }
private:
    Profile& p_;
    std::string name_;
    double t0_;
};

// One output bounding (anemoi layers/bounding.py), in place on (G, Vo) normalised outputs.
void apply_bounding(const Bounding& b, float* y, int64_t G, int64_t Vo) {
    const size_t n = b.index.size();
    const bool relu = b.type == "relu", fraction = b.type == "fraction";
#pragma omp parallel
    {
        std::vector<float> prod(n);
#pragma omp for schedule(static)
        for (int64_t g = 0; g < G; ++g) {
            float* yr = y + g * Vo;
            if (relu) {
                for (int i : b.index) yr[i] = yr[i] > 0.0f ? yr[i] : 0.0f;
                continue;
            }
            // hardtanh and fraction: clamp to [min, max].
            for (int i : b.index) yr[i] = std::min(std::max(yr[i], b.min), b.max);
            if (fraction) {
                // y[..., index] *= y[..., total_index] (already bounded); the right-hand side is read first.
                for (size_t k = 0; k < n; ++k)
                    prod[k] = yr[b.index[k]] * yr[b.total_index[b.total_index.size() == 1 ? 0 : k]];
                for (size_t k = 0; k < n; ++k) yr[b.index[k]] = prod[k];
            }
        }
    }
}

class CpuBackend final : public Backend {
public:
    CpuBackend(const Config& cfg, SafeTensorsFile& f, const Options& opts);
    void predict_step(const float* x, float* y, const ActivationHook* hook) override;

private:
    void report(const ActivationHook* hook, const char* key, const float* data, std::vector<int64_t> shape) const {
        if (hook) (*hook)(key, data, shape);
    }
    // Source side: kv[j] = [k | v](LN_src(xs[j])) for ns rows, kv row stride 2C.
    void source_side(const Block& b, const float* xs, int64_t ns, float* kv, Workspace& ws);
    // Destination side for rows [d0, d0 + nd) (nd <= ws.rows): xd, out are local (nd, C).
    void destination_side(const Block& b, const CsrGraph& g, const float* kv, int64_t d0, int64_t nd,
                          const float* xd, float* out, Workspace& ws);
    void encoder(const float* x_src, float* z, const ActivationHook* hook);
    void processor(const float* z_enc, float* latent, const ActivationHook* hook);
    void decoder(const float* latent, const float* x_src, float* dec_out, const ActivationHook* hook);

    Linear linear(SafeTensorsFile& f, const std::string& name, int64_t n_out, int64_t n_in);
    LayerNorm layer_norm(SafeTensorsFile& f, const std::string& name, int64_t dim);
    void load_block(SafeTensorsFile& f, const std::string& p, bool mapper, Block& b);
    const float* mapped(const TensorView& t);  // records a weight used in place in the mapping
    void prefault_mapped() const;

    Config cfg_;
    int threads_ = 0;
    int64_t chunk_rows_ = 65536;
    int64_t C_ = 0, G_ = 0, H_ = 0;
    int F_src_ = 0;
    std::vector<float> mul_in_, add_in_, mul_out_, add_out_;
    const float* data_attr_ = nullptr;
    const float* hidden_attr_ = nullptr;
    CsrGraph enc_g_, proc_g_, dec_g_;
    Linear enc_emb_src_, enc_emb_dst_, dec_emb_dst_, dec_out_proj_;
    LayerNorm dec_out_norm_;
    Block enc_, dec_;
    std::vector<Block> proc_;
    Profile prof_;
    Arena arena_;  // activations, kept across calls
    std::vector<std::pair<const void*, size_t>> mapped_;
};

Linear CpuBackend::linear(SafeTensorsFile& f, const std::string& name, int64_t n_out, int64_t n_in) {
    Linear L;
    L.w = mapped(f.take(name + ".weight", DType::F32, {n_out, n_in}));
    if (!cfg_.is_absent(name + ".bias")) L.b = mapped(f.take(name + ".bias", DType::F32, {n_out}));
    L.n_out = n_out;
    L.n_in = n_in;
    return L;
}

LayerNorm CpuBackend::layer_norm(SafeTensorsFile& f, const std::string& name, int64_t dim) {
    LayerNorm N;
    N.w = mapped(f.take(name + ".weight", DType::F32, {dim}));
    if (!cfg_.is_absent(name + ".bias")) N.b = mapped(f.take(name + ".bias", DType::F32, {dim}));
    N.dim = dim;
    N.eps = cfg_.layer_norm_eps;
    return N;
}

void CpuBackend::load_block(SafeTensorsFile& f, const std::string& p, bool mapper, Block& b) {
    const int64_t C = C_, M = cfg_.mlp_hidden, A = cfg_.edge_attr_dim;
    b.q = linear(f, p + ".q", C, C);
    b.k = linear(f, p + ".k", C, C);
    b.v = linear(f, p + ".v", C, C);
    b.self = linear(f, p + ".self", C, C);
    b.proj = linear(f, p + ".proj", C, C);
    b.mlp1 = linear(f, p + ".mlp1", M, C);
    b.mlp2 = linear(f, p + ".mlp2", C, M);
    const Linear e = linear(f, p + ".edge", C, A);
    b.edge_wt.resize(static_cast<size_t>(A * C));
    for (int64_t c = 0; c < C; ++c)
        for (int64_t a = 0; a < A; ++a) b.edge_wt[a * C + c] = e.w[c * A + a];
    b.edge.wt = b.edge_wt.data();
    b.edge.b = e.b;
    b.edge.n_in = A;
    b.edge.n_out = C;
    b.ln_dst = layer_norm(f, p + ".ln_dst", C);
    b.ln_mlp = layer_norm(f, p + ".ln_mlp", C);
    b.ln_src = mapper ? layer_norm(f, p + ".ln_src", C) : b.ln_dst;
}

CpuBackend::CpuBackend(const Config& cfg, SafeTensorsFile& f, const Options& opts) : cfg_(cfg) {
    threads_ = opts.num_threads > 0 ? opts.num_threads : omp_get_max_threads();
    omp_set_num_threads(threads_);
    cpu::set_blas_single_threaded();
    if (const char* s = std::getenv("VARDA_CPU_CHUNK_ROWS")) chunk_rows_ = std::max<int64_t>(1, std::atoll(s));
    if (const char* s = std::getenv("VARDA_CPU_PROFILE")) prof_.on = std::atoi(s) != 0;

    C_ = cfg.num_channels;
    G_ = cfg.n_data_nodes;
    H_ = cfg.n_hidden_nodes;
    F_src_ = cfg.src_features();
    const int64_t A = cfg.node_attr_dim;

    const float* mul = f.take("norm.mul", DType::F32, {cfg.n_data_vars}).as<float>();
    const float* add = f.take("norm.add", DType::F32, {cfg.n_data_vars}).as<float>();
    for (int i : cfg.norm_input) mul_in_.push_back(mul[i]), add_in_.push_back(add[i]);
    for (int i : cfg.norm_output) mul_out_.push_back(mul[i]), add_out_.push_back(add[i]);
    data_attr_ = mapped(f.take("nodes.data.attr", DType::F32, {G_, A}));
    hidden_attr_ = mapped(f.take("nodes.hidden.attr", DType::F32, {H_, A}));

    auto graph = [&](const std::string& name, int64_t n_src, int64_t n_dst) {
        const TensorView& ei = f.take("graph." + name + ".edge_index");
        const TensorView& ea = f.take("graph." + name + ".edge_attr");
        if (ea.shape.size() != 2 || ea.shape[1] != cfg.edge_attr_dim)
            throw std::runtime_error(ea.name + ": expected [E, " + std::to_string(cfg.edge_attr_dim) + "], got " +
                                     ea.shape_str());
        return build_csr_by_dst(ei, ea, n_src, n_dst);
    };
    enc_g_ = graph("enc", G_, H_);
    proc_g_ = graph("proc", H_, H_);
    dec_g_ = graph("dec", H_, G_);

    enc_emb_src_ = linear(f, "enc.emb_src", C_, F_src_);
    enc_emb_dst_ = linear(f, "enc.emb_dst", C_, A);
    dec_emb_dst_ = linear(f, "dec.emb_dst", C_, F_src_);
    dec_out_norm_ = layer_norm(f, "dec.out_norm", C_);
    dec_out_proj_ = linear(f, "dec.out_proj", static_cast<int64_t>(cfg.n_output_steps) * cfg.n_output_vars, C_);

    proc_.reserve(cfg.num_processor_layers);  // Block::edge points into its own edge_wt
    for (const BlockInfo& bi : cfg.blocks) {
        const bool mapper = bi.kind == "graph_transformer_mapper_block";
        if (bi.name == "enc.block") load_block(f, bi.name, mapper, enc_);
        else if (bi.name == "dec.block") load_block(f, bi.name, mapper, dec_);
        else {
            proc_.emplace_back();
            load_block(f, bi.name, mapper, proc_.back());
        }
    }
    if (static_cast<int>(proc_.size()) != cfg.num_processor_layers)
        throw std::runtime_error("CPU backend: processor block count mismatch");
    // The graphs and lin_edge weights were copied; every other weight is used in place in the
    // mapping. Fault it in now (in parallel), so the first predict_step does not page it in
    // from the file system one page fault at a time.
    prefault_mapped();
}

const float* CpuBackend::mapped(const TensorView& t) {
    mapped_.emplace_back(t.data, t.nbytes);
    return t.as<float>();
}

void CpuBackend::prefault_mapped() const {
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    for (const auto& r : mapped_) {
        const uintptr_t lo = reinterpret_cast<uintptr_t>(r.first) & ~(page - 1);
        const uintptr_t hi = reinterpret_cast<uintptr_t>(r.first) + r.second;
        madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_WILLNEED);  // asynchronous read-ahead hint
    }
    unsigned sum = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : sum)
    for (size_t i = 0; i < mapped_.size(); ++i) {
        const volatile unsigned char* p = static_cast<const unsigned char*>(mapped_[i].first);
        for (size_t o = 0; o < mapped_[i].second; o += page) sum += p[o];
    }
    (void)sum;
}

void CpuBackend::source_side(const Block& b, const float* xs, int64_t ns, float* kv, Workspace& ws) {
    const int64_t C = C_;
    for (int64_t r0 = 0; r0 < ns; r0 += ws.rows) {
        const int64_t n = std::min(ws.rows, ns - r0);
        float* ln = ws.ln;
        {
            Timer t(prof_, "layernorm");
            cpu::layer_norm(b.ln_src, xs + r0 * C, C, n, ln, C);
        }
        Timer t(prof_, "gemm k|v");
        cpu::linear(b.k, ln, C, n, kv + r0 * 2 * C, 2 * C);
        cpu::linear(b.v, ln, C, n, kv + r0 * 2 * C + C, 2 * C);
    }
}

void CpuBackend::destination_side(const Block& b, const CsrGraph& g, const float* kv, int64_t d0, int64_t nd,
                                  const float* xd, float* out, Workspace& ws) {
    const int64_t C = C_, M = cfg_.mlp_hidden;
    float* ln = ws.ln;    // LN_dst(xd), then h = proj(a + x_r) + xd
    float* qs = ws.qs;    // [q | x_r]
    float* att = ws.att;  // attention, then LN_mlp(h)
    float* hid = ws.hid;  // GELU(mlp1(LN_mlp(h)))
    {
        Timer t(prof_, "layernorm");
        cpu::layer_norm(b.ln_dst, xd, C, nd, ln, C);
    }
    {
        Timer t(prof_, "gemm q|self");
        cpu::linear(b.q, ln, C, nd, qs, 2 * C);
        cpu::linear(b.self, ln, C, nd, qs + C, 2 * C);
    }
    {
        Timer t(prof_, "attention");
        cpu::graph_attention(g, d0, nd, qs, 2 * C, kv, kv + C, 2 * C, b.edge, cfg_.num_heads, cfg_.head_dim, att, C);
    }
    {
        Timer t(prof_, "elementwise");
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < nd; ++r) {
            float* a = att + r * C;
            const float* s = qs + r * 2 * C + C;
            for (int64_t c = 0; c < C; ++c) a[c] = a[c] + s[c];  // a + x_r
        }
    }
    {
        Timer t(prof_, "gemm proj");
        cpu::linear(b.proj, att, C, nd, ln, C, [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) ln[r * C + c] = ln[r * C + c] + xd[r * C + c];  // proj(...) + xd
        });
    }
    {
        Timer t(prof_, "layernorm");
        cpu::layer_norm(b.ln_mlp, ln, C, nd, att, C);
    }
    {
        Timer t(prof_, "gemm mlp1+gelu");
        cpu::linear(b.mlp1, att, C, nd, hid, M, [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
            cpu::gelu_block(hid + r0 * M + c0, r1 - r0, c1 - c0, M, cfg_.gelu_tanh);
        });
    }
    {
        Timer t(prof_, "gemm mlp2");
        cpu::linear(b.mlp2, hid, M, nd, out, C, [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) out[r * C + c] = out[r * C + c] + ln[r * C + c];  // mlp(LN(h)) + h
        });
    }
}

void CpuBackend::encoder(const float* x_src, float* z, const ActivationHook* hook) {
    const int64_t C = C_, G = G_, H = H_;
    Workspace ws(arena_, std::min(chunk_rows_, std::max(G, H)), C, cfg_.mlp_hidden);

    // Source side on every data row: emb_src, LN_src, k|v. The anemoi mapper embeds only the
    // connected rows of each destination chunk; per row the maths is the same.
    float* kv = arena_.get("kv", G * 2 * C);
    float* emb = arena_.get("chunk.a", ws.rows * C);
    std::unique_ptr<float[]> emb_full = hook ? alloc(G * C) : nullptr;  // contract enc.emb_src
    for (int64_t r0 = 0; r0 < G; r0 += ws.rows) {
        const int64_t n = std::min(ws.rows, G - r0);
        {
            Timer t(prof_, "gemm embed");
            cpu::linear(enc_emb_src_, x_src + r0 * F_src_, F_src_, n, emb, C);
        }
        if (emb_full) std::memcpy(emb_full.get() + r0 * C, emb, sizeof(float) * n * C);
        source_side(enc_, emb, n, kv + r0 * 2 * C, ws);
    }
    report(hook, "enc.emb_src", emb_full.get(), {G, C});
    emb_full.reset();

    float* xd = arena_.get("enc.emb_dst", H * C);
    {
        Timer t(prof_, "gemm embed");
        cpu::linear(enc_emb_dst_, hidden_attr_, cfg_.node_attr_dim, H, xd, C);
    }
    report(hook, "enc.emb_dst", xd, {H, C});
    for (int64_t d0 = 0; d0 < H; d0 += ws.rows) {
        const int64_t n = std::min(ws.rows, H - d0);
        destination_side(enc_, enc_g_, kv, d0, n, xd + d0 * C, z + d0 * C, ws);
    }
    report(hook, "enc.out", z, {H, C});
}

void CpuBackend::processor(const float* z_enc, float* latent, const ActivationHook* hook) {
    const int64_t C = C_, H = H_;
    Workspace ws(arena_, std::min(chunk_rows_, H), C, cfg_.mlp_hidden);
    float* kv = arena_.get("kv", H * 2 * C);
    float* buf[2] = {arena_.get("proc.a", H * C), arena_.get("proc.b", H * C)};
    const float* x = z_enc;
    for (int i = 0; i < cfg_.num_processor_layers; ++i) {
        float* y = buf[i % 2];
        source_side(proc_[i], x, H, kv, ws);
        for (int64_t d0 = 0; d0 < H; d0 += ws.rows) {
            const int64_t n = std::min(ws.rows, H - d0);
            destination_side(proc_[i], proc_g_, kv, d0, n, x + d0 * C, y + d0 * C, ws);
        }
        report(hook, ("proc." + std::to_string(i) + ".out").c_str(), y, {H, C});
        x = y;
    }
    // Latent skip: processor(z) + z.
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < H * C; ++i) latent[i] = cfg_.latent_skip ? x[i] + z_enc[i] : x[i];
    report(hook, "latent", latent, {H, C});
}

void CpuBackend::decoder(const float* latent, const float* x_src, float* dec_out, const ActivationHook* hook) {
    const int64_t C = C_, G = G_, H = H_;
    const int64_t n_out = dec_out_proj_.n_out;
    Workspace ws(arena_, std::min(chunk_rows_, std::max(G, H)), C, cfg_.mlp_hidden);
    float* kv = arena_.get("kv", H * 2 * C);
    source_side(dec_, latent, H, kv, ws);

    // Destination side in chunks of data rows; contract tensors are assembled only with a hook.
    float* xd = arena_.get("chunk.a", ws.rows * C);
    float* blk = arena_.get("chunk.b", ws.rows * C);
    std::unique_ptr<float[]> emb_full = hook ? alloc(G * C) : nullptr;  // dec.emb_dst
    std::unique_ptr<float[]> blk_full = hook ? alloc(G * C) : nullptr;  // dec.block_out
    for (int64_t d0 = 0; d0 < G; d0 += ws.rows) {
        const int64_t n = std::min(ws.rows, G - d0);
        {
            Timer t(prof_, "gemm embed");
            cpu::linear(dec_emb_dst_, x_src + d0 * F_src_, F_src_, n, xd, C);
        }
        destination_side(dec_, dec_g_, kv, d0, n, xd, blk, ws);
        // node_data_extractor: Linear(LayerNorm(block_out)).
        {
            Timer t(prof_, "layernorm");
            cpu::layer_norm(dec_out_norm_, blk, C, n, ws.ln, C);
        }
        {
            Timer t(prof_, "gemm out_proj");
            cpu::linear(dec_out_proj_, ws.ln, C, n, dec_out + d0 * n_out, n_out);
        }
        if (emb_full) {
            std::memcpy(emb_full.get() + d0 * C, xd, sizeof(float) * n * C);
            std::memcpy(blk_full.get() + d0 * C, blk, sizeof(float) * n * C);
        }
    }
    report(hook, "dec.emb_dst", emb_full.get(), {G, C});
    report(hook, "dec.block_out", blk_full.get(), {G, C});
    report(hook, "dec.out", dec_out, {G, n_out});
}

void CpuBackend::predict_step(const float* x, float* y, const ActivationHook* hook) {
    omp_set_num_threads(threads_);
    prof_.t.clear();
    const double t_start = omp_get_wtime();
    const int64_t T = cfg_.n_input_steps, Vi = cfg_.n_input_vars, Vo = cfg_.n_output_vars;
    const int64_t G = G_, H = H_, C = C_, F = F_src_, A = cfg_.node_attr_dim;
    report(hook, "in.x", x, {1, T, G, Vi});

    // Normalise: x * mul, rounded, then + add (two fp32 ops, no FMA).
    float* xn = arena_.get("x_norm", T * G * Vi);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < T * G; ++r)
        for (int64_t v = 0; v < Vi; ++v) {
            const float m = x[r * Vi + v] * mul_in_[v];
            xn[r * Vi + v] = m + add_in_[v];
        }
    report(hook, "in.x_norm", xn, {1, T, 1, G, Vi});

    // Encoder input: time-major features, then the node attributes.
    float* x_src = arena_.get("x_src", G * F);
#pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; ++g) {
        float* row = x_src + g * F;
        for (int64_t t = 0; t < T; ++t) std::memcpy(row + t * Vi, xn + (t * G + g) * Vi, sizeof(float) * Vi);
        std::memcpy(row + T * Vi, data_attr_ + g * A, sizeof(float) * A);
    }
    report(hook, "enc.x_src", x_src, {G, F});
    report(hook, "enc.x_dst", hidden_attr_, {H, A});

    float* z = arena_.get("z", H * C);
    float* latent = arena_.get("latent", H * C);
    float* dec = arena_.get("dec_out", G * Vo);
    double t0 = omp_get_wtime();
    encoder(x_src, z, hook);
    prof_.add("= encoder", omp_get_wtime() - t0);
    t0 = omp_get_wtime();
    processor(z, latent, hook);
    prof_.add("= processor", omp_get_wtime() - t0);
    t0 = omp_get_wtime();
    decoder(latent, x_src, dec, hook);
    prof_.add("= decoder", omp_get_wtime() - t0);

    // Skip residual on the prognostic outputs (normalised input step residual_step), in y.
    const float* x_skip = xn + cfg_.residual_step * G * Vi;
#pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; ++g) {
        float* yr = y + g * Vo;
        std::memcpy(yr, dec + g * Vo, sizeof(float) * Vo);
        for (size_t k = 0; k < cfg_.residual_output.size(); ++k)
            yr[cfg_.residual_output[k]] = yr[cfg_.residual_output[k]] + x_skip[g * Vi + cfg_.residual_input[k]];
    }
    report(hook, "out.pre_bound", y, {1, 1, 1, G, Vo});

    // Bounding, in config order, in normalised space.
    for (const Bounding& b : cfg_.bounding) apply_bounding(b, y, G, Vo);
    report(hook, "out.norm", y, {1, 1, 1, G, Vo});

    // Inverse normalisation: (y - add) / mul.
#pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; ++g)
        for (int64_t v = 0; v < Vo; ++v) {
            const float d = y[g * Vo + v] - add_out_[v];
            y[g * Vo + v] = d / mul_out_[v];
        }
    report(hook, "out.y", y, {1, 1, 1, G, Vo});

    if (prof_.on) {
        std::fprintf(stderr, "[varda cpu] predict_step %.3f s (%d threads)\n", omp_get_wtime() - t_start, threads_);
        for (const auto& p : prof_.t) std::fprintf(stderr, "[varda cpu]   %-16s %8.3f s\n", p.first.c_str(), p.second);
    }
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, SafeTensorsFile& file, const Options& opts) {
    return std::make_unique<CpuBackend>(cfg, file, opts);
}

}  // namespace varda
