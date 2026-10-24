#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Matches Args in backend_metal.mm. All indexing is checked to fit uint at load.
struct Args {
    uint n, a, b, c, d, e, f, g;
    float x, y;
};

kernel void normalise(device const float* in [[buffer(0)]], device const float* mul [[buffer(1)]],
                      device const float* add [[buffer(2)]], device float* out [[buffer(3)]],
                      constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) { float v = in[i] * mul[i % p.a]; out[i] = v + add[i % p.a]; }
}

kernel void assemble(device const float* in [[buffer(0)]], device const float* attr [[buffer(1)]],
                     device float* out [[buffer(2)]], constant Args& p [[buffer(8)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint row = i / p.a, col = i % p.a;
    out[i] = col < p.b * p.c ? in[((col / p.b) * p.d + row) * p.b + col % p.b]
                             : attr[row * p.e + col - p.b * p.c];
}

kernel void bias_add(device float* out [[buffer(0)]], device const float* bias [[buffer(1)]],
                     constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = out[i] + bias[i % p.a];
}

kernel void add_arrays(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]], constant Args& p [[buffer(8)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = a[i] + b[i];
}

kernel void copy_array(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                       constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = in[i];
}

// One SIMD group per row, including widths that are not multiples of its size.
// Metal has no fp64: center on a common origin before compensated fp32 moments,
// then normalize centered values directly to avoid cancellation in x*rstd+shift.
kernel void layernorm(device const float* in [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]], device float* out [[buffer(3)]],
                      constant Args& p [[buffer(8)]], uint tid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]]) {
    uint row = tid / simd_width;
    if (row >= p.n) return;
    uint base = row * p.a;
    const float origin = in[base];
    float sum = 0, correction = 0;
    for (uint c = lane; c < p.a; c += simd_width) {
        float v = (in[base + c] - origin) - correction, next = sum + v;
        correction = (next - sum) - v; sum = next;
    }
    float mean = simd_sum(sum) / float(p.a);
    sum = 0; correction = 0;
    for (uint c = lane; c < p.a; c += simd_width) {
        float d = (in[base + c] - origin) - mean;
        float v = d * d - correction, next = sum + v;
        correction = (next - sum) - v; sum = next;
    }
    float rstd = 1.0f / sqrt(simd_sum(sum) / float(p.a) + p.x);
    for (uint c = lane; c < p.a; c += simd_width) {
        float v = ((in[base + c] - origin) - mean) * rstd;
        v = v * w[c];
        out[base + c] = p.b ? v + b[c] : v;
    }
}

// erf form of GELU, not the tanh approximation. A&S 7.1.26 evaluates erf with
// maximum absolute approximation error 1.5e-7, below fp32 parity tolerances.
float erf_fp32(float x) {
    float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
    float q = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                - 0.284496736f) * t + 0.254829592f) * t;
    return copysign(1.0f - q * exp(-a * a), x);
}

kernel void gelu(device float* values [[buffer(0)]], constant Args& p [[buffer(8)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float v = values[i];
    if (p.a) {
        float cube = v * v * v;
        values[i] = 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * cube)));
    } else {
        values[i] = v * 0.5f * (1.0f + erf_fp32(v * 0.7071067811865475f));
    }
}

float edge_value(device const float* attrs, device const float* w, device const float* b,
                 uint edge, uint channel, constant Args& p) {
    float value = attrs[edge * p.d] * w[channel * p.d];
    for (uint a = 1; a < p.d; ++a) value += attrs[edge * p.d + a] * w[channel * p.d + a];
    return p.e ? value + b[channel] : value;
}

// One score per (edge, head). This stores E*heads values, never an E*channels
// projected-edge tensor. d0 and nd restrict the dispatch to complete CSR rows.
kernel void attention_scores(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                             device const int* sources [[buffer(2)]], device const int* dests [[buffer(3)]],
                             device const float* attrs [[buffer(4)]], device const float* w [[buffer(5)]],
                             device const float* b [[buffer(6)]], device float* scores [[buffer(7)]],
                             constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint edge = p.g + i / p.b, head = i % p.b;
    uint dst = uint(dests[edge]) - p.f, src = uint(sources[edge]);
    float acc[8] = {0,0,0,0,0,0,0,0};
    uint c = 0, offset = head * p.c;
    for (; c + 8 <= p.c; c += 8)
        for (uint j = 0; j < 8; ++j) {
            uint channel = offset + c + j;
            float e = edge_value(attrs, w, b, edge, channel, p);
            acc[j] += q[dst * p.a + channel] * (k[src * p.a + channel] + e);
        }
    float sum = ((acc[0]+acc[1])+(acc[2]+acc[3])) + ((acc[4]+acc[5])+(acc[6]+acc[7]));
    for (; c < p.c; ++c) {
        uint channel = offset + c;
        sum += q[dst * p.a + channel] * (k[src * p.a + channel] + edge_value(attrs,w,b,edge,channel,p));
    }
    scores[edge * p.b + head] = sum / sqrt(float(p.c));
}

kernel void attention_softmax(device const int* rows [[buffer(0)]], device float* scores [[buffer(1)]],
                              constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint dst = p.f + i / p.b, head = i % p.b;
    int lo = rows[dst], hi = rows[dst + 1];
    if (lo == hi) return;
    float largest = scores[uint(lo) * p.b + head];
    for (int e = lo + 1; e < hi; ++e) largest = max(largest, scores[uint(e) * p.b + head]);
    float sum = 0;
    for (int e = lo; e < hi; ++e) {
        float v = exp(scores[uint(e) * p.b + head] - largest);
        scores[uint(e) * p.b + head] = v; sum += v;
    }
    sum = sum + 1e-16f;
    for (int e = lo; e < hi; ++e) scores[uint(e) * p.b + head] /= sum;
}

kernel void attention_values(device const float* v [[buffer(0)]], device const int* rows [[buffer(1)]],
                             device const int* sources [[buffer(2)]], device const float* attrs [[buffer(3)]],
                             device const float* w [[buffer(4)]], device const float* b [[buffer(5)]],
                             device const float* scores [[buffer(6)]], device float* out [[buffer(7)]],
                             constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint dst = p.f + i / p.a, channel = i % p.a, head = channel / p.c;
    float sum = 0;
    for (int e = rows[dst]; e < rows[dst + 1]; ++e) {
        float value = v[uint(sources[e]) * p.a + channel] + edge_value(attrs,w,b,uint(e),channel,p);
        sum += value * scores[uint(e) * p.b + head];
    }
    out[i] = sum;  // also defines zero-in-degree destinations
}

kernel void residual(device const float* in [[buffer(0)]], device const float* dec [[buffer(1)]],
                     device const int* ins [[buffer(2)]], device const int* outs [[buffer(3)]],
                     device float* out [[buffer(4)]], constant Args& p [[buffer(8)]],
                     uint row [[thread_position_in_grid]]) {
    if (row >= p.n) return;
    for (uint c = 0; c < p.a; ++c) out[row * p.a + c] = dec[row * p.a + c];
    for (uint j = 0; j < p.c; ++j)
        out[row * p.a + uint(outs[j])] += in[(p.d * p.e + row) * p.b + uint(ins[j])];
}

kernel void bound_clamp(device float* out [[buffer(0)]], device const int* indices [[buffer(1)]],
                        constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint at = (i / p.b) * p.a + uint(indices[i % p.b]);
    float v = out[at];
    if (p.c == 0) out[at] = v > 0.0f || isnan(v) ? v : 0.0f;
    else out[at] = isnan(v) ? v : min(max(v, p.x), p.y);
}

kernel void bound_product(device const float* in [[buffer(0)]], device const int* indices [[buffer(1)]],
                          device const int* totals [[buffer(2)]], device float* scratch [[buffer(3)]],
                          constant Args& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        uint row = i / p.b, j = i % p.b;
        scratch[i] = in[row * p.a + uint(indices[j])] * in[row * p.a + uint(totals[p.c == 1 ? 0 : j])];
    }
}

kernel void bound_assign(device const float* scratch [[buffer(0)]], device const int* indices [[buffer(1)]],
                         device float* out [[buffer(2)]], constant Args& p [[buffer(8)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[(i / p.b) * p.a + uint(indices[i % p.b])] = scratch[i];
}

kernel void denormalise(device float* out [[buffer(0)]], device const float* mul [[buffer(1)]],
                        device const float* add [[buffer(2)]], constant Args& p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) { float v = out[i] - add[i % p.a]; out[i] = v / mul[i % p.a]; }
}
