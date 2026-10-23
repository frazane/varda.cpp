// The production LayerNorm shader against an independent double-precision
// reference, including cancellation-prone rows and partially occupied groups.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "varda/metal/kernels_source.h"

namespace {
struct Args {
    uint32_t n, a, b, c = 0, d = 0, e = 0, f = 0, g = 0;
    float x = 1e-5f, y = 0;
};
static_assert(sizeof(Args) == 40, "Metal parameter layout");

struct Result { double rel, abs; bool valid; };

Result layernorm_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pipeline, int channels, int rows,
                      int pattern, bool bias, std::mt19937& rng) {
    std::normal_distribution<float> normal(0, 1);
    const size_t count = size_t(channels) * rows;
    std::vector<float> x(count), w(channels), b(channels);
    for (size_t i = 0; i < count; ++i) {
        switch (pattern) {
            case 0: x[i] = normal(rng); break;
            case 1: x[i] = 0.125f; break;
            case 2: x[i] = 1000 + 0.01f * normal(rng); break;
            default: x[i] = (i % 2 ? 10000 : -10000) + normal(rng); break;
        }
    }
    for (int c = 0; c < channels; ++c) {
        w[c] = 0.5f + std::abs(normal(rng));
        b[c] = normal(rng) * 0.2f;
    }
    const Args args{uint32_t(rows), uint32_t(channels), uint32_t(bias)};
    std::vector<double> reference(count);
    for (int r = 0; r < rows; ++r) {
        double mean = 0, variance = 0;
        for (int c = 0; c < channels; ++c) mean += x[r * channels + c];
        mean /= channels;
        for (int c = 0; c < channels; ++c) {
            const double delta = x[r * channels + c] - mean;
            variance += delta * delta;
        }
        variance /= channels;
        for (int c = 0; c < channels; ++c)
            reference[r * channels + c] = (x[r * channels + c] - mean) /
                std::sqrt(variance + args.x) * w[c] + (bias ? b[c] : 0);
    }

    auto upload = [&](const std::vector<float>& v) {
        return [device newBufferWithBytes:v.data() length:v.size() * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> input = upload(x), weight = upload(w), offset = upload(b);
    id<MTLBuffer> output = [device newBufferWithLength:(count + 32) * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    if (!input || !weight || !offset || !output) throw std::runtime_error("Metal test allocation failed");
    float* values = static_cast<float*>(output.contents);
    constexpr float guard = 1234567.0f;
    std::fill_n(values, count + 32, guard);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) throw std::runtime_error("Metal test encoding failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:offset offset:0 atIndex:2];
    [encoder setBuffer:output offset:16 * sizeof(float) atIndex:3];
    [encoder setBytes:&args length:sizeof(args) atIndex:8];
    const NSUInteger simd = pipeline.threadExecutionWidth;
    const NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup) / simd * simd;
    [encoder dispatchThreads:MTLSizeMake(rows * simd, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);

    double scale = 0, error = 0;
    bool valid = true;
    for (size_t i = 0; i < count; ++i) {
        valid &= std::isfinite(values[i + 16]);
        scale = std::max(scale, std::abs(reference[i]));
        error = std::max(error, std::abs(double(values[i + 16]) - reference[i]));
    }
    for (int i = 0; i < 16; ++i)
        valid &= values[i] == guard && values[count + 16 + i] == guard;
    return {scale > 0 ? error / scale : error, error, valid};
}
}  // namespace

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || !device.hasUnifiedMemory) {
                std::puts("SKIP: no unified-memory Metal GPU");
                return 77;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) throw std::runtime_error("cannot create test command queue");
            MTLCompileOptions* options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
#else
            options.fastMathEnabled = NO;
#endif
            options.languageVersion = MTLLanguageVersion2_3;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:varda::metal::kernel_source]
                                                         options:options error:&error];
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"layernorm"] error:&error];
            if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);

            std::mt19937 rng(731);
            int cases = 0, failures = 0;
            double worst = 0, worst_abs = 0;
            for (int channels : {1, 7, 31, 32, 33, 35, 63, 64, 65, 255, 256, 257, 768, 1024, 2048, 4097})
                for (int rows : {1, 9, 37})
                    for (int pattern = 0; pattern < 4; ++pattern)
                        for (bool bias : {false, true}) {
                            @autoreleasepool {
                                const Result r = layernorm_case(device, queue, pipeline, channels, rows, pattern, bias, rng);
                                ++cases;
                                if (r.rel > worst) { worst = r.rel; worst_abs = r.abs; }
                                if (!r.valid || r.rel > 1e-5) {
                                    ++failures;
                                    std::printf("FAIL C=%d rows=%d pattern=%d bias=%d rel_absmax=%.3e max_abs=%.3e\n",
                                                channels, rows, pattern, int(bias), r.rel, r.abs);
                                }
                            }
                        }
            std::printf("%s fp32 LayerNorm: %d cases, %d failures; worst rel_absmax %.3e max_abs %.3e tol 1e-5\n",
                        device.name.UTF8String, cases, failures, worst, worst_abs);
            return failures ? 1 : 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
