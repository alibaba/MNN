//
//  MetalPrepare.mm
//  MNN
//
//  Created by MNN on 2026/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MetalPrepare.hpp"
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <os/signpost.h>
#include <sys/mman.h>
#include <unistd.h>

namespace MNN {
std::shared_ptr<BenchMappedWeight> BenchMappedWeight::map(int fd, size_t offset, size_t elements) {
    if (fd < 0 || (offset + elements / 2) % alignof(uint16_t)) {
        return nullptr;
    }
    auto weight = std::make_shared<BenchMappedWeight>();
    const size_t page = getpagesize();
    const size_t aligned = offset / page * page;
    weight->codes = offset - aligned;
    weight->alpha = weight->codes + elements / 2;
    weight->length = (weight->alpha + elements / 16 + page - 1) / page * page;
    void* addr = mmap(nullptr, weight->length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, aligned);
    if (addr == MAP_FAILED) {
        return nullptr;
    }
    weight->address = addr;
    return weight;
}
BenchMappedWeight::~BenchMappedWeight() {
    if (address) {
        munmap(address, length);
    }
}

struct BenchMetalPrepare::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    struct Output {
        id<MTLBuffer> codes, scales;
        std::shared_ptr<__CVBuffer> owner;
    };
    std::array<std::array<Output, 7>, 2> outputs;
    bool initialize() {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        // Embed the measured kernel so the benchmark is independent of its working directory.
        NSString* source = @
#include "Requantize.metal.inc"
            ;
        auto lib = [device newLibraryWithSource:source options:options error:&error];
        auto fn = [lib newFunctionWithName:@"requantLinear"];
        if (fn) {
            pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        }
        if (!queue || !pipeline || pipeline.threadExecutionWidth != 32 ||
            pipeline.maxTotalThreadsPerThreadgroup < 128) {
            NSLog(@"Unsupported weight conversion pipeline: %@", error);
            pipeline = nil;
            return false;
        }
        return true;
    }
};
BenchMetalPrepare::BenchMetalPrepare() : mImpl(new Impl) {}
BenchMetalPrepare::~BenchMetalPrepare() = default;

bool BenchMetalPrepare::convert(const std::array<BenchProjection, 7>& projections, int layer) {
    @autoreleasepool {
        auto& gpu = *mImpl;
        if (!gpu.pipeline && !gpu.initialize()) {
            return false;
        }
        auto cmd = [gpu.queue commandBuffer];
        cmd.label = @"MNN convert next layer";
        auto encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:gpu.pipeline];
        for (size_t i = 0; i < projections.size(); ++i) {
            const auto& p = projections[i];
            auto& out = gpu.outputs[layer % 2][i];
            const size_t pitch = CVPixelBufferGetBytesPerRow(p.image);
            if (out.owner.get() != p.image) {
                auto surface = CVPixelBufferGetIOSurface(p.image);
                const size_t page = getpagesize();
                size_t n = pitch * CVPixelBufferGetHeight(p.image);
                n = (n + page - 1) / page * page;
                void* addr = surface ? IOSurfaceGetBaseAddress(surface) : nullptr;
                if (!addr || uintptr_t(addr) % page || pitch % 4 || n > IOSurfaceGetAllocSize(surface)) {
                    [encoder endEncoding];
                    return false;
                }
                out.owner.reset(CVPixelBufferRetain(p.image), CVPixelBufferRelease);
                out.codes = [gpu.device newBufferWithBytesNoCopy:addr
                                                          length:n
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
                out.scales = [gpu.device newBufferWithLength:p.rows * sizeof(int16_t)
                                                     options:MTLResourceStorageModeShared];
            }
            const auto& src = *p.source;
            auto input = [gpu.device newBufferWithBytesNoCopy:src.address
                                                       length:src.length
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            if (!input || !out.codes || !out.scales || !cmd || !encoder) {
                [encoder endEncoding];
                return false;
            }
            const uint32_t params[] = {
                uint32_t(p.rows), uint32_t(p.cols), src.codes, src.alpha, uint32_t(CVPixelBufferGetWidth(p.image)),
                uint32_t(pitch)};
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:out.scales offset:0 atIndex:1];
            [encoder setBytes:params length:sizeof(params) atIndex:2];
            [encoder setBuffer:out.codes offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"Weight conversion failed: %@", cmd.error);
            return false;
        }
        for (size_t i = 0; i < projections.size(); ++i) {
            memcpy(projections[i].scales, gpu.outputs[layer % 2][i].scales.contents,
                   projections[i].rows * sizeof(int16_t));
        }
        static os_log_t log = os_log_create("org.mnn.coreml", "prefill");
        os_signpost_event_emit(log, OS_SIGNPOST_ID_EXCLUSIVE, "WeightGPU", "layer=%d start=%llu end=%llu anchor=%llu",
                               layer, (unsigned long long)(cmd.GPUStartTime * 1e9),
                               (unsigned long long)(cmd.GPUEndTime * 1e9),
                               (unsigned long long)clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
        return true;
    }
}
} // namespace MNN
