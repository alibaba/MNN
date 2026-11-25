//
//  MetalPrepare.hpp
//  MNN
//
//  Created by MNN on 2026/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_BENCH_METAL_PREPARE_HPP
#define MNN_BENCH_METAL_PREPARE_HPP
#include <CoreVideo/CoreVideo.h>
#include <array>
#include <cstdint>
#include <memory>
namespace MNN {
struct BenchMappedWeight {
    static std::shared_ptr<BenchMappedWeight> map(int fd, size_t offset, size_t elements);
    ~BenchMappedWeight();
    void* address = nullptr;
    size_t length = 0;
    uint32_t codes = 0, alpha = 0;
};
struct BenchProjection {
    const BenchMappedWeight* source;
    int rows, cols;
    CVPixelBufferRef image;
    int16_t* scales;
};
class BenchMetalPrepare {
public:
    BenchMetalPrepare();
    ~BenchMetalPrepare();
    bool convert(const std::array<BenchProjection, 7>& projections, int layer);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
} // namespace MNN
#endif
