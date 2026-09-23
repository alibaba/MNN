//
//  Deconv2x2.hpp
//  MNN
//
//  Copyright (c) 2026, Alibaba Group Holding Limited
//

#ifndef MNN_DECONV_2X2_HPP
#define MNN_DECONV_2X2_HPP

#include <algorithm>

namespace MNN {

// Source: [channel blocks, 4 kernel positions, tile length, Pack].
// Output: [channel blocks, batch, 2 * input height, 2 * input width, Pack].
// parameters: input start, tile length, input width, input height, batch, channel blocks.
template <typename V, int Pack>
void deconv2x2Post(const float* src, float* dst, const float* bias, const float* post, const int* parameters) {
    const int start = parameters[0], count = parameters[1], width = parameters[2], height = parameters[3];
    const int batch = parameters[4], channels = parameters[5];
    const V zero(0.0f), lower(post[2]), upper(post[3]);
    for (int offset = 0; offset < count;) {
        const int index = start + offset;
        const int b = index / (width * height);
        const int iy = (index % (width * height)) / width;
        const int ix = index % width;
        const int length = std::min(count - offset, width - ix);
        for (int z = 0; z < channels; ++z) {
            auto s = src + (z * 4 * count + offset) * Pack;
            auto d = dst + (((z * batch + b) * height * 2 + iy * 2) * width * 2 + ix * 2) * Pack;
            const auto biasValue = V::load(bias + z * Pack);
            for (int fy = 0; fy < 2; ++fy) {
                auto left = s + fy * 2 * count * Pack;
                auto right = left + count * Pack;
                auto row = d + fy * width * 2 * Pack;
                for (int x = 0; x < length; ++x) {
                    // Keep the original add-to-zero, bias, and clamp order.
                    auto a = V::load(left + x * Pack) + zero;
                    auto b0 = V::load(right + x * Pack) + zero;
                    a = V::max(V::min(a + biasValue, upper), lower);
                    b0 = V::max(V::min(b0 + biasValue, upper), lower);
                    V::save(row + 2 * x * Pack, a);
                    V::save(row + (2 * x + 1) * Pack, b0);
                }
            }
        }
        offset += length;
    }
}

} // namespace MNN
#endif
