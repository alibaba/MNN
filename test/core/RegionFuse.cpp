//
//  RegionFuse.cpp
//  MNNTests
//
//  Created by wangzhaode on 2020/9/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/Tensor.hpp>
#include <string.h>
#include "core/TensorUtils.hpp"

using namespace MNN;
static std::string _printRegion(const Tensor::InsideDescribe::Region& reg) {
    char info[2048];
    sprintf(info, "size: %d, %d, %d; src: %d, %d, %d, %d; dst: %d, %d, %d, %d", reg.size[0], reg.size[1], reg.size[2], reg.src.offset, reg.src.stride[0], reg.src.stride[1], reg.src.stride[2], reg.dst.offset, reg.dst.stride[0], reg.dst.stride[1], reg.dst.stride[2]);
    info[2047] = 0;
    return std::string(info);
}
static std::pair<size_t, size_t> _computeMinSrcDstSize(const Tensor::InsideDescribe::Region& reg) {
    size_t srcSize = 1 + reg.src.offset;
    size_t dstSize = 1 + reg.dst.offset;
    for (int i=0; i<3; ++i) {
        if (reg.src.stride[i] > 0) {
            srcSize += reg.src.stride[i] * reg.size[i];
        }
        if (reg.dst.stride[i] > 0) {
            dstSize += reg.dst.stride[i] * reg.size[i];
        }
    }
    return std::make_pair(srcSize, dstSize);
}
static bool _computeRaw(std::vector<int>& dst, const std::vector<int>& src, const Tensor::InsideDescribe::Region& reg) {
    int dstOffset = reg.dst.offset;
    int srcOffset = reg.src.offset;
    ::memset(dst.data(), 0, dst.size() * sizeof(int));
    for (int z=0; z<reg.size[0]; ++z) {
        int srcZ = srcOffset + z * reg.src.stride[0];
        int dstZ = dstOffset + z * reg.dst.stride[0];
        for (int y=0; y<reg.size[1]; ++y) {
            int srcY = srcZ + y * reg.src.stride[1];
            int dstY = dstZ + y * reg.dst.stride[1];
            for (int x=0; x<reg.size[2]; ++x) {
                int srcX = srcY + x * reg.src.stride[2];
                int dstX = dstY + x * reg.dst.stride[2];
                if (srcX < 0 || srcX >= src.size() || dstX < 0 || dstX >= dst.size()) {
                    return false;
                }
                dst[dstX] = src[srcX];
            }
        }
    }
    return true;
}

class RegionFuseTest : public MNNTestCase {
public:
    using Region = Tensor::InsideDescribe::Region;
    virtual ~RegionFuseTest() = default;
    virtual bool run(int precision) {
        constexpr int N = 12;
        // [src_offset, src_stride_0_1_2, dst_offset, dst_stride_0_1_2, size_0_1_2]
        int data[N*3][11] = {
            // 2D-transpose + 2D-transpose = memcpy: [1, 4, 16] => [1, 16, 4] => [1, 4, 16]
            {0, 1, 1, 16, 0, 1, 4, 1, 1, 16, 4},
            {0, 1, 1, 4, 0, 1, 16, 1, 1, 4, 16},
            {0, 1, 16, 1, 0, 1, 16, 1, 1, 4, 16},
            // transpose + memcpy = transpose: [1, 4, 16] => [1, 16, 4] => [16, 1, 4]
            {0, 1, 1, 16, 0, 1, 4, 1, 1, 16, 4},
            {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 64},
            {0, 1, 1, 16, 0, 1, 4, 1, 1, 16, 4},
            // transpose + transpose' = transpose'': [3, 4, 5] => [5, 3, 4] => [4, 5, 3]
            {0, 1, 1, 5, 0, 1, 12, 1, 1, 5, 12},
            {0, 1, 1, 4, 0, 1, 15, 1, 1, 4, 15},
            {0, 5, 1, 20, 0, 15, 3, 1, 4, 5, 3},
            // memcpy + memcpy' = memcpy'': offset:2 => offset:3 => offser:6+2-3=5, clip: range: 3-19 & 6-22 = 6-19, size=13
            {2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 16},
            {6, 1, 1, 1, 0, 1, 1, 1, 1, 1, 16},
            {5, 1, 1, 1, 0, 1, 1, 1, 1, 1, 13},
            // transpose + slice (offset align) => [3, 3, 4] => [3, 4, 3] => [2, 4, 3]
            {0, 12, 1, 4, 0, 12, 3, 1, 3, 4, 3},
            {12, 36, 3, 1, 0, 24, 3, 1, 1, 8, 3},
            {12, 12, 1, 4, 0, 12, 3, 1, 2, 4, 3},
            // transpose + slice (offset dont align) => [3, 3, 4] => [3, 4, 3] => [1, 6, 3] <can't fuse!>
            {0, 12, 1, 4, 0, 12, 3, 1, 3, 4, 3},
            {18, 36, 3, 1, 0, 18, 3, 1, 1, 6, 3},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            // copy + expand (src < dst) => [34491] => [34645] => [34645, 2] , clip [34491, 34645] -> [34491, 2]
            {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 34491},
            {0, 1, 1, 1, 0, 2, 1, 1, 34645, 1, 1},
            {0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 34491},
            // transpose + slice: [3, 256, 940] => [3, 940, 256] => [1, 256, 940] (expand_val = 1)
            {0, 240640, 1, 940, 0, 240640, 256, 1, 3, 940, 256},
            {0, 1, 256, 1, 0, 1, 768, 1, 1, 940, 256},
            {0, 240640, 1, 940, 0, 721920, 768, 1, 1, 940, 256},
            // transpose + slice (stride = 0) <can't fuse>
            {0, 4608, 1, 36, 0, 4608, 128, 1, 1, 36, 128},
            {0, 128, 0, 1, 0, 256, 128, 1, 6, 2, 128},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            // transpose + slice (dont align, not full copy) <can't fuse>
            {0, 1600, 1, 4, 0, 1600, 400, 1, 53, 4, 400},
            {0, 400, 20, 1, 0, 400, 20, 1, 190, 20, 20},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            // pad + transpose + slice + transpose (not full copy) 
            {0, 12321, 111, 1, 0, 12544, 112, 1, 32, 111, 111},
            {113, 12544, 112, 1, 0, 12321, 111, 1, 32, 111, 111},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            // concat + stack
            {0, 32, 1, 1, 32, 64, 1, 1, 20, 32, 1},
            {0, 0, 1280, 1, 0, 1, 640, 1, 1, 1, 640},
            {0, 0, 32, 1, 32, 0, 64, 1, 1, 10, 32},
        };
        TensorUtils::FuseWrap fuseUtils;
        for (int i = 0; i < N; i++) {
            Region src, dst;
            src.origin = nullptr;
            dst.origin = nullptr;
            ::memcpy(&src, data[3 * i], 44);
            ::memcpy(&dst, data[3 * i + 1], 44);
            auto srcSize = _computeMinSrcDstSize(src);
            auto dstSize = _computeMinSrcDstSize(dst);
            auto midSize = ALIMAX(srcSize.second, dstSize.first);
            std::vector<int> srcData(srcSize.first);
            std::vector<int> midData(midSize);
            std::vector<int> dstData(dstSize.second);
            std::vector<int> dstDataFuse(dstSize.second);
            for (int v=0; v<srcSize.first; ++v) {
                srcData[v] = v + 1;
            }
            auto computeRes = _computeRaw(midData, srcData, src);
            MNN_ASSERT(computeRes);
            computeRes = _computeRaw(dstData, midData, dst);
            MNN_ASSERT(computeRes);

            bool fused = fuseUtils.match(src, dst);
            Region newDst = dst;
            if (fused) {
                fuseUtils.apply(src, newDst);
            }
            if (data[3 * i + 2][0] < 0 && !fused) {
                continue;
            }
            if (!fused) {
                MNN_ERROR("regionfuse %d test failed for fuse!\n", i);
                auto srcStr = _printRegion(src);
                auto dstStr = _printRegion(dst);
                auto tarStr = _printRegion(newDst);
                MNN_PRINT("Fuse Error:\n %s \n %s\n To: \n", srcStr.c_str(), dstStr.c_str());
                MNN_PRINT("%s\n", tarStr.c_str());
                return false;
            }
            computeRes = _computeRaw(dstDataFuse, srcData, newDst);
            if ((0 != ::memcmp(dstDataFuse.data(), dstData.data(), dstData.size() * sizeof(int))) || (!computeRes)) {
                MNN_ERROR("%d regionfuse compute error\n", i);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RegionFuseTest, "core/regionfuse");
