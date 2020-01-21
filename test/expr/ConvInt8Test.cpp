//
//  ConvInt8Test.cpp
//  MNNTests
//
//  Created by MNN on 2019/010/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <random>
#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
using namespace MNN::Express;
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = roundf((float)(data + bias) * scale);
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(value);
}

class ConvInt8Test : public MNNTestCase {
public:
    static bool _testKernel(INTS kernel) {
        INTS strides = {1, 1};
        INTS dilate = {1, 1};
        INTS channel = {11, 7};
        INTS pad = {3, 4};
        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(channel[1] * channel[0] * kernel[0] * kernel[1]);
        VARP x = _Input({1, channel[0], 204, 215}, NC4HW4, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<int8_t>();
        for (int i=0; i<xInfo->size; ++i) {
            xPtr[i] = (i % 254) - 127;
        }
        for (int i=0; i<channel[1]; ++i) {
            bias[i] = (10000 + i*i*10 - i*i*i) % 12580;
            scale[i] = ((127-i)*i % 128) / 20000.0f;
            for (int j=0; j<channel[0];++j) {
                auto weightCurrent = weight.data() + (i*channel[0]+j)*kernel[0]*kernel[1];
                for (int k=0; k<kernel[0]*kernel[1]; ++k) {
                    weightCurrent[k] = ((i*i + j*j + k*k) % 254) - 127;
                }
            }
        }
        std::vector<int8_t> originWeight = weight;
        auto originScale = scale;
        auto originBias = bias;
        auto y = _Conv(std::move(weight), std::move(bias), std::move(scale), x, channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad);
        auto yInfo = y->getInfo();
        auto yPtr = y->readMap<int8_t>();
        auto ow = yInfo->dim[3];
        auto oh = yInfo->dim[2];
        auto iw = xInfo->dim[3];
        auto ih = xInfo->dim[2];
        for (int oz=0; oz<channel[1]; ++oz) {
            auto ozC4 = oz / 4;
            auto ozRemain = oz % 4;
            int32_t biasValue = (10000 + oz*oz*10 - oz*oz*oz) % 12580;
            float scaleValue = ((127-oz)*oz % 128) / 20000.0f;
            for (int oy=0; oy<oh; ++oy) {
                for (int ox = 0; ox<ow; ++ox) {
                    auto computeResult = yPtr[(ozC4*oh*ow+ow*oy+ox)*4 + ozRemain];
                    int32_t sum = 0;
                    for (int sz=0; sz<channel[0]; ++sz) {
                        auto szC4 = sz / 4;
                        auto szRemain = sz % 4;
                        auto srcPtr = xPtr + szC4 * iw * ih * 4 + szRemain;
                        for (int ky=0; ky<kernel[1]; ++ky) {
                            int sy = ky * dilate[1] + oy * strides[1] - pad[1];
                            if (sy >= ih || sy < 0) {
                                continue;
                            }
                            for (int kx=0; kx<kernel[0]; ++kx) {
                                int sx = kx * dilate[0] + ox * strides[0] - pad[0];
                                if (sx >= iw || sx < 0) {
                                    continue;
                                }
                                sum += (int)srcPtr[sx*4+sy*iw*4] *
                                (int)originWeight[((
                                                    oz* channel[0]+sz)
                                                   *kernel[1]+ky)
                                                  *kernel[0]+kx];
                            }
                        }
                    }
                    auto targetValue = int32ToInt8(sum, biasValue, scaleValue);
                    if (targetValue != computeResult) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    virtual bool run() {
        auto res = _testKernel({3, 3});
        if (!res) {
            MNN_ERROR("Error for test kernel 3x3 for convint8\n");
            return false;
        }
        res = _testKernel({1, 3});
        if (!res) {
            MNN_ERROR("Error for test kernel 1x3 for convint8\n");
            return false;
        }
        res = _testKernel({1, 1});
        if (!res) {
            MNN_ERROR("Error for test kernel 1x1 for convint8\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvInt8Test, "expr/ConvInt8");
