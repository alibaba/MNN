//
//  ConvInt8Test.cpp
//  MNNTests
//
//  Created by MNN on b'2020/02/19'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include <random>
#include "MNN_generated.h"
#include "MNNTestSuite.h"
using namespace MNN::Express;
using namespace MNN;
static PadMode _convertPadMode(PaddingMode mode) {
    switch (mode) {
        case CAFFE:
            return PadMode_CAFFE;
        case VALID:
            return PadMode_VALID;
        case SAME:
            return PadMode_SAME;
        default:
            break;
    }
    return PadMode_CAFFE;
}

inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = roundf((float)(data + bias) * scale);
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(value);
}

// y = Conv(x, w), x and y is C4 ordered format, weight is [oc, ic, kh, kw] raw format.
static std::vector<int8_t> naiveConvInt8C4(const int8_t* x, const int8_t* weight, const int* bias, const float* scale,
                                           int ow, int oh, int iw, int ih, int ic, int oc, int kw, int kh, int padX, int padY, int padValue = 0,
                                           int strideX = 1, int strideY = 1, int dilateX = 1, int dilateY = 1, int batch = 1) {
    int ic4 = (ic + 3) / 4, oc4 = (oc + 3) / 4;
    std::vector<int8_t> yCorrect(batch * oc4 * oh * ow * 4, 0);
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int ozC4 = oz / 4, ozRemain = oz % 4;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    int32_t yInt32 = 0;
                    for (int sz = 0; sz < ic; ++sz) {
                        int szC4 = sz / 4, szRemain = sz % 4;
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * strideX + kx * dilateX - padX, iy = oy * strideY + ky * dilateY - padY;
                                int8_t xValue = padValue;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = x[(((b * ic4 + szC4) * ih + iy) * iw + ix) * 4 + szRemain];
                                }
                                yInt32 += xValue * weight[((oz * ic + sz) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    yCorrect[(((b * oc4 + ozC4) * oh + oy) * ow + ox) * 4 + ozRemain] = int32ToInt8(yInt32, bias[oz], scale[oz]);
                }
            }
        }
    }
    return yCorrect;
}

class ConvInt8TestCommon : public MNNTestCase {
protected:
    static bool testKernel(INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int nbit = 8, bool overflow = false) {
        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(channel[1] * channel[0] * kernel[0] * kernel[1]);
        int iw = inputShape[0], ih = inputShape[1];
        VARP x     = _Input({1, channel[0], ih, iw}, NC4HW4, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        auto xPtr  = x->writeMap<int8_t>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        for (int i = 0; i < xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1)) + xMin; // x in [xMin, xMax]
        }
        for (int i = 0; i < channel[1]; ++i) {
            bias[i]  = (10000 + i * i * 10 - i * i * i) % 12580;
            scale[i] = ((127 - i) * i % 128) / 20000.0f;
            for (int j = 0; j < channel[0]; ++j) {
                auto weightCurrent = weight.data() + (i * channel[0] + j) * kernel[0] * kernel[1];
                for (int k = 0; k < kernel[0] * kernel[1]; ++k) {
                    weightCurrent[k] = ((i * i + j * j + k * k) % (xMax - xMin + 1)) + xMin; // w in [xMin, xMax]
                }
            }
        }
        VARP y;
        if (overflow) {
            y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), x,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, 0, 0, -127, 127, true);
        } else {
            y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), x,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, 0, 0, -127, 127, false);
        }
        auto yInfo = y->getInfo();
        auto yPtr  = y->readMap<int8_t>();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        auto targetValues = naiveConvInt8C4(xPtr, weight.data(), bias.data(), scale.data(),
                                            ow, oh, iw, ih, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1]);
        for (int i = 0; i < targetValues.size(); ++i) {
            int8_t targetValue = targetValues[i], computeResult = yPtr[i];
            if (targetValue != computeResult) {
                MNN_PRINT("ConvInt8 result Error: %d -> %d\n", targetValue, computeResult);
                return false;
            }
        }
        return true;
    }
};

class ConvInt8Im2colGemmTest : public ConvInt8TestCommon {
public:
    virtual bool run() {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {3, 4}, inputShape = {215, 204}; // {w, h}
        INTS channel = {11, 7}; // {ci, co}
        std::vector<std::vector<int>> kernels = {
            {4, 2}, {1, 5}, {7, 1}
        };
        std::vector<std::string> titles = {"4x2", "1x5", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                return false;
            }
        }
        inputShape = {215, 201};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};

class ConvInt8WinogradTest : public ConvInt8TestCommon {
public:
    virtual bool run() {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {1, 1}, inputShape = {128, 128}; // {w, h}
        INTS channel = {4, 4}; // {ci, co}
        
        std::vector<std::vector<int>> kernels = {
            {3, 3}, {7, 1}, {1, 7}
        };
        std::vector<std::string> titles = {"3x3", "1x7", "7x1"};
        std::vector<int> bits = {5, 6, 6};
        
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, bits[i]);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvInt8Im2colGemmTest, "op/ConvInt8/im2col_gemm");
MNNTestSuiteRegister(ConvInt8WinogradTest, "op/ConvInt8/winograd");
