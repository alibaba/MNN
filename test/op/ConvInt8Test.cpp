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
// weight: [group, ocGroup, icGroup, kh, kw]
static std::vector<int8_t> naiveConvInt8(const int8_t* x, const int8_t* weight, const int* bias, const float* scale,
                                           int ow, int oh, int iw, int ih, int ic, int oc, int kw, int kh, int padX, int padY, int group, int padValue = 0,
                                           int strideX = 1, int strideY = 1, int dilateX = 1, int dilateY = 1, int batch = 1) {
    int ocGroup = oc / group, icGroup = ic / group;
    std::vector<int8_t> yCorrect(batch * oc * oh * ow, 0);
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int gId = oz / ocGroup;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    int32_t yInt32 = 0;
                    auto destOffset = ((b * oc + oz) * oh + oy) * ow + ox;
                    for (int sz = gId * icGroup; sz < (gId + 1) * icGroup; ++sz) {
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * strideX + kx * dilateX - padX, iy = oy * strideY + ky * dilateY - padY;
                                int8_t xValue = padValue;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = x[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }
                                yInt32 += xValue * weight[(((gId * ocGroup + oz % ocGroup) * icGroup + sz % icGroup) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    yCorrect[destOffset] = int32ToInt8(yInt32, bias[oz], scale[oz]);
                }
            }
        }
    }
    return yCorrect;
}

class ConvInt8TestCommon : public MNNTestCase {
protected:
    static bool testKernel(INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int nbit = 8, bool overflow = false, int group = 1, int batch = 1) {
        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(channel[1] * channel[0] * kernel[0] * kernel[1]);
        int iw = inputShape[0], ih = inputShape[1];
        VARP x     = _Input({batch, channel[0], ih, iw}, NCHW, halide_type_of<int8_t>());
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
        auto saveWeight = weight;
        auto saveBias = bias;
        auto saveScale = scale;
        VARP y;
        auto xC4 = _Convert(x, NC4HW4);
        if (overflow) {
            y     = _Conv(std::move(weight), std::move(bias), std::move(scale), xC4,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, group, pad, false, 0, 0, -127, 127, true);
        } else {
            y     = _Conv(std::move(weight), std::move(bias), std::move(scale), xC4,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, group, pad, false, 0, 0, -127, 127, false);
        }
        y = _Convert(y, NCHW);
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        auto targetValues = naiveConvInt8(xPtr, saveWeight.data(), saveBias.data(), saveScale.data(),
                                            ow, oh, iw, ih, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1], group, 0, strides[0], strides[1], dilate[0], dilate[1], batch);
        auto yPtr  = y->readMap<int8_t>();
        for (int i = 0; i < targetValues.size(); ++i) {
            int8_t targetValue = targetValues[i], computeResult = yPtr[i];
            // Because of round implement in ARM / X86 / PC may cause 1 / 0 / -1 diff, don't care about this error
            auto error = (int32_t)targetValue - (int32_t)computeResult;
            if (error * error > 1) {
                MNN_PRINT("%d x %d, ConvInt8 result %d Error: %d -> %d\n", ow, oh, i, targetValue, computeResult);
                return false;
            }
        }
        return true;
    }
};

class ConvInt8Im2colGemmTest : public ConvInt8TestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {3, 4}, inputShape = {34, 23}; // {w, h}
        INTS channel = {64, 64}; // {ci, co}
        std::vector<std::vector<int>> kernels = {
            {4, 2}, {1, 5}, {7, 1}
        };
        std::vector<std::string> titles = {"4x2", "1x5", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 2);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 3);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 5);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 2);
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
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {1, 1}, inputShape = {76, 31}; // {w, h}
        INTS channel = {32, 32}; // {ci, co}
        
        std::vector<std::vector<int>> kernels = {
            {4, 4}, {3, 3}, {7, 1}, {1, 7}, {2, 3}, {3, 2} // {w, h}
        };
        std::vector<std::string> titles = {
            "4x4", "3x3", "1x7", "7x1", "3x2", "2x3"
        };
        std::vector<int> bits = {5, 5, 5, 5, 5, 5};
        
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, bits[i], false, 1, 3);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};

class DepthwiseConvInt8Test : public ConvInt8TestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {21, 13}; // {w, h}
        int channel = 64;
        std::vector<std::vector<int>> kernels = {
            {3, 3}
        };
        std::vector<std::string> titles = {
            "3x3"
        };
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 8, false, channel, 4);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 3, true, channel);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvInt8Im2colGemmTest, "op/ConvInt8/im2col_gemm");
MNNTestSuiteRegister(ConvInt8WinogradTest, "op/ConvInt8/winograd");
MNNTestSuiteRegister(DepthwiseConvInt8Test, "op/ConvInt8/depthwise");
