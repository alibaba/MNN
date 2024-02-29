//
//  ConvSpeedInt8Test.cpp
//  MNNTests
//
//  Created by MNN on 2019/010/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"
#include "TestUtils.h"
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
static VARP _HybridConv(std::vector<int8_t>&& weight, std::vector<float>&& bias, std::vector<float>&& alpha, VARP x, INTS channel, INTS kernelSize,
           PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu, bool relu6, int nbits) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->symmetricQuan.reset(new QuantizedFloatParamT);
    conv2D->common->padMode     = _convertPadMode(pad);
    if (pads.size() == 2) {
        conv2D->common->padX        = pads[0];
        conv2D->common->padY        = pads[1];
    } else {
        conv2D->common->pads = std::move(pads);
    }
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    conv2D->common->relu6 = relu6;
    conv2D->common->relu = relu;
    conv2D->quanParameter.reset(new IDSTQuanT);
    if (nbits == 8) {
        conv2D->quanParameter->type = 4;
    } else {
        conv2D->quanParameter->type = 1;
    }
    conv2D->quanParameter->buffer = std::move(weight);
    conv2D->quanParameter->alpha = std::move(alpha);
    conv2D->quanParameter->quantScale = 1.0f;
    conv2D->weight.clear();
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = std::move(bias);
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

static float findAbsMax(const float *weights, const int count) {
    float absMax = fabs(weights[0]);
    for (int i = 1; i < count; i++) {
        float value = fabs(weights[i]);
        if (value > absMax) {
            absMax = value;
        }
    }

    return absMax;
}

class HybridConvSpeedTestCommon : public MNNTestCase {
protected:
    static bool testKernel(std::string title, INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int batch = 1, int nbit = 8, int precision = 1, bool testSpeed = false) {
        float fac = 1.23;
        int res = 10;
        float tail = 0.2;
        int ic = channel[0], oc = channel[1];
        int iw = inputShape[0], ih = inputShape[1];
        std::vector<float> bias(oc), biastest(oc), biasdup(oc);
        std::vector<int8_t> quantWeight(oc * ic);
        std::vector<float> weightFp32(oc * ic);
        std::vector<float> wScale(oc);

        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold;
        VARP x = _Input({batch, ic, 1, 1}, NC4HW4, halide_type_of<float>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<float>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        for (int i=0; i<xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1)) * 0.27;
        }
        for (int i = 0; i < oc; ++i) {
            bias[i] = i % 10 + 0.005;
            for (int j = 0; j < ic; ++j) {
                weightFp32[i * ic + j] = (i * ic + j) % res * fac + tail;
            }
        }
        ::memcpy(biastest.data(), bias.data(), oc * sizeof(float));
        ::memcpy(biasdup.data(), bias.data(), oc * sizeof(float));
        for (int k = 0; k < oc; ++k) {
            int beginIndex = k * ic;
            auto absMax = findAbsMax(weightFp32.data() + beginIndex, ic);
            wScale[k] = absMax / threshold;
            
            for (int i = 0; i < ic; ++i) {
                float* ptr = weightFp32.data() + beginIndex;
                int8_t quantVal = static_cast<int8_t>(ptr[i] / wScale[k]);
                quantWeight[k * ic + i] = quantVal;
            }
        }
        auto y     = _HybridConv(std::move(quantWeight), std::move(bias), std::move(wScale), x,
                           channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, false, nbit);
        auto yfp32 = _Conv(std::move(weightFp32), std::move(biasdup), x, {ic, oc}, {1, 1});

        if (nbit != 8) {
            std::unique_ptr<MNN::OpT> op(y->expr().first->get()->UnPack());
            op->main.AsConvolution2D()->symmetricQuan->nbits = nbit;
            y = Variable::create(Expr::create(op.get(), {x}));
            op.reset();
        }
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
#if defined (__aarch64__) && (precision == 2)
#define FLOAT_T __fp16
#else
#define FLOAT_T float
#endif
        auto yPtr  = y->readMap<FLOAT_T>();
        auto tgPtr = yfp32->readMap<FLOAT_T>();
        auto elesize = oc * batch;
        if (nbit == 8) {
            for (int i = 0; i < elesize; ++i) {
                float targetValue = tgPtr[i], computeResult = yPtr[i];
                float diff = targetValue - computeResult;
                float ratio = fabsf(diff) / fmax(targetValue, computeResult);
                if (targetValue != 0 && computeResult != 0 && ratio > 0.02) {
                    MNN_PRINT("HybridConv result Error: %f -> %f\n", targetValue, computeResult);
                    return false;
                } else if ((targetValue == 0 || computeResult == 0) && fabsf(diff) > 0.02) {
                    MNN_PRINT("HybridConv result Error: %f -> %f\n", targetValue, computeResult);
                    return false;
                }
            }
        }
        if (testSpeed) {
            x.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<FLOAT_T>();
                y->readMap<FLOAT_T>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("%s input=(%dx%dx%dx%d) output=(%dx%dx%dx%d) avg time = %f\n",
                      title.c_str(), batch, ic, 1, 1, batch, oc, 1, 1, 1.0 * time / LOOP);
        }

        return true;
    }
};

class HybridConvSpeedInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {1, 1}; // {w, h}
        INTS channel0 = {2048, 512}; // {ci, co}
        INTS channel1 = {1496, 256};
        int batch[2] = {1, 13};
        std::vector<int> kernels = {1, 1};
        std::vector<int> weightBits = {8};
        bool lowmemory = true;
        for (auto& bits : weightBits) {
            for (int n = 0; n < 2; ++n) {
                auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel0, pad, strides, dilate, batch[n], bits, precision, true);
                if (!res) {
                    MNN_ERROR("Error: low memory hybridConv when n=%d, ci=%d, c0=%d\n", batch[n], channel0[0], channel0[1]);
                    return false;
                }
            }
            for (int n = 0; n < 2; ++n) {
                auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel1, pad, strides, dilate, batch[n], bits, precision, true);
                if (!res) {
                    MNN_ERROR("Error: low memory hybridConv when n=%d, ci=%d, c0=%d\n", batch[n], channel1[0], channel1[1]);
                    return false;
                }
            }
        }
        return true;
    }
};

class HybridConvInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS channel0 = {2048, 512}; // {ci, co}
        INTS channel1 = {1496, 256};
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {1, 1}; // {w, h}
        int batch[2] = {1, 13};
        std::vector<int> kernels = {1, 1};
        std::vector<int> weightBits = {8};
        bool lowmemory = true;
        for (auto& bits : weightBits) {
            for (int n = 0; n < 2; ++n) {
                auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel0, pad, strides, dilate, batch[n], bits, precision);
                if (!res) {
                    MNN_ERROR("Error: low memory hybridConv when n=%d, ci=%d, c0=%d\n", batch[n], channel0[0], channel0[1]);
                    return false;
                }
            }
            for (int n = 0; n < 2; ++n) {
                auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel1, pad, strides, dilate, batch[n], bits, precision);
                if (!res) {
                    MNN_ERROR("Error: low memory hybridConv when n=%d, ci=%d, c0=%d\n", batch[n], channel1[0], channel1[1]);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(HybridConvInt8Test, "op/lowMemory/HybridConv");
MNNTestSuiteRegister(HybridConvSpeedInt8Test, "speed/HybridConv");
