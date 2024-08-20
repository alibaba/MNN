//
//  CommonOpCreator.hpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonOpCreator_hpp
#define CommonOpCreator_hpp
#include "TestUtils.h"
#include "core/IDSTEncoder.hpp"
namespace MNN {
static PadMode _convertPadMode(Express::PaddingMode mode) {
    switch (mode) {
        case Express::PaddingMode::CAFFE:
            return PadMode_CAFFE;
        case Express::PaddingMode::VALID:
            return PadMode_VALID;
        case Express::PaddingMode::SAME:
            return PadMode_SAME;
        default:
            break;
    }
    return PadMode_CAFFE;
}
static Express::VARP _HybridConv(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& alpha, Express::VARP x, std::vector<int> channel, std::vector<int> kernelSize,
                          Express::PaddingMode pad, std::vector<int> stride, std::vector<int> dilate, int group, std::vector<int> pads, bool relu, bool relu6, int nbits, bool async) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    int kSize = kernelSize[0] * kernelSize[1] * channel[0] / group;
    int kNum = channel[1];
    int clampMin = -(1 << (nbits - 1));
    conv2D->quanParameter = std::move(IDSTEncoder::encode(weight.data(), alpha, kSize, kNum, async, nullptr, clampMin, nbits));
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
    conv2D->weight.clear();
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = bias;
    return (Express::Variable::create(Express::Expr::create(convOp.get(), {x})));
}

static float findAbsMax(const float *weights, const int count) {
    float absMax = 0.00000001f;
    for (int i = 0; i < count; i++) {
        float value = fabs(weights[i]);
        if (value > absMax) {
            absMax = value;
        }
    }

    return absMax;
}

static std::pair<float,float> findMinMax(const float *weights, const int count) {
    float absMax = 0.00000001f;
    if (0 == count) {
        return std::make_pair(0.0f, 1.0f);
    }
    float minV = weights[0];
    float maxV = weights[0];
    for (int i = 1; i < count; i++) {
        float value = weights[i];
        if (value > maxV) {
            maxV = value;
        }
        if (value < minV) {
            minV = value;
        }
    }

    return std::make_pair(minV, maxV);
}

};


#endif
