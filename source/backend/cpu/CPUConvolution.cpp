//
//  CPUConvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUConvolution.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "backend/cpu/compute/ConvolutionFloatFactory.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/ConvolutionCommon.hpp"

namespace MNN {

CPUConvolution::CPUConvolution(const Convolution2DCommon *convOp, Backend *b) : MNN::Execution(b), mCommon(convOp) {
    mPostFunction = getPostFunction();
}

int CPUConvolution::reorderWeightSize(int depth, int outputCount, int kernelSize, int unit) {
    int unit2 = unit * unit;
    return UP_DIV(outputCount, unit) * UP_DIV(depth, unit) * kernelSize * unit2;
}

void CPUConvolution::reorderWeight(float *dest, const float *source, int depth, int outputCount, int kernelSize,
                                   float *cache) {
    auto alignDepth = ALIGN_UP4(depth);
    for (int b = 0; b < outputCount; ++b) {
        auto dst = cache + b * alignDepth * kernelSize;
        auto src = source + b * depth * kernelSize;
        MNNPackC4(dst, src, kernelSize, depth);
    }
    MNNPackC4(dest, cache, kernelSize * ALIGN_UP4(depth), outputCount);
    auto count = UP_DIV(depth, 4) * kernelSize * UP_DIV(outputCount, 4);
    MNNReorder4x4ByPlatform(dest, count);
}

ErrorCode CPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto pad = ConvolutionCommon::convolutionPad(input, output, mCommon);
    mPadY = pad.second;
    mPadX = pad.first;
    return NO_ERROR;
}

CPUConvolution::POSTFUNCTION CPUConvolution::getPostFunction() const {
    if (mCommon->relu()) {
        return MNNAddBiasRelu;
    }
    if (mCommon->relu6()) {
        return MNNAddBiasRelu6;
    }
    return MNNAddBias;
}

class ConvolutionFactory : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return ConvolutionFloatFactory::create(inputs, outputs, op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
} // namespace MNN
