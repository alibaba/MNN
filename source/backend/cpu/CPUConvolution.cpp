//
//  CPUConvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConvolution.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "compute/ConvolutionFloatFactory.h"
//#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"

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
    AUTOTIME;
    auto alignDepth = ALIGN_UP4(depth);
    for (int b = 0; b < outputCount; ++b) {
        auto dst = cache + b * alignDepth * kernelSize;
        auto src = source + b * depth * kernelSize;
        MNNPackC4(dst, src, kernelSize, depth);
    }
    MNNPackC4(dest, cache, kernelSize * ALIGN_UP4(depth), outputCount);
}

ErrorCode CPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        mPadX               = padNeededWidth / 2;
        mPadY               = padNeededHeight / 2;
        return NO_ERROR;
    }
    mPadX = mCommon->padX();
    mPadY = mCommon->padY();

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
