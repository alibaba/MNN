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

namespace MNN {

CPUConvolution::CPUConvolution(const Convolution2DCommon *convOp, Backend *b) : MNN::Execution(b), mCommon(convOp) {
    mPostFunction = getPostFunction();
}

int CPUConvolution::reorderWeightSize(int depth, int outputCount, int kernelSize, int unit) {
    int unit2 = unit * unit;
    return UP_DIV(outputCount, unit) * UP_DIV(depth, unit) * kernelSize * unit2;
}

void CPUConvolution::reorderWeight(float *dest, const float *source, int depth, int outputCount, int kernelSize,
                                   int unit) {
    int unit2             = unit * unit;
    int alignedWeightSize = UP_DIV(outputCount, unit) * UP_DIV(depth, unit) * kernelSize * unit2;
    int cur               = 0;
    int batch_4           = ALIGN_UP4(outputCount) / unit;
    for (int b = 0; b < outputCount; ++b) {
        int b_4      = b / unit;
        float *dst_b = dest + b_4 * (alignedWeightSize / batch_4);
        int mx       = b % unit;
        for (int d = 0; d < depth; ++d) {
            int my       = d % unit;
            int d_4      = d / unit;
            float *dst_d = dst_b + d_4 * kernelSize * unit2;
            for (int y = 0; y < kernelSize; ++y) {
                float *dst_y          = dst_d + y * unit2;
                dst_y[unit * my + mx] = source[cur++];
            }
        }
    }
}

ErrorCode CPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        mPadX = padNeededWidth / 2;
        mPadY = padNeededHeight / 2;
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
