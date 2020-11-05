//
//  CPUDilation2D.cpp
//  MNN
//
//  Created by MNN on 2018/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUDilation2D.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"

#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

namespace MNN {

CPUDilation2D::CPUDilation2D(Backend *b, const MNN::Op *op) : Execution(b) {
    auto convOp = op->main_as_Convolution2D();
    auto common = convOp->common();
    const int kh = common->kernelY(), kw = common->kernelX();
    const int depth = common->outputCount();
    mWeight.reset(Tensor::createDevice<float>({UP_DIV(depth, 4), kh * kw * 4}));
    bool succ = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Failed to acquire memory for filters\n");
        return;
    }
    MNNPackC4(mWeight->host<float>(), convOp->weight()->data(), kh * kw, depth);
    mPadMode = common->padMode();
    mKernelSize[0] = kh;
    mKernelSize[1] = kw;
    mStrides[0] = common->strideY();
    mStrides[1] = common->strideX();
    mDilations[0] = common->dilateY();
    mDilations[1] = common->dilateX();
}

CPUDilation2D::~CPUDilation2D() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDilation2D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mPads[0] = mPads[1] = 0;
    if (mPadMode == PadMode_SAME) {
        int inputHeightNeed = (outputs[0]->height() - 1) * mStrides[0] + (mKernelSize[0] - 1) * mDilations[0] + 1;
        int inputWidthNeed = (outputs[0]->width() - 1) * mStrides[1] + (mKernelSize[1] - 1) * mDilations[1] + 1;
        mPads[0] = (inputHeightNeed - inputs[0]->height()) / 2;
        mPads[1] = (inputWidthNeed - inputs[0]->height()) / 2;
    }
    return NO_ERROR;
}

ErrorCode CPUDilation2D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0], output = outputs[0];
    
    const int threadNumber = reinterpret_cast<CPUBackend*>(backend())->threadNumber();
    const int inputHeight = input->height(), inputWidth = input->width();
    const int outputHeight = output->height(), outputWidth = output->width();
    const int outputDepth4 = UP_DIV(output->channel(), 4), depthStep = UP_DIV(outputDepth4, threadNumber);
    const int kernelY = mKernelSize[0], kernelX = mKernelSize[1];
    const int strideY = mStrides[0], strideX = mStrides[1];
    const int dilationY = mDilations[0], dilationX = mDilations[1];
    const int padY = mPads[0], padX = mPads[1];
    
    auto computeFunc = [=](int tId, const float* inputOrigin, const float* weight, float* outputOrigin) {
        const int depthFrom = tId * depthStep, depthEnd = ALIMIN(depthFrom + depthStep, outputDepth4);
        if (depthFrom >= depthEnd) {
            return;
        }
        for (int d = depthFrom; d < depthEnd; ++d) {
            auto inputData = inputOrigin + d * inputHeight * inputWidth * 4;
            auto weightData = weight + d * kernelY * kernelX * 4;
            auto outputData = outputOrigin + d * outputHeight * outputWidth * 4;
            for (int h = 0; h < outputHeight; ++h) {
                const int hOffset = h * strideY - padY;
                for (int w = 0; w < outputWidth; ++w) {
                    const int wOffset = w * strideX - padX;
                    Vec4 result = 0;
                    for (int kh = 0; kh < kernelY; ++kh) {
                        const int hOffset_ = hOffset + kh * dilationY;
                        if (hOffset_ < 0 || hOffset_ >= inputHeight) {
                            continue;
                        }
                        for (int kw = 0; kw < kernelX; ++kw) {
                            const int wOffset_ = wOffset + kw * dilationX;
                            if (wOffset_ < 0 || wOffset_ >= inputWidth) {
                                continue;
                            }
                            auto tmp = Vec4::load(inputData + (hOffset_ * inputWidth + wOffset_) * 4) + Vec4::load(weightData + (kh * kernelX + kw) * 4);
                            result = Vec4::max(result, tmp);
                        }
                    }
                    Vec4::save(outputData + (h * outputWidth + w) * 4, result);
                }
            }
        }
    };
    
    for (int batch = 0; batch < output->batch(); ++batch) {
        const float* inputOrigin = input->host<float>() + batch * input->stride(0);
        const float* weight = mWeight->host<float>();
        float* outputOrigin = output->host<float>() + batch * output->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            computeFunc((int)tId, inputOrigin, weight, outputOrigin);
        }
        MNN_CONCURRENCY_END()
    }
    return NO_ERROR;
}

class CPUDilation2DCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUDilation2D(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDilation2DCreator, OpType_Dilation2D);

}
