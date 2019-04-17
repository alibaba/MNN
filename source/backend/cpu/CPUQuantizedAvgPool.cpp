//
//  CPUQuantizedAvgPool.cpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUQuantizedAvgPool.hpp"
#include "CPUBackend.hpp"
#include "CPUQuantizationUtils.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "OptimizedComputer.hpp"

namespace MNN {

CPUQuantizedAvgPool::CPUQuantizedAvgPool(Backend *backend, const Op *CPUQuantizedAvgPoolOp) : Execution(backend) {
    auto CPUQuantizedAvgPool = CPUQuantizedAvgPoolOp->main_as_QuantizedAvgPool();
    mIstflite                = (CPUQuantizedAvgPool->modelFormat() == ModeFormat_TFLITE);
    mKernelWidth             = CPUQuantizedAvgPool->kernelX();
    mKernelHeight            = CPUQuantizedAvgPool->kernelY();
    mPadWidth                = CPUQuantizedAvgPool->padX();
    mPadHeight               = CPUQuantizedAvgPool->padY();
    mStrideWidth             = CPUQuantizedAvgPool->strideX();
    mStrideHeight            = CPUQuantizedAvgPool->strideY();
    mPadMode                 = CPUQuantizedAvgPool->padType();
    mOutputActivationMin     = CPUQuantizedAvgPool->outputActivationMin();
    mOutputActivationMax     = CPUQuantizedAvgPool->outputActivationMax();
}

ErrorCode CPUQuantizedAvgPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (!mIstflite) {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 3);
        mOutputActivationMin                    = 0;
        mOutputActivationMax                    = 255;
        const float minInput                    = inputs[1]->host<float>()[0];
        const float maxInput                    = inputs[2]->host<float>()[0];
        ((float *)outputs[1]->buffer().host)[0] = minInput;
        ((float *)outputs[2]->buffer().host)[0] = maxInput;
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(input->buffer().dimensions == 4);

    // input : nhwc
    const int32_t inBatch   = input->buffer().dim[0].extent;
    const int32_t inRows    = input->buffer().dim[1].extent;
    const int32_t inCols    = input->buffer().dim[2].extent;
    const int32_t inChannel = input->buffer().dim[3].extent;

    int32_t padRows          = mPadHeight;
    int32_t padCols          = mPadWidth;
    const int32_t windowRows = mKernelHeight;
    const int32_t windowCols = mKernelWidth;
    const int32_t rowStride  = mStrideHeight;
    const int32_t colStride  = mStrideWidth;
    const int32_t outHeight  = output->buffer().dim[1].extent;
    const int32_t outWidth   = output->buffer().dim[2].extent;

    switch (mPadMode) {
        case PoolPadType_CAFFE:
            MNN_ASSERT(false);
            break;
        case PoolPadType_VALID:
            padRows = padCols = 0;
            break;
        case PoolPadType_SAME:
            auto widthNeeded  = (outWidth - 1) * colStride + windowCols - inCols;
            auto heightNeeded = (outHeight - 1) * rowStride + windowRows - inRows;
            mPadWidth         = widthNeeded > 0 ? widthNeeded / 2 : 0;
            mPadHeight        = heightNeeded > 0 ? heightNeeded / 2 : 0;
            break;
    }

    uint8_t *inputPtr  = (uint8_t *)input->buffer().host;
    uint8_t *outputPtr = (uint8_t *)output->buffer().host;

    std::vector<int> inputDims;
    inputDims.push_back(inBatch);
    inputDims.push_back(inRows);
    inputDims.push_back(inCols);
    inputDims.push_back(inChannel);

    std::vector<int> outputDims;
    outputDims.push_back(output->length(0));
    outputDims.push_back(output->length(1));
    outputDims.push_back(output->length(2));
    outputDims.push_back(output->length(3));
    Optimized::AveragePool(inputPtr, inputDims, mStrideWidth, mStrideHeight, mPadWidth, mPadHeight, mKernelWidth,
                           mKernelHeight, mOutputActivationMin, mOutputActivationMax, outputPtr, outputDims);

    return NO_ERROR;
}

class CPUQuantizedAvgPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedAvgPool(backend, op);
    }
};
REGISTER_CPU_OP_CREATOR(CPUQuantizedAvgPoolCreator, OpType_QuantizedAvgPool);
} // namespace MNN
