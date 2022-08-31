//
//  CPUQuantizedAvgPool.cpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "backend/cpu/CPUQuantizedAvgPool.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "backend/cpu/compute/OptimizedComputer.hpp"

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


ErrorCode CPUQuantizedAvgPool::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(input->buffer().dimensions == 4);

    int32_t inBatch   = input->buffer().dim[0].extent;
    int32_t inRows    = input->buffer().dim[2].extent;
    int32_t inCols    = input->buffer().dim[3].extent;
    int32_t inChannel = input->buffer().dim[1].extent;

    const int32_t windowRows = mKernelHeight;
    const int32_t windowCols = mKernelWidth;
    const int32_t rowStride  = mStrideHeight;
    const int32_t colStride  = mStrideWidth;
    int32_t outHeight  = output->buffer().dim[2].extent;
    int32_t outWidth   = output->buffer().dim[3].extent;

    switch (mPadMode) {
        case PoolPadType_CAFFE:
            MNN_ASSERT(false);
            break;
        case PoolPadType_VALID:
            mPadHeight = mPadWidth = 0;
            break;
        case PoolPadType_SAME:
            auto widthNeeded  = (outWidth - 1) * colStride + windowCols - inCols;
            auto heightNeeded = (outHeight - 1) * rowStride + windowRows - inRows;
            mPadWidth         = widthNeeded > 0 ? widthNeeded / 2 : 0;
            mPadHeight        = heightNeeded > 0 ? heightNeeded / 2 : 0;
            break;
    }

    mInputDims = {inBatch, inRows, inCols, inChannel};
    mOutputDims = {output->batch(), output->height(), output->width(), output->channel()};

    return NO_ERROR;
}


ErrorCode CPUQuantizedAvgPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {


    uint8_t *inputPtr  = inputs[0]->host<uint8_t>();
    uint8_t *outputPtr = outputs[0]->host<uint8_t>();

    Optimized::AveragePool(inputPtr, mInputDims, mStrideWidth, mStrideHeight, mPadWidth, mPadHeight, mKernelWidth,
                               mKernelHeight, mOutputActivationMin, mOutputActivationMax, outputPtr, mOutputDims);

    return NO_ERROR;
}

class CPUQuantizedAvgPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedAvgPool(backend, op);
    }
};
} // namespace MNN
#endif

namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUQuantizedAvgPoolCreator, OpType_QuantizedAvgPool);
};
