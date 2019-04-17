//
//  CPUQuantizedMaxPool.cpp
//  MNN
//
//  Created by MNN on 2018/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUQuantizedMaxPool.hpp"
#include "CPUBackend.hpp"
#include "CPUQuantizationUtils.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUQuantizedMaxPool::CPUQuantizedMaxPool(Backend *backend, const Op *op) : Execution(backend) {
    auto mp       = op->main_as_QuantizedMaxPool();
    mIstflite     = (mp->modelFormat() == ModeFormat_TFLITE);
    mKernelWidth  = mp->kernelX();
    mKernelHeight = mp->kernelY();
    mPadWidth     = mp->padX();
    mPadHeight    = mp->padY();
    mStrideWidth  = mp->strideX();
    mStrideHeight = mp->strideY();
    mPadMode      = mp->padType();
}

ErrorCode CPUQuantizedMaxPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(input->buffer().dimensions == 4);

    if (!mIstflite) {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 3);
        const float minInput                    = inputs[1]->host<float>()[0];
        const float maxInput                    = inputs[2]->host<float>()[0];
        ((float *)outputs[1]->buffer().host)[0] = minInput;
        ((float *)outputs[2]->buffer().host)[0] = maxInput;
    }

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
        case PoolPadType_VALID:
            padRows = padCols = 0;
            break;
        case PoolPadType_SAME: {
            auto widthNeeded  = (outWidth - 1) * colStride + windowCols - inCols;
            auto heightNeeded = (outHeight - 1) * rowStride + windowRows - inRows;
            mPadWidth         = widthNeeded > 0 ? widthNeeded / 2 : 0;
            mPadHeight        = heightNeeded > 0 ? heightNeeded / 2 : 0;
            break;
        }
        default:
            MNN_ASSERT(false);
            break;
    }

    uint8_t *inputPtr            = (uint8_t *)input->buffer().host;
    uint8_t *outputPtr           = (uint8_t *)output->buffer().host;
    const uint8_t minAsQuantized = 0;

    for (int batchIndex = 0; batchIndex < inBatch; batchIndex++) {
        uint8_t *outputBatchPtr = outputPtr + batchIndex * outWidth * outHeight * inChannel;
        uint8_t *inputBatchPtr  = inputPtr + batchIndex * inCols * inRows * inChannel;

        for (int channelIndex = 0; channelIndex < inChannel; channelIndex++) {
            for (int outHeightIndex = 0; outHeightIndex < outHeight; outHeightIndex++) {
                for (int outWidthIndex = 0; outWidthIndex < outWidth; outWidthIndex++) {
                    uint8_t maxTemp          = std::numeric_limits<uint8_t>::min();
                    int32_t inputHeightIndex = outHeightIndex * rowStride - padRows;
                    int32_t inputWidthIndex  = outWidthIndex * colStride - padCols;
                    uint8_t *outputTemp      = (uint8_t *)(outputBatchPtr + outHeightIndex * outWidth * inChannel +
                                                      outWidthIndex * inChannel + channelIndex);
                    for (int windowRowsIndex = 0; windowRowsIndex < windowRows; windowRowsIndex++) {
                        for (int windowColsIndex = 0; windowColsIndex < windowCols; windowColsIndex++) {
                            if (((inputWidthIndex + windowColsIndex) < 0) ||
                                ((inputWidthIndex + windowColsIndex) >= inCols) ||
                                ((inputHeightIndex + windowRowsIndex) < 0) ||
                                ((inputHeightIndex + windowRowsIndex) >= inRows)) {
                                maxTemp = std::max(minAsQuantized, maxTemp);
                            } else {
                                maxTemp = std::max(
                                    inputBatchPtr[(inputHeightIndex + windowRowsIndex) * inCols * inChannel +
                                                  (inputWidthIndex + windowColsIndex) * inChannel + channelIndex],
                                    maxTemp);
                            }
                        }
                    }
                    *outputTemp = maxTemp;
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUQuantizedMaxPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedMaxPool(backend, op);
    }
};
REGISTER_CPU_OP_CREATOR(CPUQuantizedMaxPoolCreator, OpType_QuantizedMaxPool);
} // namespace MNN
