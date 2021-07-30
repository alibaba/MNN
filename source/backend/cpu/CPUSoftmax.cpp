//
//  CPUSoftmax.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "backend/cpu/CPUSoftmax.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "CPUTensorConvert.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

int CPUSoftmax::_softmax1(const float *srcData, float *dstData, int outside, int channel, int threadNum) {
    MNN_CONCURRENCY_BEGIN(tId, threadNum)
    {
        const float *srcY = srcData + tId * channel;
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += channel * threadNum, dstY += channel * threadNum) {
            MNNSoftmax(dstY, srcY, channel);
        }
    }
    MNN_CONCURRENCY_END();
    return 0;
}
int CPUSoftmax::_softmaxCommon(const float *srcData, float *dstData, int inside, int outside, int channel,
                               float *maxValue, float *sumValue, int threadNum) {
    if (inside == 1)
        return _softmax1(srcData, dstData, outside, channel, threadNum);

    const int stepY = inside * channel;
    MNN_CONCURRENCY_BEGIN(tId, threadNum);
    {
        const float *srcY  = srcData + tId * stepY;
        float *dstY        = dstData + tId * stepY;
        float *maxValueSub = maxValue + tId * inside;

        for (int y = (int)tId; y < outside; y += threadNum, srcY += stepY * threadNum, dstY += stepY * threadNum) {
            memcpy(maxValueSub, srcY, sizeof(float) * inside);
            const float *src = srcY + inside;
            for (int c = 1; c < channel; ++c, src += inside) {
                for (int x = 0; x < inside; ++x) {
                    if (src[x] > maxValueSub[x])
                        maxValueSub[x] = src[x];
                }
            }
            src        = srcY;
            float *dst = dstY;
            for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    dst[x] = -src[x] + maxValueSub[x];
                }
            }
        }
    }
    MNN_CONCURRENCY_END();

    auto totalSize = channel * inside * outside;
    //Exp
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(totalSize);
    int sizeDivide = schedule.first;
    int scheduleNumber = schedule.second;

    MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
        int start = sizeDivide * (int)tId;
        int realSize = sizeDivide;
        if (tId == scheduleNumber -1 ) {
            realSize = totalSize - start;
        }
        if (realSize > 0) {
            MNNExp(dstData + start, dstData + start, realSize);
        }
    }
    MNN_CONCURRENCY_END();
    
    MNN_CONCURRENCY_BEGIN(tId, threadNum);
    {
        const float *srcY  = srcData + tId * stepY;
        float *dstY        = dstData + tId * stepY;
        float *sumValueSub = sumValue + tId * inside;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += stepY * threadNum, dstY += stepY * threadNum) {
            memset(sumValueSub, 0, sizeof(float) * inside);
            float *dst = dstY;
            for (int c = 0; c < channel; ++c, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    sumValueSub[x] += dst[x];
                }
            }
            dst = dstY;
            for (int c = 0; c < channel; ++c, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    dst[x] /= sumValueSub[x];
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
    return 0;
}

ErrorCode CPUSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input           = inputs[0];
    const int dimensions = input->buffer().dimensions;
    int axis = mAxis;
    if (axis < 0) {
        axis += dimensions;
    }

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;

    if (mNeedUnpackC4) {
        int totalSize = 1;
        for (int i = 1; i < dimensions; ++i) {
            totalSize *= input->length(i);
        }
        mStorage.buffer().dim[0].extent = input->length(0);
        mStorage.buffer().dim[1].extent = totalSize;
        TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        mStorage.buffer().dimensions    = 2;
        mStorage.buffer().type          = input->getType();
        backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    }

    int inside = 1;
    int dims   = input->buffer().dimensions;
    for (int i = axis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }

    if (inside != 1) { // not run _softmax1, we need maxValue Tensor and sumValue Tensor.
        int threadNum = ((CPUBackend *)backend())->threadNumber();

        mMaxValue.buffer().dim[0].extent = inside * threadNum;
        mMaxValue.buffer().dimensions    = 1;
        mMaxValue.setType(DataType_DT_FLOAT);
        backend()->onAcquireBuffer(&mMaxValue, Backend::DYNAMIC);

        mSumValue.buffer().dim[0].extent = inside * threadNum;
        mSumValue.buffer().dimensions    = 1;
        mSumValue.setType(DataType_DT_FLOAT);
        backend()->onAcquireBuffer(&mSumValue, Backend::DYNAMIC);

        backend()->onReleaseBuffer(&mMaxValue, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mSumValue, Backend::DYNAMIC);
    }

    if (mNeedUnpackC4) {
        backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

ErrorCode CPUSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputTensor        = inputs[0];
    auto outputTensor       = outputs[0];
    const auto inputDataPtr = inputTensor->host<float>();
    auto outputDataPtr      = outputTensor->host<float>();
    const int batch         = inputTensor->batch();
    const auto dims         = inputTensor->buffer().dimensions;
    int axis = mAxis;
    if (axis < 0) {
        axis += inputTensor->dimensions();
    }

    float *tempData = nullptr;
    if (mNeedUnpackC4) {
        tempData = mStorage.host<float>();
    }

    int areaInput = 1;
    for (int i = 2; i < dims; ++i) {
        areaInput *= inputTensor->length(i);
    }
    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= inputTensor->length(i);
    }
    channel = inputTensor->length(axis);
    for (int i = axis + 1; i < dims; ++i) {
        inside *= inputTensor->length(i);
    }

    int threadNum = ((CPUBackend *)backend())->threadNumber();
    if (!mNeedUnpackC4) {
        _softmaxCommon(inputDataPtr, outputDataPtr, inside, outside, channel, mMaxValue.host<float>(),
                   mSumValue.host<float>(), threadNum);
        return NO_ERROR;
    }
    auto outputSize = outputTensor->elementSize();
    int batchSize = outputSize / batch;
    auto functions = static_cast<CPUBackend*>(backend())->functions();
    int offset[] = {
        areaInput,
        areaInput
    };
    CPUTensorConverter::convert(inputDataPtr, outputDataPtr, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW, batch, areaInput, inputTensor->channel(), functions->bytes, functions);
    _softmaxCommon(outputDataPtr, tempData, inside, outside, channel, mMaxValue.host<float>(), mSumValue.host<float>(), threadNum);
    CPUTensorConverter::convert(tempData, outputDataPtr, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4, batch, areaInput, inputTensor->channel(), functions->bytes, functions);
    return NO_ERROR;
}

CPUSoftmax::CPUSoftmax(Backend *b, int axis) : MNN::Execution(b), mAxis(axis), mStorage(2), mNeedUnpackC4(false) {
    // nothing to do
}

Execution* CPUSoftmax::create(const MNN::Op *op, Backend *backend) {
    auto axis = op->main_as_Axis()->axis();
    return new CPUSoftmax(backend, axis);
}

class CPUSoftmaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto axis = op->main_as_Axis()->axis();
        return CPUSoftmax::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSoftmaxCreator, OpType_Softmax);

} // namespace MNN
