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
#include "CPUCast.hpp"

namespace MNN {
static void ___MNNSoftmax(float* dest, const float* source, size_t size, MNNBinaryExecute mulfunction) {
    float exprOffset[4] = {
        1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    // Compute Max
    {
        int32_t inputCountUnit = size / (4 * 2);
        int32_t remain         = size - (inputCountUnit * 4 * 2);
        float Max = source[0];
        if (inputCountUnit > 0) {
            float maxArray[4] = {Max, Max, Max, Max};
            MNNMaxFloat((float*)source, maxArray, inputCountUnit);
            for (int i = 0; i < 4; i++) {
                Max = ALIMAX(Max, maxArray[i]);
            }
        }
        if (remain > 0) {
            int currentIndex = inputCountUnit * 4 * 2;
            for (int i = 0; i < remain; i++) {
                float currentInputData = source[currentIndex + i];
                Max = ALIMAX(Max, currentInputData);
            }
        }
        exprOffset[2] = -Max;
    }
    MNNExp(dest, source, exprOffset, size);
    float sumDiv = 1.0f / exprOffset[3];
    mulfunction(dest, dest, &sumDiv, size, 1);
}

int CPUSoftmax::_softmaxCommon(const uint8_t *srcData, uint8_t *dstData) {
    auto cpuBn = static_cast<CPUBackend*>(backend());
    auto core = cpuBn->functions();
    auto fp32Core = core;
    if (core->bytes != 4) {
        fp32Core = MNNGetCoreFunctions();
    }
    MNNBinaryExecute addFunction;
    MNNUnaryExecute recFunction;
    MNNBinaryExecute mulFunction;
    mulFunction = fp32Core->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_MUL);

    auto bytes = core->bytes;
    int threadNumber = ALIMIN(cpuBn->threadNumber(), mOutside);
    int outsideStride = mChannel * mInside;
    if (mInside > core->pack && mChannel < core->pack) {
        auto maxFunction = core->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_MAXIMUM);
        auto subFunction = core->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_SUB);
        addFunction = fp32Core->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_ADD);
        recFunction = fp32Core->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_RECIPROCAL, 1);//Use high precision
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            float* tempOutput = nullptr;
            float* tempInput = nullptr;
            if (mTmpInput.ptr()) {
                tempInput = (float*)(mTmpInput.ptr() + tId * outsideStride * sizeof(float));
            }
            
            if (mTmpOutput.ptr()) {
                tempOutput = (float*)(mTmpOutput.ptr() + tId * outsideStride * sizeof(float));
            }

            for (int o=tId; o<mOutside; o+=threadNumber) {
                auto srcO = srcData + o * outsideStride * mLowOrInt8;
                auto dstO = dstData + o * outsideStride * mLowOrInt8;
                // Max
                if (mLowOrInt8 == 1) {
                    CPUCastCreator::cast(srcO, tempInput, CPUCastCreator::INT8_TO_FlOAT, outsideStride, mInQuantAttr->scale, mInQuantAttr->zero, mInQuantAttr->min, mInQuantAttr->max, cpuBn);
                    ::memcpy(tempOutput, tempInput, mInside * 4);
                    for (int z = 1; z < mChannel; ++z) {
                        maxFunction(tempOutput, tempOutput, tempInput + z * mInside, mInside, -1);
                    }
                } else {
                    ::memcpy(tempInput, srcO, mInside * mLowOrInt8);
                    for (int z = 1; z < mChannel; ++z) {
                        maxFunction(tempInput, tempInput, srcO + z * mInside * mLowOrInt8, mInside, -1);
                    }
                }
                // Sub Max
                for (int z=0; z<mChannel; ++z) {
                    if (mLowOrInt8 == 1) {
                        subFunction(tempInput + z * mInside, tempInput + z * mInside, tempOutput, mInside, -1);
                    } else {
                        subFunction(dstO + z * mInside * mLowOrInt8, srcO + z * mInside * mLowOrInt8, tempInput, mInside, -1);
                    }
                }
                // Exp
                float exprOffset[4] = {
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f
                };
                auto workSrc = (float*)srcO;
                auto workDst = (float*)dstO;
                if (mLowOrInt8 != 4) {
                    workSrc = tempInput;
                    workDst = tempOutput;
                    if (mLowOrInt8 == 2) {
                        core->MNNLowpToFp32((int16_t*)(dstO), workSrc, outsideStride);
                    }
                }
                // Use Fp32 to compute Begin
                MNNExp(workDst, workSrc, exprOffset, outsideStride);

                // Sum to tempInput
                ::memcpy(tempInput, workDst, mInside * sizeof(float));

                for (int z=1; z<mChannel; ++z) {
                    addFunction(tempInput, tempInput, workDst + z * mInside, mInside, -1);
                }
                recFunction(tempInput, tempInput, mInside);
                for (int z=0; z<mChannel; ++z) {
                    mulFunction(workDst + z * mInside, workDst + z * mInside, tempInput, mInside, -1);
                }
                // Use Fp32 Compute end
                if (mLowOrInt8 == 2) {
                    core->MNNFp32ToLowp(workDst, (int16_t*)(dstO), outsideStride);
                } else if (mLowOrInt8 == 1) {
                    CPUCastCreator::cast(workDst, dstO, CPUCastCreator::FlOAT_TO_INT8, outsideStride, mOutQuantAttr->scale, mOutQuantAttr->zero, mOutQuantAttr->min, mOutQuantAttr->max, cpuBn);
                } else {
                    // do nothing.
                }
            }
        };
        MNN_CONCURRENCY_END();
        return 0;
    }
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        float* tempInput;
        float* tempOutput;
        if (mTmpInput.ptr()) {
            tempInput = (float*)(mTmpInput.ptr() + tId * outsideStride * sizeof(float));
        }
        if (mTmpOutput.ptr()) {
            tempOutput = (float*)(mTmpOutput.ptr() + tId * outsideStride * sizeof(float));
        }
        for (int o=tId; o<mOutside; o+=threadNumber) {
            auto srcO = srcData + o * outsideStride * mLowOrInt8;
            auto dstO = dstData + o * outsideStride * mLowOrInt8;
            auto workSrc = (float*)srcO;
            auto workDst = (float*)dstO;
            // Pretreat
            if (1 == mInside) {
                if (mLowOrInt8 == 2) {
                    core->MNNLowpToFp32((int16_t*)(srcO), tempInput, outsideStride);
                    workDst = tempOutput;
                    workSrc = tempInput;
                } else if (mLowOrInt8 == 1) {
                    CPUCastCreator::cast(srcO, tempInput, CPUCastCreator::INT8_TO_FlOAT, outsideStride, mInQuantAttr->scale, mInQuantAttr->zero, mInQuantAttr->min, mInQuantAttr->max, cpuBn);
                    workDst = tempOutput;
                    workSrc = tempInput;
                }
            } else {
                int dims[] = {
                    mChannel,
                    mInside,
                    mInside,
                    mChannel
                };
                if (mLowOrInt8 == 2) {
                    MNN_ASSERT(bytes == 2);
                    MNNTranspose16Bit((int16_t*)tempOutput, (int16_t*)(srcO), dims);
                    core->MNNLowpToFp32((int16_t*)tempOutput, tempInput, outsideStride);
                    workDst = tempOutput;
                    workSrc = tempInput;
                } else if (mLowOrInt8 == 1) {
                    CPUCastCreator::cast(srcO, tempOutput, CPUCastCreator::INT8_TO_FlOAT, outsideStride, mInQuantAttr->scale, mInQuantAttr->zero, mInQuantAttr->min, mInQuantAttr->max, cpuBn);
                    MNNTranspose32Bit((int32_t*)tempInput, (int32_t*)tempOutput, dims);
                    workDst = tempOutput;
                    workSrc = tempInput;
                } else {
                    // Use output to cache transpoe result
                    MNNTranspose32Bit((int32_t*)dstO, (int32_t*)(srcO), dims);
                    workDst = tempInput;
                    workSrc = (float*)dstO;
                }
            }
            for (int v=0; v<mInside; ++v) {
                //TODO: Fix x86 compute error and use the same function
#ifdef MNN_USE_SSE
                MNNSoftmax(workDst+v*mChannel, workSrc+v*mChannel, mChannel);
#else
                ___MNNSoftmax(workDst+v*mChannel, workSrc+v*mChannel, mChannel, mulFunction);
#endif
            }
            // PostTreat
            if (1 == mInside) {
                if (mLowOrInt8 == 2) {
                    core->MNNFp32ToLowp(tempOutput, (int16_t*)(dstO), outsideStride);
                } else if (mLowOrInt8 == 1) {
                    CPUCastCreator::cast(tempOutput, dstO, CPUCastCreator::FlOAT_TO_INT8, outsideStride, mOutQuantAttr->scale, mOutQuantAttr->zero, mOutQuantAttr->min, mOutQuantAttr->max, cpuBn);
                }
            } else {
                int dims[] = {
                    mInside,
                    mChannel,
                    mChannel,
                    mInside
                };
                if (mLowOrInt8 == 2) {
                    MNN_ASSERT(bytes == 2); 
                    core->MNNFp32ToLowp((float*)tempOutput, (int16_t*)tempInput, outsideStride);
                    MNNTranspose16Bit((int16_t*)dstO, (int16_t*)(tempInput), dims);
                } else if (mLowOrInt8 == 1) {
                    MNNTranspose32Bit((int32_t*)tempInput, (int32_t*)tempOutput, dims);
                    CPUCastCreator::cast(tempInput, dstO, CPUCastCreator::FlOAT_TO_INT8, outsideStride, mOutQuantAttr->scale, mOutQuantAttr->zero, mOutQuantAttr->min, mOutQuantAttr->max, cpuBn);
                } else {
                    MNNTranspose32Bit((int32_t*)dstO, (int32_t*)(tempInput), dims);
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

    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= input->length(i);
    }
    channel = input->length(axis);
    for (int i = axis + 1; i < dimensions; ++i) {
        inside *= input->length(i);
    }
    mInside = inside;
    mOutside = outside;
    mChannel = channel;

    mLowOrInt8 = 4;
    if (static_cast<CPUBackend*>(backend())->functions()->bytes != 4) {
        mLowOrInt8 = 2;
    }
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mLowOrInt8 = 1;
    }
    mInQuantAttr = TensorUtils::getDescribe(inputs[0])->quantAttr;
    mOutQuantAttr = TensorUtils::getDescribe(outputs[0])->quantAttr;
    auto cpuBn = static_cast<CPUBackend*>(backend());
    if (inside != 1 || mLowOrInt8 != 4) { // not run _softmax1, we need maxValue Tensor and sumValue Tensor.
        int threadNum = cpuBn->threadNumber();
        auto buf = cpuBn->getBufferAllocator();
        threadNum = ALIMIN(threadNum, outside);
        
        mTmpInput = buf->alloc(threadNum * inside * channel * sizeof(float));
        if (mLowOrInt8 != 4) {
            mTmpOutput = buf->alloc(threadNum * inside * channel * sizeof(float));
            buf->free(mTmpOutput);
        }
        buf->free(mTmpInput);
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

    float *tempData = nullptr;
    if (mNeedUnpackC4) {
        tempData = mStorage.host<float>();
    }

    int areaInput = 1;
    for (int i = 2; i < dims; ++i) {
        areaInput *= inputTensor->length(i);
    }

    int threadNum = ((CPUBackend *)backend())->threadNumber();
    if (!mNeedUnpackC4) {
        _softmaxCommon((uint8_t*)inputDataPtr, (uint8_t*)outputDataPtr);
        return NO_ERROR;
    }
    auto functions = static_cast<CPUBackend*>(backend())->functions();
    CPUTensorConverter::convert(inputDataPtr, outputDataPtr, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW, batch, areaInput, inputTensor->channel(), mLowOrInt8, functions);
    _softmaxCommon((uint8_t*)outputDataPtr, (uint8_t*)tempData);
    CPUTensorConverter::convert(tempData, outputDataPtr, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4, batch, areaInput, inputTensor->channel(), mLowOrInt8, functions);
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
        return CPUSoftmax::create(op, backend);
        
    }
};

REGISTER_CPU_OP_CREATOR(CPUSoftmaxCreator, OpType_Softmax);

} // namespace MNN
