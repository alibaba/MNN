//
//  CPUEltwiseInt8.cpp
//  MNN
//
//  Created by MNN on 2019/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUEltwiseInt8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

extern "C" {
void MNNScaleAddInt8(int8_t* dst, const int8_t* src0, const int8_t* src1, const float* scale0, const float* scale1,
                     const float* outputScale, const size_t size);
}

namespace MNN {

CPUEltwiseInt8::CPUEltwiseInt8(Backend* backend, const Op* op) : Execution(backend) {
    isEltwiseInt8 = op->type() == OpType_EltwiseInt8;
    if (!isEltwiseInt8) {
        return;
    }
    auto param    = op->main_as_EltwiseInt8();
    auto copyData = [=](std::shared_ptr<Tensor>& tensor, const QuantizedFloatParam* scale) {
        const int size = scale->tensorScale()->size();
        tensor.reset(Tensor::createDevice<float>({ALIGN_UP4(size)}));
        bool success = backend->onAcquireBuffer(tensor.get(), Backend::STATIC);
        if (!success) {
            return;
        }
        ::memset(tensor->host<float>(), 0, ALIGN_UP4(size) * sizeof(float));
        ::memcpy(tensor->host<float>(), scale->tensorScale()->data(), size * sizeof(float));
    };

    copyData(mInput0Scales, param->inputQuan0());
    copyData(mInput1Scales, param->inputQuan1());
    copyData(mOutputScales, param->outputQuan());
}

CPUEltwiseInt8::~CPUEltwiseInt8() {
    if (!isEltwiseInt8) {
        return;
    }
    backend()->onReleaseBuffer(mInput0Scales.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mInput1Scales.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mOutputScales.get(), Backend::STATIC);
}

ErrorCode CPUEltwiseInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0           = inputs[0];
    auto input1           = inputs[1];
    auto output           = outputs[0];
    const int batch       = input0->batch();
    const int icDiv4      = UP_DIV(input0->channel(), 4);
    const int batchStride = input0->stride(0);
    const int width       = input0->width();
    const int height      = input0->height();
    const int oc4Stride   = width * height;

    const float *scale0Ptr, *scale1Ptr, *outputScalePtr;
    std::vector<float> scale0(input0->channel()), scale1(input1->channel()), outputScale(output->channel());
    if (isEltwiseInt8) {
        scale0Ptr      = mInput0Scales->host<float>();
        scale1Ptr      = mInput1Scales->host<float>();
        outputScalePtr = mOutputScales->host<float>();
    } else {
        std::fill(scale0.begin(), scale0.end(), TensorUtils::getDescribe(input0)->quantAttr->scale);
        std::fill(scale1.begin(), scale1.end(), TensorUtils::getDescribe(input1)->quantAttr->scale);
        std::fill(outputScale.begin(), outputScale.end(), 1 / TensorUtils::getDescribe(output)->quantAttr->scale);
        scale0Ptr = scale0.data();
        scale1Ptr = scale1.data();
        outputScalePtr = outputScale.data();
    }

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
#ifdef MNN_USE_SSE
        const auto src0Batch = input0->host<uint8_t>() + bIndex * batchStride;
        const auto src1Batch = input1->host<uint8_t>() + bIndex * batchStride;
        auto dstBatch        = output->host<uint8_t>() + bIndex * batchStride;
#else
        const auto src0Batch = input0->host<int8_t>() + bIndex * batchStride;
        const auto src1Batch = input1->host<int8_t>() + bIndex * batchStride;
        auto dstBatch        = output->host<int8_t>() + bIndex * batchStride;
#endif
        MNN_CONCURRENCY_BEGIN(tId, icDiv4) {
            const auto src0ChannelPtr        = src0Batch + tId * oc4Stride * 4;
            const auto src1ChannelPtr        = src1Batch + tId * oc4Stride * 4;
            const auto scale0ChannelPtr      = scale0Ptr + tId * 4;
            const auto scale1ChannelPtr      = scale1Ptr + tId * 4;
            const auto outputScaleChannelPtr = outputScalePtr + tId * 4;
            auto dstChannelPtr               = dstBatch + tId * oc4Stride * 4;
#ifdef MNN_USE_NEON
            MNNScaleAddInt8(dstChannelPtr, src0ChannelPtr, src1ChannelPtr, scale0ChannelPtr, scale1ChannelPtr,
                            outputScaleChannelPtr, oc4Stride);
#elif defined(MNN_USE_SSE)
            const uint8_t zeroPoint = 128;
            for (int i = 0; i < oc4Stride; ++i) {
                for (int k = 0; k < 4; ++k) {
                    float sum = static_cast<float>((int8_t)(src0ChannelPtr[i * 4 + k] - zeroPoint)) * scale0ChannelPtr[k] +
                                static_cast<float>((int8_t) (src1ChannelPtr[i * 4 + k] - zeroPoint)) * scale1ChannelPtr[k];
                    float value              = sum * outputScaleChannelPtr[k];
                    dstChannelPtr[i * 4 + k] = static_cast<uint8_t>(std::max(std::min(value, 127.0f), -127.0f)) + zeroPoint;
                }
            }
#else   
            for (int i = 0; i < oc4Stride; ++i) {
                for (int k = 0; k < 4; ++k) {
                    float sum = static_cast<float>(src0ChannelPtr[i * 4 + k]) * scale0ChannelPtr[k] +
                                static_cast<float>(src1ChannelPtr[i * 4 + k]) * scale1ChannelPtr[k];
                    float value              = sum * outputScaleChannelPtr[k];
                    dstChannelPtr[i * 4 + k] = static_cast<int8_t>(std::max(std::min(value, 127.0f), -127.0f));
                }
            }
#endif
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}

class CPUEltwiseInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUEltwiseInt8(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUEltwiseInt8Creator, OpType_EltwiseInt8);

} // namespace MNN
