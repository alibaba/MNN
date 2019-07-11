//
//  CPUInt8ToFloat.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUInt8ToFloat.hpp"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Macro.h"

extern "C" {
void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size);
}

namespace MNN {

CPUInt8ToFloat::CPUInt8ToFloat(Backend* backend, const MNN::Op* param) : Execution(backend) {
    auto scale         = param->main_as_QuantizedFloatParam();
    const int scaleLen = scale->tensorScale()->size();
    mScales.reset(Tensor::createDevice<float>({ALIGN_UP4(scaleLen)}));
    mValid = backend->onAcquireBuffer(mScales.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    memset(mScales->host<float>(), 0, ALIGN_UP4(scaleLen) * sizeof(float));
    memcpy(mScales->host<float>(), scale->tensorScale()->data(), scaleLen * sizeof(float));
}
CPUInt8ToFloat::~CPUInt8ToFloat() {
    backend()->onReleaseBuffer(mScales.get(), Backend::STATIC);
}
ErrorCode CPUInt8ToFloat::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const auto inputDataPtr = input->host<int8_t>();
    auto outputDataPtr      = output->host<float>();
    const auto scaleDataPtr = mScales->host<float>();
    const int channels      = input->channel();
    const int icDiv4        = UP_DIV(channels, 4);
    const int batch         = input->batch();
    const int batchStride   = input->stride(0);
    const int width         = input->width();
    const int height        = input->height();
    const int oc4Stride     = width * height;

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcBatch = inputDataPtr + bIndex * batchStride;
        auto dstBatch       = outputDataPtr + bIndex * batchStride;

        MNN_CONCURRENCY_BEGIN(tId, icDiv4) {
            const auto srcChannelPtr   = srcBatch + tId * oc4Stride * 4;
            const auto scaleChannelPtr = scaleDataPtr + tId * 4;
            auto dstChannlePtr         = dstBatch + tId * oc4Stride * 4;

#ifdef MNN_USE_NEON
            MNNInt8ScaleToFloat(dstChannlePtr, srcChannelPtr, scaleChannelPtr, oc4Stride);
#else
            for (int i = 0; i < oc4Stride; ++i) {
                const auto srcStart = srcChannelPtr + i * 4;
                auto dstStart       = dstChannlePtr + i * 4;
                for (int j = 0; j < 4; ++j) {
                    dstStart[j] = static_cast<float>(srcStart[j]) * scaleChannelPtr[j];
                }
            }
#endif
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUInt8ToFloatCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUInt8ToFloat(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUInt8ToFloatCreator, OpType_Int8ToFloat);

} // namespace MNN
