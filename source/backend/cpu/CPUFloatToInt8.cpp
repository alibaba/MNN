//
//  CPUFloatToInt8.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUFloatToInt8.hpp"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Int8FunctionsOpt.h"
#include "Macro.h"

namespace MNN {

CPUFloatToInt8::CPUFloatToInt8(Backend* backend, const MNN::Op* param) : Execution(backend) {
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
CPUFloatToInt8::~CPUFloatToInt8() {
    backend()->onReleaseBuffer(mScales.get(), Backend::STATIC);
}

ErrorCode CPUFloatToInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode CPUFloatToInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const auto inputDataPtr = input->host<float>();
    auto outputDataPtr      = output->host<int8_t>();
    const auto scaleDataPtr = mScales->host<float>();
    const int channels      = input->channel();
    const int icDiv4        = UP_DIV(channels, 4);
    const int batch         = input->batch();
    const int batchStride   = input->stride(0);
    const int width         = input->width();
    const int height        = input->height();
    const int oc4Stride     = width * height;
    auto numberThread       = std::min(icDiv4, ((CPUBackend*)backend())->threadNumber());

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcBatch = inputDataPtr + bIndex * batchStride;
        auto dstBatch       = outputDataPtr + bIndex * batchStride;

        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int z = (int)tId; z < icDiv4; z += numberThread) {
                const auto srcChannelPtr   = srcBatch + z * oc4Stride * 4;
                const auto scaleChannelPtr = scaleDataPtr + z * 4;
                auto dstChannlePtr         = dstBatch + z * oc4Stride * 4;
                MNNFloat2Int8(srcChannelPtr, dstChannlePtr, oc4Stride, scaleChannelPtr, -127, 127);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUFloatToInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUFloatToInt8(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUFloatToInt8Creator, OpType_FloatToInt8);

} // namespace MNN
