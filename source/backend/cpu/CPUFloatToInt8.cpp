//
//  CPUFloatToInt8.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUFloatToInt8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUFloatToInt8::CPUFloatToInt8(Backend* backend, const MNN::Op* param) : Execution(backend) {
    auto scale         = param->main_as_QuantizedFloatParam();
    const int scaleLen = scale->tensorScale()->size();
    mClipBits = scale->nbits();
    mScales.reset(Tensor::createDevice<float>({ALIGN_UP4(scaleLen)}));
    mValid = backend->onAcquireBuffer(mScales.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    if (1 == scaleLen) {
        mSingle = true;
        for (int i = 0; i < 4; ++i) {
            mScales->host<float>()[i] = scale->tensorScale()->data()[0];
        }
    } else {
        memset(mScales->host<float>(), 0, ALIGN_UP4(scaleLen) * sizeof(float));
        memcpy(mScales->host<float>(), scale->tensorScale()->data(), scaleLen * sizeof(float));
    }

    mZeroPoint = scale->zeroPoint();
    mClampMin = scale->clampMin();
    mClampMax = scale->clampMax();
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
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(inputs[0])->dimensionFormat);

    const auto inputDataPtr = input->host<float>();
    auto outputDataPtr      = output->host<int8_t>();
    const auto scaleDataPtr = mScales->host<float>();
    const int channels      = input->channel();
    int icDiv4        = UP_DIV(channels, 4);
    const int batch         = input->batch();
    const int batchStride   = input->stride(0);
    int oc4Stride           = 1;
    for (int i = 2; i < input->dimensions(); ++i) {
        oc4Stride *= input->length(i);
    }
    if (mSingle) {
        oc4Stride = icDiv4 * oc4Stride;
        icDiv4 = 1;
    }
    int total = batch * icDiv4;
    auto numberThread       = std::min(icDiv4, ((CPUBackend*)backend())->threadNumber());

    MNN_CONCURRENCY_BEGIN(tId, total) {
        int bIndex = tId / icDiv4;
        int z = tId % icDiv4;
        const auto srcChannelPtr   = inputDataPtr + tId * oc4Stride * 4;
        const auto scaleChannelPtr = scaleDataPtr + z * 4;
        auto dstChannlePtr         = outputDataPtr + tId * oc4Stride * 4;
        MNNFloat2Int8(srcChannelPtr, dstChannlePtr, oc4Stride, scaleChannelPtr, mClampMin, mClampMax, mZeroPoint);
    }
    MNN_CONCURRENCY_END();
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
