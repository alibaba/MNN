//
//  CPUInt8ToFloat.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInt8ToFloat.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "compute/Int8FunctionsOpt.h"
#include "compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUInt8ToFloat::CPUInt8ToFloat(Backend* backend, const MNN::Op* param) : Execution(backend) {
    auto scale         = param->main_as_QuantizedFloatParam();
    const int scaleLen = scale->tensorScale()->size();
    auto pack = static_cast<CPUBackend*>(backend)->functions()->pack;
    mScales.reset(Tensor::createDevice<float>({UP_DIV(scaleLen, pack) * pack}));
    mValid = backend->onAcquireBuffer(mScales.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    if (1 == scaleLen) {
        mSingle = true;
        for (int i = 0; i < pack; ++i) {
            mScales->host<float>()[i] = scale->tensorScale()->data()[0];
        }
    } else {
        memset(mScales->host<float>(), 0, UP_DIV(scaleLen, pack) * pack * sizeof(float));
        memcpy(mScales->host<float>(), scale->tensorScale()->data(), scaleLen * sizeof(float));
    }
    mZeroPoint = scale->zeroPoint();
}
CPUInt8ToFloat::~CPUInt8ToFloat() {
    backend()->onReleaseBuffer(mScales.get(), Backend::STATIC);
}
ErrorCode CPUInt8ToFloat::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto pack = static_cast<CPUBackend*>(backend())->functions()->pack;
    auto int8F = static_cast<CPUBackend*>(backend())->int8Functions();

    const auto inputDataPtr = input->host<int8_t>();
    auto outputDataPtr      = output->host<float>();
    const auto scaleDataPtr = mScales->host<float>();
    const int channels      = input->channel();
    int icDiv4        = UP_DIV(channels, pack);
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

    MNN_CONCURRENCY_BEGIN(tId, total) {
        int bIndex = tId / icDiv4;
        int z = tId % icDiv4;
        const auto srcChannelPtr   = inputDataPtr + tId * oc4Stride * pack;
        const auto scaleChannelPtr = scaleDataPtr + z * pack;
        auto dstChannlePtr         = outputDataPtr + tId * oc4Stride * pack;
        int8F->MNNInt8ScaleToFloat(dstChannlePtr, srcChannelPtr, scaleChannelPtr, oc4Stride, mZeroPoint);
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

class CPUInt8ToFloatCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (nullptr == op->main_as_QuantizedFloatParam()) {
            return new CastWrapExecution(backend, DataType_DT_FLOAT);
        }
        return new CPUInt8ToFloat(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUInt8ToFloatCreator, OpType_Int8ToFloat);

} // namespace MNN
