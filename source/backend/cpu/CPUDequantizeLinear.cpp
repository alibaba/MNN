//
//  CPUDequantizeLinear.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUDequantizeLinear.hpp"
#include "core/TensorUtils.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {

CPUDequantizeLinear::CPUDequantizeLinear(Backend *b, float* scale, int8_t* zeroPoints, int size, int axis, int inputBits) : MNN::Execution(b){
    mSize = size;
    mAxis = axis;
    mInputBits = inputBits;
}
ErrorCode CPUDequantizeLinear::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mInputBits == 8) {
        mFunc = dequantizeFunc<int8_t>;
    } else if (mInputBits == 16) {
        mFunc = dequantizeFunc<int16_t>;
    } else {
        mFunc = dequantizeFunc<int32_t>;
    }
    float *scale = inputs[1]->host<float>();
    int8_t *zero = nullptr;
    if (inputs.size() > 2) {
        zero = inputs[2]->host<int8_t>();;
    }
    if (mSize == 1) {
        mQuantScales.resize(4, *scale);
        if (nullptr != zero) {
            mQuantZeroPoints.resize(4, *zero);
        } else {
            mQuantZeroPoints.resize(4, 0);
        }
    } else {
        mQuantScales.resize(mSize);
        ::memcpy(mQuantScales.data(), scale, sizeof(float) * mSize);
        if (nullptr != zero) {
            mQuantZeroPoints.resize(mSize);
            ::memcpy(mQuantZeroPoints.data(), zero, mSize);
        } else {
            mQuantZeroPoints.resize(mSize);
        }
    }
    return NO_ERROR;
}
ErrorCode CPUDequantizeLinear::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)  {
    auto input = inputs[0];
    int N = input->length(0);
    ssize_t size = N;
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto dst = outputs[0]->host<float>();
    auto src = input->host<int8_t>();
    mFunc(dst, src, input->dimensions(), input->size(), mSize, UNIT, mQuantScales.data(), mQuantZeroPoints.data(), core);
    return NO_ERROR;
}

class CPUDequantizeLinearCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto dataType = inputs[0]->getType();
        if (dataType.bits != 8 && dataType.bits != 16 && dataType.bits != 32) {
            MNN_ERROR("Input of Dequantize must be int8/uint8/fp16/int32\n");
            return nullptr;
        }
        int inputBits = dataType.bits;
        int size = op->main_as_DequantizeLinear()->scaleSize();
        int axis = op->main_as_DequantizeLinear()->scaleAxis();
        if (inputs.size() > 2) {
            return new CPUDequantizeLinear(backend, inputs[1]->host<float>(), inputs[2]->host<int8_t>(), size, axis, inputBits);
        }
        return new CPUDequantizeLinear(backend, inputs[1]->host<float>(), nullptr, size, axis, inputBits);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDequantizeLinearCreator, OpType_DequantizeLinear);

} // namespace MNN
