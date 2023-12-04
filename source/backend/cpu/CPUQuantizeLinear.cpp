//
//  CPUQuantizeLinear.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUQuantizeLinear.hpp"
#include "compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUQuantizeLinear::CPUQuantizeLinear(Backend *b, int size, int axis) : MNN::Execution(b){
    mSize = size;
    mAxis = axis;
}

ErrorCode CPUQuantizeLinear::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int size = mSize;
    float* scale = inputs[1]->host<float>();
    int8_t* zero = nullptr;
    if (inputs.size() > 2) {
        zero = inputs[2]->host<int8_t>();
    }
    if (mSize == 1) {
        float s = scale[0] == 0?0: 1/ scale[0];
        mQuantScales.resize(4, s);
        if (nullptr != zero) {
            int8_t z = *zero;
            mQuantZeroPoints.resize(4, z);
        } else {
            mQuantZeroPoints.resize(4);
        }
    } else { // TODO scale: (1,D)
        
    }
    return NO_ERROR;
}
ErrorCode CPUQuantizeLinear::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)  {
    auto input = inputs[0];
    int N = input->length(0), C = input->length(1), H = input->length(2), W = input->length(3);
    ssize_t size = N * C * H * W;
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int maxValue = 127;
    int minValue = -128;
#ifdef MNN_USE_SSE
    auto dst = outputs[0]->host<uint8_t>();
    int offset = 128;
#else
    auto dst = outputs[0]->host<int8_t>();
    int offset = 0;
#endif
    if (mSize == 1) {
        auto src = input->host<float>();
        int sizeDiv = (int)size / UNIT;
        core->MNNFloat2Int8(src, (int8_t*)dst, size / UNIT, mQuantScales.data(), -128, 127, mQuantZeroPoints[0]);
        for (int i = sizeDiv * UNIT; i < size; ++i) {
            int v = (int)roundf(src[i] * mQuantScales[0]) + mQuantZeroPoints[0] + offset;
            v = std::max(minValue + offset, std::min(maxValue + offset, v));
            dst[i] = v;
        }
    } else {
        
    }
        return NO_ERROR;
}

class CPUQuantizeLinearCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int size = op->main_as_QuantizeLinear()->scaleSize();
        int axis = op->main_as_QuantizeLinear()->scaleAxis();
        return new CPUQuantizeLinear(backend, size, axis);
    }
};

REGISTER_CPU_OP_CREATOR(CPUQuantizeLinearCreator, OpType_QuantizeLinear);

} // namespace MNN
