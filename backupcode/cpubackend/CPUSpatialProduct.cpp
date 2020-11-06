//
//  CPUSpatialProduct.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUSpatialProduct.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
CPUSpatialProduct::CPUSpatialProduct(Backend *b) : MNN::Execution(b) {
    // nothing to do
}

ErrorCode CPUSpatialProduct::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Assume
    // bottom[0] dim CxHxW
    // bottom[1] dim 1xHxW
    // top[0]    dim CxHxW
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];
    int w             = inputTensor->width();
    int h             = inputTensor->height();
    int channels      = UP_DIV(inputTensor->channel(), 4);
    int size          = w * h;

    auto inputT1 = inputs[1];

    // second, top[0](CxHxW) = bottom[0](CxHxW) * bottom[1](CxHxW)
    for (int q = 0; q < channels; q++) {
        const float *ptr  = inputTensor->host<float>() + q * size * 4;
        const float *ptr1 = inputT1->host<float>();
        float *outptr     = outputTensor->host<float>() + q * size * 4;

        for (int v = 0; v < size; ++v) {
#ifdef MNN_USE_NEON
            vst1q_f32(outptr + 4 * v, vld1q_f32(ptr + 4 * v) * vdupq_n_f32(ptr1[4 * v]));
#else
            for (int j = 0; j < 4; ++j) {
                outptr[4 * v + j] = ptr[4 * v + j] * ptr1[4 * v + 0];
            }
#endif
        }
    }
    return NO_ERROR;
}

class CPUSpatialProductCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUSpatialProduct(backend);
    }
};
REGISTER_CPU_OP_CREATOR(CPUSpatialProductCreator, OpType_SpatialProduct);
} // namespace MNN
