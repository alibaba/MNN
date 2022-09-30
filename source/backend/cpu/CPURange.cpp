//
//  CPURange.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPURange.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

template <typename T>
CPURange<T>::CPURange(Backend* backend) : Execution(backend) {
    // nothing to do
}

template <typename T>
ErrorCode CPURange<T>::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const T start = inputs[0]->host<T>()[0];
    const T delta = inputs[2]->host<T>()[0];
    int32_t outputSize = outputs[0]->buffer().dim[0].extent;
    auto flat = outputs[0]->host<T>();
    T val     = start;
    for (int32_t i = 0; i < outputSize; ++i) {
        flat[i] = T(val);
        val += delta;
    }
    return NO_ERROR;
}

class CPURangeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto code = inputs[0]->getType().code;
        switch (code) {
            case halide_type_int:
                return new MNN::CPURange<int32_t>(backend);
            case halide_type_float:
                return new MNN::CPURange<float>(backend);
            default:
                MNN_ASSERT(false); // unsupported type
                return nullptr;
        }
    }
};

REGISTER_CPU_OP_CREATOR(CPURangeCreator, OpType_Range);
} // namespace MNN
