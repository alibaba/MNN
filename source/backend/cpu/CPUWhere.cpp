//
//  CPUWhere.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUWhere.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

template <typename T>
std::vector<int32_t> _collect(Tensor* t) {
    const T* ptr = t->host<T>();
    std::vector<int32_t> collect;
    for (int i = 0; i < t->elementSize(); i++) {
        if (ptr[i] > 0) {
            collect.push_back(i);
        }
    }
    return collect;
}

ErrorCode CPUWhere::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib           = inputs[0]->buffer();
    auto outputData    = outputs[0]->host<int32_t>();

    std::vector<int32_t> collect;
    if (ib.type == halide_type_of<float>()) {
        collect = _collect<float>(inputs[0]);
    } else if (ib.type == halide_type_of<int32_t>()) {
        collect = _collect<int32_t>(inputs[0]);
    } else if (ib.type == halide_type_of<uint8_t>()) {
        collect = _collect<uint8_t>(inputs[0]);
    }

    //MNN_ASSERT(outputs[0]->batch() == trueVec.size());
    for (int i = 0; i < collect.size(); i++) {
        int index = collect[i];
        for (int j = 0; j < ib.dimensions; j++) {
            int result    = ib.dim[j].stride == 0 ? index : index / ib.dim[j].stride;
            index         = index - result * ib.dim[j].stride;
            outputData[i * ib.dimensions + j] = result;
        }
    }
    return NO_ERROR;
}

class CPUWhereCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUWhere(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUWhereCreator, OpType_Where);
} // namespace MNN
