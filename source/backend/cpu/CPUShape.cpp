//
//  CPUShape.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUShape.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
namespace MNN {

ErrorCode CPUShape::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());

    auto& ib         = inputs[0]->buffer();
    int32_t* outData = outputs[0]->host<int32_t>();
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        outData[0] = ib.dim[0].extent;
        outData[1] = ib.dim[2].extent;
        outData[2] = ib.dim[3].extent;
        outData[3] = ib.dim[1].extent;
    } else {
        for (int i = 0; i < ib.dimensions; i++) {
            outData[i] = ib.dim[i].extent;
        }
    }
    return NO_ERROR;
}

class CPUShapeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUShape(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUShapeCreator, OpType_Shape);
} // namespace MNN
