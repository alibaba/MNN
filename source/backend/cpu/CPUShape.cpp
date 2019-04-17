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

namespace MNN {

ErrorCode CPUShape::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());

    auto& ib         = inputs[0]->buffer();
    int32_t* outData = outputs[0]->host<int32_t>();
    auto dataFormat  = inputs[0]->getDimensionType();
    if (Tensor::TENSORFLOW == dataFormat) {
        for (int i = 0; i < ib.dimensions; i++) {
            outData[i] = ib.dim[i].extent;
        }
    } else {
        MNN_ASSERT(4 == ib.dimensions); // NCHW: only for (Input/Conv->Shape)
        outData[0] = ib.dim[0].extent;
        outData[1] = ib.dim[2].extent;
        outData[2] = ib.dim[3].extent;
        outData[3] = ib.dim[1].extent;
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
