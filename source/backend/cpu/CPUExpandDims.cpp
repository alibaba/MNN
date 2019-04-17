//
//  CPUExpandDims.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUExpandDims.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUExpandDims::CPUExpandDims(Backend *b) : MNN::Execution(b) {
    // nothing to do
}

ErrorCode CPUExpandDims::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());

    return NO_ERROR;
}

ErrorCode CPUExpandDims::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    /// !!!! notice !!!! data shouldn't be reordered!
    ::memcpy(output->buffer().host, input->buffer().host, input->size());
    return NO_ERROR;
}

class CPUExpandDimsCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUExpandDims(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUExpandDimsCreator, OpType_ExpandDims);

} // namespace MNN
