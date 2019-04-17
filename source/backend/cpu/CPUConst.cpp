//
//  CPUConst.cpp
//  MNN
//
//  Created by MNN on 2018/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConst.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {
// get data pointer from blob
static void *_blobData(const Blob *b) {
    void *result = nullptr;
    switch (b->dataType()) {
        case DataType_DT_FLOAT:
            result = (void *)b->float32s()->Data();
            break;
        case DataType_DT_INT32:
            result = (void *)b->int32s()->Data();
            break;
        case DataType_DT_QUINT8:
            return (void *)b->uint8s()->Data();
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return result;
}

CPUConst::CPUConst(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

ErrorCode CPUConst::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());

    auto output    = outputs[0];
    auto parameter = mOp->main_as_Blob();
    memcpy(output->host<float>(), MNN::_blobData(parameter), output->size());
    return NO_ERROR;
}

ErrorCode CPUConst::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

class CPUConstCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUConst(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConstCreator, OpType_Const);

} // namespace MNN
