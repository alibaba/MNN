//
//  CPUZeroLike.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUZeroLike.hpp"
namespace MNN {
ErrorCode CPUZeroLike::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    ::memset(outputs[0]->host<float>(), 0, outputs[0]->size());
    return NO_ERROR;
}
class CPUZeroLikeCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUZeroLike(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUZeroLikeCreator, OpType_ZerosLike);
REGISTER_CPU_OP_CREATOR(CPUZeroLikeCreator, OpType_ZeroGrad);
} // namespace MNN
