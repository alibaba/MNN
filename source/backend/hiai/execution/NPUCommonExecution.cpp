//
//  NPUCommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUCommonExecution.hpp"
namespace MNN {

NPUCommonExecution::NPUCommonExecution(Backend *backend, const Op *op) : Execution(backend), mOp(op) {
    mNpuBackend = (NPUBackend *)backend;
}

ErrorCode NPUCommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode NPUCommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

}; // namespace MNN
