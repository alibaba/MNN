//
//  CoreMLCommonExecution.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLCommonExecution.hpp"
namespace MNN {

CoreMLCommonExecution::CoreMLCommonExecution(Backend *backend, const Op *op) : Execution(backend), mOp(op) {
    mCoreMLBackend = (CoreMLBackend*)backend;
}

ErrorCode CoreMLCommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CoreMLCommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

void CoreMLCommonExecution::initLayer() {
    if (mLayer_ == nullptr) {
        mLayer_ = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(mLayer_);
    }
    mCoreMLBackend->setLayerName(mLayer_, mOp->name() ? mOp->name()->str() : "DebugName");
}
void CoreMLCommonExecution::setLayerInputsAndOutputs(CoreML__Specification__NeuralNetworkLayer* layer, std::vector<std::string>&& inputs, std::vector<std::string>&& outputs) {
    mCoreMLBackend->setLayerInputs(layer, std::move(inputs));
    mCoreMLBackend->setLayerOutputs(layer, std::move(outputs));
}
}; // namespace MNN
