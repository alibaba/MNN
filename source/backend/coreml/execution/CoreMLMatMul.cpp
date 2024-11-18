//
//  CoreMLMatMul.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLMatMul.hpp"
namespace MNN {

static void _makeMatMul() {
    
}
CoreMLMatMul::CoreMLMatMul(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLMatMul::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputName = mCoreMLBackend->getTensorName(outputs[0]);
    std::string matmulOutput = outputName;
    if (inputs.size() > 2) {
        // Has Bias
        matmulOutput = matmulOutput + "--matmul";
    }
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BATCHED_MATMUL;
    mLayer_->batchedmatmul = mCoreMLBackend->create<CoreML__Specification__BatchedMatMulLayerParams>();
    core_ml__specification__batched_mat_mul_layer_params__init(mLayer_->batchedmatmul);
    if (mOp->main_type() == OpParameter_MatMul) {
        mLayer_->batchedmatmul->transposea = mOp->main_as_MatMul()->transposeA();
        mLayer_->batchedmatmul->transposeb = mOp->main_as_MatMul()->transposeB();
    } else if (mOp->main_type() == OpParameter_BatchMatMulParam) {
        mLayer_->batchedmatmul->transposea = mOp->main_as_BatchMatMulParam()->adjX();
        mLayer_->batchedmatmul->transposeb = mOp->main_as_BatchMatMulParam()->adjY();
    }
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0]), mCoreMLBackend->getTensorName(inputs[1])}, {matmulOutput});
    mCoreMLBackend->setLayerName(mLayer_, "MatMul");
    mCoreMLBackend->addLayer(mLayer_);
    if (inputs.size() > 2) {
        // Add Bias
        auto biasLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(biasLayer);
        mCoreMLBackend->setLayerName(biasLayer, outputName + "Bias");
        mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ADD_BROADCASTABLE;
        mLayer_->addbroadcastable = mCoreMLBackend->create<CoreML__Specification__AddBroadcastableLayerParams>();
        core_ml__specification__add_broadcastable_layer_params__init(mLayer_->addbroadcastable);
        setLayerInputsAndOutputs(biasLayer, {matmulOutput, mCoreMLBackend->getTensorName(inputs[2])}, {outputName});
        mCoreMLBackend->addLayer(biasLayer);
    }
    return NO_ERROR;
}


REGISTER_COREML_OP_CREATOR(CoreMLMatMul, OpType_BatchMatMul)
REGISTER_COREML_OP_CREATOR(CoreMLMatMul, OpType_MatMul)

} // namespace MNN
