//
//  CoreMLArgMax.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLArgMax.hpp"

namespace MNN {

CoreMLArgMax::CoreMLArgMax(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto argmaxParam = mOp->main_as_ArgMax();
    if (argmaxParam->softmaxThreshold() != 0 || argmaxParam->topK() > 1 || argmaxParam->outMaxVal() != 0) {
        MNN_ERROR("[CoreML] ArgMax Don't support softmaxThreshold, topK, outMaxVal.");
    }

    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ARG_MAX;
    mLayer_->argmax = mCoreMLBackend->create<CoreML__Specification__ArgMaxLayerParams>();
    core_ml__specification__arg_max_layer_params__init(mLayer_->argmax);
    mLayer_->argmax->axis = argmaxParam->axis();
    mLayer_->argmax->removedim = true;
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLArgMax, OpType_ArgMax)
} // namespace MNN
