//
//  CoreMLRelu6.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLRelu6.hpp"

namespace MNN {

CoreMLRelu6::CoreMLRelu6(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    if (nullptr != op->main()) {
        auto p = op->main_as_Relu6();
        mMinValue = p->minValue();
        mMaxValue = p->maxValue();
    }
    initLayer();
}

ErrorCode CoreMLRelu6::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CLIP;
    mLayer_->clip = mCoreMLBackend->create<_CoreML__Specification__ClipLayerParams>();
    core_ml__specification__clip_layer_params__init(mLayer_->clip);
    mLayer_->clip->maxval = mMaxValue;
    mLayer_->clip->minval = mMinValue;

    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLRelu6, OpType_ReLU6)
} // namespace MNN
