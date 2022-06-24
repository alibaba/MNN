//
//  CoreMLScale.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLScale.hpp"

namespace MNN {

CoreMLScale::CoreMLScale(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto scaleParam = mOp->main_as_Scale();
    auto mnnScale = scaleParam->scaleData();
    auto mnnBias = scaleParam->biasData();
    auto channel = scaleParam->channels();
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SCALE;
    mLayer_->scale = mCoreMLBackend->create<CoreML__Specification__ScaleLayerParams>();
    core_ml__specification__scale_layer_params__init(mLayer_->scale);
    mLayer_->scale->n_shapescale = 1;
    mLayer_->scale->shapescale = mCoreMLBackend->create<uint64_t>(mLayer_->scale->n_shapescale);
    *mLayer_->scale->shapescale = channel;
    mLayer_->scale->scale = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
    core_ml__specification__weight_params__init(mLayer_->scale->scale);
    mLayer_->scale->scale->n_floatvalue = mnnScale->size();
    mLayer_->scale->scale->floatvalue = mCoreMLBackend->create<float>(mLayer_->scale->scale->n_floatvalue);
    memcpy(mLayer_->scale->scale->floatvalue, mnnScale->data(), mnnScale->size() * sizeof(float));
    if (mnnBias->size() > 0) {
        mLayer_->scale->hasbias = true;
        mLayer_->scale->n_shapebias = 1;
        mLayer_->scale->shapebias = mCoreMLBackend->create<uint64_t>(mLayer_->scale->n_shapebias);
        *mLayer_->scale->shapebias = channel;
        mLayer_->scale->bias = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
        core_ml__specification__weight_params__init(mLayer_->scale->bias);
        mLayer_->scale->bias->n_floatvalue = mnnBias->size();
        mLayer_->scale->bias->floatvalue = mCoreMLBackend->create<float>(mLayer_->scale->scale->n_floatvalue);
        memcpy(mLayer_->scale->bias->floatvalue, mnnBias->data(), mnnBias->size() * sizeof(float));
    }
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLScale, OpType_Scale)
} // namespace MNN
