//
//  CoreMLLayerNorm.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLLayerNorm.hpp"

namespace MNN {

CoreMLLayerNorm::CoreMLLayerNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLLayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto layerNormParam = mOp->main_as_LayerNorm();
    auto normalized_shape = layerNormParam->axis();
    auto shape = inputs[0]->shape();
    int normalized_size = 1;
    std::vector<int> normalized_shape_to_layer;
    if (shape.size() == 3 && normalized_shape->size() == 1 && normalized_shape->Get(0) == 2) { // Layernorm comes from Onnx::instance norm.
        normalized_size = shape[2];
        normalized_shape_to_layer.push_back(shape[2]);
    } else { // Layernorm comes form binary fuse.
        if (normalized_shape->Get(0) < 0) { // axis like (-3,-2,-1)
            for (int i = 0; i < normalized_shape->size(); ++i) {
                int dim = (int)shape.size() + normalized_shape->Get(i);
                normalized_shape_to_layer.push_back(shape[dim]);
                normalized_size *= shape[dim];
            }
        } else { // axis like (1,2,3)
            for (int i = 0; i < normalized_shape->size(); ++i) {
                int dim = normalized_shape->Get(i);
                normalized_shape_to_layer.push_back(shape[dim]);
                normalized_size *= shape[dim];
            }
        }
    }

    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LAYER_NORMALIZATION;
    mLayer_->layernormalization = mCoreMLBackend->create<CoreML__Specification__LayerNormalizationLayerParams>();
    core_ml__specification__layer_normalization_layer_params__init(mLayer_->layernormalization);
    mLayer_->layernormalization->n_normalizedshape = normalized_shape_to_layer.size();
    mLayer_->layernormalization->normalizedshape = mCoreMLBackend->create<int64_t>(mLayer_->layernormalization->n_normalizedshape);
    for (int i = 0; i < mLayer_->layernormalization->n_normalizedshape; ++i) {
        mLayer_->layernormalization->normalizedshape[i] = normalized_shape_to_layer[i];
    }
    mLayer_->layernormalization->eps = layerNormParam->epsilon();
    int gammaSize = normalized_size;
    std::vector<float> gamma_(gammaSize, 1.0f);
    std::vector<float> beta_(gammaSize, 0.f);
    // Default gamma and beta.
    // Build and initialize coreml layer info.
    mLayer_->layernormalization->gamma = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
    mLayer_->layernormalization->beta = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
    core_ml__specification__weight_params__init(mLayer_->layernormalization->gamma);
    core_ml__specification__weight_params__init(mLayer_->layernormalization->beta);
    mLayer_->layernormalization->gamma->n_floatvalue = gammaSize;
    mLayer_->layernormalization->gamma->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->gamma->n_floatvalue);
    memcpy(mLayer_->layernormalization->gamma->floatvalue, gamma_.data(), gammaSize * sizeof(float));
    mLayer_->layernormalization->beta->n_floatvalue = gammaSize;
    mLayer_->layernormalization->beta->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->beta->n_floatvalue);
    memcpy(mLayer_->layernormalization->beta->floatvalue, beta_.data(), gammaSize * sizeof(float));

    if (layerNormParam->gamma() && layerNormParam->beta()) {
        // Get op info.
        int size = layerNormParam->gamma()->size();
        if (layerNormParam->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in CPULayerNorm.\n");
        }
        const float* gammaData = layerNormParam->gamma()->data();
        const float* betaData = layerNormParam->beta()->data();
        
        // Copy value.
        mLayer_->layernormalization->gamma->n_floatvalue = size;
        mLayer_->layernormalization->gamma->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->gamma->n_floatvalue);
        memcpy(mLayer_->layernormalization->gamma->floatvalue, gammaData, size * sizeof(float));
        mLayer_->layernormalization->beta->n_floatvalue = size;
        mLayer_->layernormalization->beta->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->beta->n_floatvalue);
        memcpy(mLayer_->layernormalization->beta->floatvalue, betaData, size * sizeof(float));
    }

    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
    
    /*
    int channel = inputs[0]->shape()[1];
     mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BATCHNORM;
     mLayer_->batchnorm = mCoreMLBackend->create<CoreML__Specification__BatchnormLayerParams>();
     core_ml__specification__batchnorm_layer_params__init(mLayer_->batchnorm);
    mLayer_->batchnorm->channels = channel;
     mLayer_->batchnorm->instancenormalization = true;
     mLayer_->batchnorm->computemeanvar = true;
     mLayer_->batchnorm->epsilon = layerNormParam->epsilon();
    int gammaSize = channel;
     std::vector<float> gamma_(gammaSize, 1.0f);
     std::vector<float> beta_(gammaSize, 0.f);
     // Default gamma and beta.
     // Build and initialize coreml layer info.
     mLayer_->batchnorm->gamma = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
     mLayer_->batchnorm->beta = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
     core_ml__specification__weight_params__init(mLayer_->layernormalization->gamma);
     core_ml__specification__weight_params__init(mLayer_->layernormalization->beta);
     // default gamma=1,beta=0.
     mLayer_->batchnorm->gamma->n_floatvalue = gammaSize;
     mLayer_->batchnorm->gamma->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->gamma->n_floatvalue);
     memcpy(mLayer_->batchnorm->gamma->floatvalue, gamma_.data(), gammaSize * sizeof(float));
     mLayer_->batchnorm->beta->n_floatvalue = gammaSize;
     mLayer_->batchnorm->beta->floatvalue = mCoreMLBackend->create<float>(mLayer_->layernormalization->beta->n_floatvalue);
     memcpy(mLayer_->batchnorm->beta->floatvalue, beta_.data(), gammaSize * sizeof(float));
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
    */
}

REGISTER_COREML_OP_CREATOR(CoreMLLayerNorm, OpType_LayerNorm)
} // namespace MNN
