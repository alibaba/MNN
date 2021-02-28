//
//  TRTActivation.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTActivation.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {
    
TRTActivation::TRTActivation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTActivation::onEncode(const std::vector<ITensor *> &xOp) {
    IActivationLayer *activationLayer{nullptr};
    if (mOp->type() == OpType_ReLU) {
        float slope = 0.0f;
        if (nullptr != mOp->main_as_Relu()) {
            slope = mOp->main_as_Relu()->slope();
        }
        if (slope == 0.0f) {
            activationLayer = mTrtBackend->getNetwork()->addActivation(*(xOp[0]), nvinfer1::ActivationType::kRELU);
        } else {
            // Use Prelu plugin
            auto plu         = createPluginWithOutput(mOutputs);
            auto preluPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
            nvinfer1::IPluginLayer *plugin =
                mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)preluPlugin));
            if (plugin == nullptr) {
                printf("plugin == nullptr !!!");
            }
            mTrtBackend->pushReleaseLayer(preluPlugin);
            return {plugin->getOutput(0)};
        }
    } else if (mOp->type() == OpType_Sigmoid) {
        activationLayer = mTrtBackend->getNetwork()->addActivation(*(xOp[0]), nvinfer1::ActivationType::kSIGMOID);
    } else if (mOp->type() == OpType_ReLU6) {
        activationLayer = mTrtBackend->getNetwork()->addActivation(*(xOp[0]), ActivationType::kCLIP);
        activationLayer->setAlpha(0.);
        activationLayer->setBeta(6.);
    } else {
        MNN_PRINT("activation not support this type : %d \n", mOp->type());
    }

    // activationLayer->setName(mOp->name()->str().c_str());
    return {activationLayer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTActivation>> __relu_op(OpType_ReLU);
TRTCreatorRegister<TypedCreator<TRTActivation>> __sigmoid_op(OpType_Sigmoid);
TRTCreatorRegister<TypedCreator<TRTActivation>> __relu6_op(OpType_ReLU6);

} // namespace MNN
