//
//  CoreMLActivation.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLActivation.hpp"

namespace MNN {


CoreMLActivation::CoreMLActivation(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLActivation::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto inputName = mCoreMLBackend->getTensorName(inputs[0]);
    auto opType = mOp->type();
    if (opType == OpType_Softmax) {
        mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SOFTMAX_ND;
        mLayer_->softmaxnd = mCoreMLBackend->create<CoreML__Specification__SoftmaxNDLayerParams>();
        core_ml__specification__softmax_ndlayer_params__init(mLayer_->softmaxnd);
        mLayer_->softmaxnd->axis = mOp->main_as_Axis()->axis();
    } else {
        mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
        mLayer_->activation = mCoreMLBackend->create<CoreML__Specification__ActivationParams>();
        core_ml__specification__activation_params__init(mLayer_->activation);
        switch (opType) {
            case OpType_ReLU:
                mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_LEAKY_RE_LU;
                mLayer_->activation->leakyrelu = mCoreMLBackend->create<CoreML__Specification__ActivationLeakyReLU>();
                core_ml__specification__activation_leaky_re_lu__init(mLayer_->activation->leakyrelu);
                mLayer_->activation->leakyrelu->alpha = mOp->main_as_Relu()->slope();
                break;
            case OpType_ELU:
                mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_ELU;
                mLayer_->activation->elu = mCoreMLBackend->create<CoreML__Specification__ActivationELU>();
                core_ml__specification__activation_elu__init(mLayer_->activation->elu);
                break;
            case OpType_PReLU:
            {
                if (mOp->main_as_PRelu()->slopeCount() == 1) {
                    mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_LEAKY_RE_LU;
                    mLayer_->activation->leakyrelu = mCoreMLBackend->create<CoreML__Specification__ActivationLeakyReLU>();
                    core_ml__specification__activation_leaky_re_lu__init(mLayer_->activation->leakyrelu);
                    mLayer_->activation->leakyrelu->alpha = mOp->main_as_PRelu()->slope()->data()[0];
                    break;
                }
                mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_PRE_LU;
                mLayer_->activation->prelu = mCoreMLBackend->create<CoreML__Specification__ActivationPReLU>();
                core_ml__specification__activation_pre_lu__init(mLayer_->activation->prelu);
                mLayer_->activation->prelu->alpha = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
                core_ml__specification__weight_params__init(mLayer_->activation->prelu->alpha);
                int slopeCount = mOp->main_as_PRelu()->slopeCount();
                mLayer_->activation->prelu->alpha->n_floatvalue = slopeCount;
                mLayer_->activation->prelu->alpha->floatvalue = mCoreMLBackend->create<float>(slopeCount);
                memcpy(mLayer_->activation->prelu->alpha->floatvalue, mOp->main_as_PRelu()->slope()->Data(), slopeCount * sizeof(float));
                break;
            }
            case OpType_Sigmoid:
                mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_SIGMOID;
                mLayer_->activation->sigmoid = mCoreMLBackend->create<CoreML__Specification__ActivationSigmoid>();
                core_ml__specification__activation_sigmoid__init(mLayer_->activation->sigmoid);
                break;
            default:
                return NOT_SUPPORT;
        }
    }
    setLayerInputsAndOutputs(mLayer_, {inputName}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_ReLU)
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_ELU)
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_PReLU)
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_Sigmoid)
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_Softmax)
} // namespace MNN
