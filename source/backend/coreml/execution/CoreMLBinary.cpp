//
//  CoreMLBinary.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLBinary.hpp"

namespace MNN {


CoreMLBinary::CoreMLBinary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);
    BinaryOpOperation binaryType;
    if (mOp->type() == OpType_BinaryOp) {
        binaryType = static_cast<BinaryOpOperation>(mOp->main_as_BinaryOp()->opType());
    } else if (mOp->type() == OpType_Eltwise) {
        auto elemType = mOp->main_as_Eltwise()->type();
        switch (elemType) {
            case EltwiseType_PROD:
                binaryType = BinaryOpOperation_MUL;
                break;
            case EltwiseType_SUM:
                binaryType = BinaryOpOperation_ADD;
                break;
            case EltwiseType_SUB:
                binaryType = BinaryOpOperation_SUB;
                break;
            case EltwiseType_MAXIMUM:
                binaryType = BinaryOpOperation_MAXIMUM;
                break;
        }
    }
    bool oneInput = false;
    float constVal = 0.f;
    const Tensor* input = nullptr;
    if (TensorUtils::getDescribe(inputs[0])->usage == Tensor::InsideDescribe::CONSTANT) {
        constVal = inputs[0]->host<float>()[0];
        input = inputs[1];
    } else if (TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::CONSTANT) {
        constVal = inputs[1]->host<float>()[0];
        input = inputs[0];
    }
    switch (binaryType) {
        case BinaryOpOperation_ADD:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ADD;
            mLayer_->add = mCoreMLBackend->create<CoreML__Specification__AddLayerParams>();
            core_ml__specification__add_layer_params__init(mLayer_->add);
            if (input) {
                mLayer_->add->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_SUB:
            if (input) {
                mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
                mLayer_->activation = mCoreMLBackend->create<CoreML__Specification__ActivationParams>();
                core_ml__specification__activation_params__init(mLayer_->activation);
                mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_LINEAR;
                mLayer_->activation->linear = mCoreMLBackend->create<CoreML__Specification__ActivationLinear>();
                core_ml__specification__activation_linear__init(mLayer_->activation->linear);
                mLayer_->activation->linear->alpha = 1;
                mLayer_->activation->linear->beta = -constVal;
                oneInput = true;
            } else {
                mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SUBTRACT_BROADCASTABLE;
                mLayer_->subtractbroadcastable = mCoreMLBackend->create<CoreML__Specification__SubtractBroadcastableLayerParams>();
                core_ml__specification__subtract_broadcastable_layer_params__init(mLayer_->subtractbroadcastable);
            }
            break;
        case BinaryOpOperation_MUL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MULTIPLY;
            mLayer_->multiply = mCoreMLBackend->create<CoreML__Specification__MultiplyLayerParams>();
            core_ml__specification__multiply_layer_params__init(mLayer_->multiply);
            if (input) {
                mLayer_->multiply->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_DIV:
        case BinaryOpOperation_REALDIV:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_DIVIDE_BROADCASTABLE;
            mLayer_->dividebroadcastable = mCoreMLBackend->create<CoreML__Specification__DivideBroadcastableLayerParams>();
            core_ml__specification__divide_broadcastable_layer_params__init(mLayer_->dividebroadcastable);
            break;
        case BinaryOpOperation_POW:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_POW_BROADCASTABLE;
            mLayer_->powbroadcastable = mCoreMLBackend->create<CoreML__Specification__PowBroadcastableLayerParams>();
            core_ml__specification__pow_broadcastable_layer_params__init(mLayer_->powbroadcastable);
            break;
        case BinaryOpOperation_MINIMUM:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MIN;
            mLayer_->min = mCoreMLBackend->create<CoreML__Specification__MinLayerParams>();
            core_ml__specification__min_layer_params__init(mLayer_->min);
            break;
        case BinaryOpOperation_MAXIMUM:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MAX;
            mLayer_->max = mCoreMLBackend->create<CoreML__Specification__MaxLayerParams>();
            core_ml__specification__max_layer_params__init(mLayer_->max);
            break;
        case BinaryOpOperation_GREATER:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GREATER_THAN;
            mLayer_->greaterthan = mCoreMLBackend->create<CoreML__Specification__GreaterThanLayerParams>();
            core_ml__specification__greater_than_layer_params__init(mLayer_->greaterthan);
            if (input) {
                mLayer_->greaterthan->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_GREATER_EQUAL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GREATER_EQUAL;
            mLayer_->greaterequal = mCoreMLBackend->create<CoreML__Specification__GreaterEqualLayerParams>();
            core_ml__specification__greater_equal_layer_params__init(mLayer_->greaterequal);
            if (input) {
                mLayer_->greaterequal->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_LESS:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LESS_THAN;
            mLayer_->lessthan = mCoreMLBackend->create<CoreML__Specification__LessThanLayerParams>();
            core_ml__specification__less_than_layer_params__init(mLayer_->lessthan);
            if (input) {
                mLayer_->lessthan->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_LESS_EQUAL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LESS_EQUAL;
            mLayer_->lessequal = mCoreMLBackend->create<CoreML__Specification__LessEqualLayerParams>();
            core_ml__specification__less_equal_layer_params__init(mLayer_->lessequal);
            if (input) {
                mLayer_->lessequal->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_FLOORDIV:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_FLOOR_DIV_BROADCASTABLE;
            mLayer_->floordivbroadcastable = mCoreMLBackend->create<CoreML__Specification__FloorDivBroadcastableLayerParams>();
            core_ml__specification__floor_div_broadcastable_layer_params__init(mLayer_->floordivbroadcastable);
            break;
        case BinaryOpOperation_EQUAL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_EQUAL;
            mLayer_->equal = mCoreMLBackend->create<CoreML__Specification__EqualLayerParams>();
            core_ml__specification__equal_layer_params__init(mLayer_->equal);
            if (input) {
                mLayer_->equal->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_NOTEQUAL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_NOT_EQUAL;
            mLayer_->notequal = mCoreMLBackend->create<CoreML__Specification__NotEqualLayerParams>();
            core_ml__specification__not_equal_layer_params__init(mLayer_->notequal);
            if (input) {
                mLayer_->notequal->alpha = constVal;
                oneInput = true;
            }
            break;
        case BinaryOpOperation_MOD:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MOD_BROADCASTABLE;
            mLayer_->modbroadcastable = mCoreMLBackend->create<CoreML__Specification__ModBroadcastableLayerParams>();
            core_ml__specification__mod_broadcastable_layer_params__init(mLayer_->modbroadcastable);
            break;
        case BinaryOpOperation_ATAN2:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GREATER_THAN;
            mLayer_->atan = mCoreMLBackend->create<CoreML__Specification__AtanLayerParams>();
            core_ml__specification__atan_layer_params__init(mLayer_->atan);
            break;
        case BinaryOpOperation_LOGICALOR:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LOGICAL_OR;
            mLayer_->logicalor = mCoreMLBackend->create<CoreML__Specification__LogicalOrLayerParams>();
            core_ml__specification__logical_or_layer_params__init(mLayer_->logicalor);
            break;
        default:
            MNN_ERROR("NPU Binary not support %s\n", MNN::EnumNameBinaryOpOperation(binaryType));
            break;
    }

    std::string binartInputName;
    if(oneInput) {
        binartInputName = mCoreMLBackend->getTensorName(input);
    } else {
        binartInputName = mCoreMLBackend->getTensorName(inputs[0]);
    }
    std::string binaryOutputName = mCoreMLBackend->getTensorName(outputs[0]);
    int activationType = 0;
    if(mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    if (activationType == 1) {
        binaryOutputName = binartInputName + "-" + binaryOutputName + "-Relu";
    }

    if (oneInput) {
        setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(input)}, {binaryOutputName});
    } else {
        setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0]), mCoreMLBackend->getTensorName(inputs[1])}, {binaryOutputName});
    }
    mCoreMLBackend->addLayer(mLayer_);

    if (activationType == 1) {
        auto reluLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(reluLayer);
        mCoreMLBackend->setLayerName(reluLayer, "BinaryRelu");
        reluLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
        reluLayer->activation = mCoreMLBackend->create<CoreML__Specification__ActivationParams>();
        core_ml__specification__activation_params__init(reluLayer->activation);
        reluLayer->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_RE_LU;
        reluLayer->activation->relu = mCoreMLBackend->create<CoreML__Specification__ActivationReLU>();
        core_ml__specification__activation_re_lu__init(reluLayer->activation->relu);
        setLayerInputsAndOutputs(reluLayer, {binaryOutputName}, {mCoreMLBackend->getTensorName(outputs[0])});
        mCoreMLBackend->addLayer(reluLayer);
    }

    return NO_ERROR;
}


REGISTER_COREML_OP_CREATOR(CoreMLBinary, OpType_BinaryOp)
REGISTER_COREML_OP_CREATOR(CoreMLBinary, OpType_Eltwise)

} // namespace MNN
