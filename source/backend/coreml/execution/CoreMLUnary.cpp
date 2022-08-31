//
//  CoreMLUnary.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLUnary.hpp"

namespace MNN {


CoreMLUnary::CoreMLUnary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto inputName = mCoreMLBackend->getTensorName(inputs[0]);
    auto opType = mOp->main_as_UnaryOp()->opType();
#define SET_UNARY_PARAM \
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNARY; \
    mLayer_->unary = mCoreMLBackend->create<CoreML__Specification__UnaryFunctionLayerParams>(); \
    core_ml__specification__unary_function_layer_params__init(mLayer_->unary);
    switch (opType) {
        case UnaryOpOperation_ABS:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__ABS;
            break;
        case UnaryOpOperation_EXP:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__EXP;
            break;
        case UnaryOpOperation_SQRT:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__SQRT;
            break;
        case UnaryOpOperation_RSQRT:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__RSQRT;
            break;
        case UnaryOpOperation_LOG:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__LOG;
            break;
        case UnaryOpOperation_RECIPROCAL:
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__INVERSE;
            break;
        case UnaryOpOperation_SIN:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SIN;
            mLayer_->sin = mCoreMLBackend->create<CoreML__Specification__SinLayerParams>();
            core_ml__specification__sin_layer_params__init(mLayer_->sin);
            break;
        case UnaryOpOperation_ASIN:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ASIN;
            mLayer_->asin = mCoreMLBackend->create<CoreML__Specification__AsinLayerParams>();
            core_ml__specification__asin_layer_params__init(mLayer_->asin);
            break;
        case UnaryOpOperation_SINH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SINH;
            mLayer_->sinh = mCoreMLBackend->create<CoreML__Specification__SinhLayerParams>();
            core_ml__specification__sinh_layer_params__init(mLayer_->sinh);
            break;
        case UnaryOpOperation_ASINH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ASINH;
            mLayer_->asinh = mCoreMLBackend->create<CoreML__Specification__AsinhLayerParams>();
            core_ml__specification__asinh_layer_params__init(mLayer_->asinh);
            break;
        case UnaryOpOperation_COS:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_COS;
            mLayer_->cos = mCoreMLBackend->create<CoreML__Specification__CosLayerParams>();
            core_ml__specification__cos_layer_params__init(mLayer_->cos);
            break;
        case UnaryOpOperation_ACOS:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACOS;
            mLayer_->acos = mCoreMLBackend->create<CoreML__Specification__AcosLayerParams>();
            core_ml__specification__acos_layer_params__init(mLayer_->acos);
            break;
        case UnaryOpOperation_COSH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_COSH;
            mLayer_->cosh = mCoreMLBackend->create<CoreML__Specification__CoshLayerParams>();
            core_ml__specification__cosh_layer_params__init(mLayer_->cosh);
            break;
        case UnaryOpOperation_ACOSH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACOSH;
            mLayer_->acosh = mCoreMLBackend->create<CoreML__Specification__AcoshLayerParams>();
            core_ml__specification__acosh_layer_params__init(mLayer_->acosh);
            break;
        case UnaryOpOperation_TAN:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TAN;
            mLayer_->tan = mCoreMLBackend->create<CoreML__Specification__TanLayerParams>();
            core_ml__specification__tan_layer_params__init(mLayer_->tan);
            break;
        case UnaryOpOperation_ATAN:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ATAN;
            mLayer_->atan = mCoreMLBackend->create<CoreML__Specification__AtanLayerParams>();
            core_ml__specification__atan_layer_params__init(mLayer_->atan);
            break;
        case UnaryOpOperation_TANH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TANH;
            mLayer_->tanh = mCoreMLBackend->create<CoreML__Specification__TanhLayerParams>();
            core_ml__specification__tanh_layer_params__init(mLayer_->tanh);
            break;
        case UnaryOpOperation_ATANH:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ATANH;
            mLayer_->atanh = mCoreMLBackend->create<CoreML__Specification__AtanhLayerParams>();
            core_ml__specification__atanh_layer_params__init(mLayer_->atanh);
            break;
        case UnaryOpOperation_ERF:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ERF;
            mLayer_->erf = mCoreMLBackend->create<CoreML__Specification__ErfLayerParams>();
            core_ml__specification__erf_layer_params__init(mLayer_->erf);
            break;
        case UnaryOpOperation_CEIL:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CEIL;
            mLayer_->ceil = mCoreMLBackend->create<CoreML__Specification__CeilLayerParams>();
            core_ml__specification__ceil_layer_params__init(mLayer_->ceil);
            break;
        case UnaryOpOperation_FLOOR:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_FLOOR;
            mLayer_->floor = mCoreMLBackend->create<CoreML__Specification__FloorLayerParams>();
            core_ml__specification__floor_layer_params__init(mLayer_->floor);
            break;
        case UnaryOpOperation_ROUND:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ROUND;
            mLayer_->round = mCoreMLBackend->create<CoreML__Specification__RoundLayerParams>();
            core_ml__specification__round_layer_params__init(mLayer_->round);
            break;
        case UnaryOpOperation_SIGN:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SIGN;
            mLayer_->sign = mCoreMLBackend->create<CoreML__Specification__SignLayerParams>();
            core_ml__specification__sign_layer_params__init(mLayer_->sign);
            break;
        case UnaryOpOperation_SIGMOID:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
            mLayer_->activation = mCoreMLBackend->create<CoreML__Specification__ActivationParams>();
            core_ml__specification__activation_params__init(mLayer_->activation);
            mLayer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_SIGMOID;
            mLayer_->activation->sigmoid = mCoreMLBackend->create<CoreML__Specification__ActivationSigmoid>();
            core_ml__specification__activation_sigmoid__init(mLayer_->activation->sigmoid);
            break;
        case UnaryOpOperation_LOG1P:
            // y = log(x+1)
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__LOG;
            mLayer_->unary->shift = 1;
            break;
        case UnaryOpOperation_SQUARE:
            // y = x^2
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__POWER;
            mLayer_->unary->alpha = 2;
            break;
        case UnaryOpOperation_NEG:
            // y = (-x)^1
            SET_UNARY_PARAM
            mLayer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__POWER;
            mLayer_->unary->scale = -1;
            mLayer_->unary->alpha = 1;
            break;
        case UnaryOpOperation_HARDSWISH:
        {
            // (min(max(x + 3, 0), 6) * x) / 6
            auto addLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(addLayer);
            mCoreMLBackend->setLayerName(addLayer, "hardswish-add");
            addLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ADD;
            addLayer->add = mCoreMLBackend->create<CoreML__Specification__AddLayerParams>();
            core_ml__specification__add_layer_params__init(addLayer->add);
            addLayer->add->alpha = 3.f;
            std::string addOutput = inputName + "-add";
            setLayerInputsAndOutputs(addLayer, {inputName}, {addOutput});
            mCoreMLBackend->addLayer(addLayer);

            auto reluLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(reluLayer);
            mCoreMLBackend->setLayerName(reluLayer, "hardswish-relu");
            reluLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
            reluLayer->activation = mCoreMLBackend->create<CoreML__Specification__ActivationParams>();
            core_ml__specification__activation_params__init(reluLayer->activation);
            reluLayer->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_RE_LU;
            reluLayer->activation->relu = mCoreMLBackend->create<CoreML__Specification__ActivationReLU>();
            core_ml__specification__activation_re_lu__init(reluLayer->activation->relu);
            std::string reluOutput = addOutput + "-relu";
            setLayerInputsAndOutputs(reluLayer, {addOutput}, {reluOutput});
            mCoreMLBackend->addLayer(reluLayer);

            auto thresholdLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(thresholdLayer);
            mCoreMLBackend->setLayerName(thresholdLayer, "hardswish-threshold");
            thresholdLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNARY;
            thresholdLayer->unary = mCoreMLBackend->create<CoreML__Specification__UnaryFunctionLayerParams>();
            core_ml__specification__unary_function_layer_params__init(thresholdLayer->unary);
            thresholdLayer->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__THRESHOLD;
            thresholdLayer->unary->alpha = -6;
            thresholdLayer->unary->scale = -1;
            std::string thresholdOutput = reluOutput + "-threshold";
            setLayerInputsAndOutputs(thresholdLayer, {reluOutput}, {thresholdOutput});
            mCoreMLBackend->addLayer(thresholdLayer);

            auto negmulLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
            core_ml__specification__neural_network_layer__init(negmulLayer);
            mCoreMLBackend->setLayerName(negmulLayer, "hardswish-negmul");
            negmulLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MULTIPLY;
            negmulLayer->multiply = mCoreMLBackend->create<CoreML__Specification__MultiplyLayerParams>();
            core_ml__specification__multiply_layer_params__init(negmulLayer->multiply);
            negmulLayer->multiply->alpha = -1.f / 6;
            std::string negmulOutput = thresholdOutput + "-negmul";
            setLayerInputsAndOutputs(negmulLayer, {thresholdOutput}, {negmulOutput});
            mCoreMLBackend->addLayer(negmulLayer);

            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MULTIPLY;
            mLayer_->multiply = mCoreMLBackend->create<CoreML__Specification__MultiplyLayerParams>();
            core_ml__specification__multiply_layer_params__init(mLayer_->multiply);
            setLayerInputsAndOutputs(mLayer_, {negmulOutput, inputName}, {mCoreMLBackend->getTensorName(outputs[0])});
            mCoreMLBackend->addLayer(mLayer_);
            return NO_ERROR;
        }
        case UnaryOpOperation_GELU:
        case UnaryOpOperation_GELU_STANDARD:
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GELU;
            mLayer_->gelu = mCoreMLBackend->create<CoreML__Specification__GeluLayerParams>();
            core_ml__specification__gelu_layer_params__init(mLayer_->gelu);
            break;
        /*
        // Don't support Op
        case UnaryOpOperation_EXPM1:
        case UnaryOpOperation_ERFC:
        case UnaryOpOperation_BNLL:
        case UnaryOpOperation_ERFINV:
        */
        default:
            MNN_ERROR("NPU Unary not support %s\n", MNN::EnumNameUnaryOpOperation(opType));
            break;
    }
    setLayerInputsAndOutputs(mLayer_, {inputName}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLUnary, OpType_UnaryOp)
} // namespace MNN
