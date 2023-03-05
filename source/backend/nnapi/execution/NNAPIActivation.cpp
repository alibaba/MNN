//
//  NNAPIActivation.cpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIActivation.hpp"

namespace MNN {


NNAPIActivation::NNAPIActivation(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIActivation::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputIdxs = getTensorIdxs(inputs);
    auto opType = mOp->type();
    int activateType = -1;
    switch (opType) {
        case OpType_Softmax: {
            // NNAPI Softmax inputs: [input, beta, axis]
            activateType = ANEURALNETWORKS_SOFTMAX;
            int axis = mOp->main_as_Axis()->axis();
            float beta = 1.0;
            inputIdxs.push_back(buildScalar(beta));
            bool needAxis = false;
            auto dims = inputs[0]->shape();
            for (int i = 0; i < dims.size(); i++) {
                if (i != axis && dims[i] > 1) {
                    needAxis = true;
                    break;
                }
            }
            if (needAxis) {
                inputIdxs.push_back(buildScalar(formatAxis(axis, inputs[0])));
            }
            break;
        }
        case OpType_ReLU:
            activateType = ANEURALNETWORKS_RELU;
            break;
        case OpType_ReLU6:
            activateType = ANEURALNETWORKS_RELU6;
            break;
        case OpType_PReLU:
            activateType = ANEURALNETWORKS_PRELU;
            inputIdxs.push_back(buildConstant(mOp->main_as_PRelu()->slope()->Data(),
                                              mOp->main_as_PRelu()->slopeCount() * sizeof(float),
                                              ANEURALNETWORKS_TENSOR_FLOAT32,
                                              {static_cast<uint32_t>(mOp->main_as_PRelu()->slopeCount())}));
            break;
        case OpType_Sigmoid:
            activateType = ANEURALNETWORKS_LOGISTIC;
            break;
        case OpType_ELU:
            activateType = ANEURALNETWORKS_ELU;
            inputIdxs.push_back(buildScalar(mOp->main_as_ELU()->alpha()));
            break;
        default:
            MNN_ERROR("[NNAPI] Activation not support %s\n", MNN::EnumNameOpType(opType));
            return NOT_SUPPORT;
    }
    return buildOperation(activateType, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_Softmax)
REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_ReLU)
REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_ReLU6)
REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_PReLU)
REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_Sigmoid)
REGISTER_NNAPI_OP_CREATOR(NNAPIActivation, OpType_ELU)
} // namespace MNN
