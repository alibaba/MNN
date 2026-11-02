//
//  CoreMLActivation.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLActivation.hpp"

namespace MNN {

CoreMLActivation::CoreMLActivation(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {}

ErrorCode CoreMLActivation::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto inputName = mCoreMLBackend->getTensorName(inputs[0]);
    auto opType = mOp->type();
    switch (opType) {
        case OpType_Softmax:
            operation("softmax").input("x", inputs[0]).scalar("axis", mOp->main_as_Axis()->axis()).output(outputs);
            break;
        case OpType_ReLU: {
            float slope = mOp->main_as_Relu()->slope();
            auto builder = operation(slope == 0.0f ? "relu" : "leaky_relu");
            builder.input("x", inputs[0]);
            if (slope != 0.0f) {
                builder.scalar("alpha", slope);
            }
            builder.output(outputs);
            break;
        }
        case OpType_ReLU6: {
            auto relu6 = mOp->main_as_Relu6();
            float min = relu6 ? relu6->minValue() : 0.0f;
            float max = relu6 ? relu6->maxValue() : 6.0f;
            operation("clip").input("x", inputs[0]).scalar("alpha", min).scalar("beta", max).output(outputs);
            break;
        }
        case OpType_ELU:
            operation("elu").input("x", inputs[0]).scalar("alpha", mOp->main_as_ELU()->alpha()).output(outputs);
            break;
        case OpType_PReLU: {
            auto slope = mOp->main_as_PRelu()->slope();
            std::vector<int> shape(inputs[0]->dimensions(), 1);
            if (shape.empty() || !slope || slope->size() == 0) {
                return NOT_SUPPORT;
            }
            const int axis = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC
                                 ? shape.size() - 1
                                 : std::min<int>(1, shape.size() - 1);
            shape[axis] = slope->size();
            auto alpha = buildBlob(slope->data(), slope->size() * sizeof(float),
                                   CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, shape, "slope");
            auto neg = operation("minimum")
                           .input("x", inputName)
                           .scalar("y", 0.0f, "zero")
                           .output(inputs[0]->shape(), "negative");
            auto scaled = operation("mul").input("x", neg).input("y", alpha).output(inputs[0]->shape(), "scaled");
            auto pos = operation("relu").input("x", inputName).output(inputs[0]->shape(), "positive");
            operation("add").input("x", pos).input("y", scaled).output(outputs);
            break;
        }
        case OpType_Sigmoid:
            operation("sigmoid").input("x", inputs[0]).output(outputs);
            break;
        default:
            return NOT_SUPPORT;
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_ReLU);
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_ReLU6);
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_ELU);
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_PReLU);
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_Sigmoid);
REGISTER_COREML_OP_CREATOR(CoreMLActivation, OpType_Softmax);
} // namespace MNN
