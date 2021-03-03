//
//  BinaryActivation.cpp
//  MNNConverter
//
//  Created by MNN on 2021/02/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "schema_generated.h"
#include "TFliteExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

/*See BinaryTflite.cpp for detail attribute*/
class BinaryActivationTransform : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        const auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto extra = op->main_as_Extra();
        auto opType = static_cast<tflite::BuiltinOperator>(extra->attr()->Get(0)->i());
        auto activationType = static_cast<tflite::ActivationFunctionType>(extra->attr()->Get(1)->i());
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 2);
        auto input0    = inputs[0];
        auto input1    = inputs[1];
        VARP newOutput;
        switch (opType) {
            case tflite::BuiltinOperator_POW: {
                newOutput = _Pow(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_MAXIMUM: {
                newOutput = _Maximum(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_MINIMUM: {
                newOutput = _Minimum(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_LESS: {
                newOutput = _Less(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_GREATER_EQUAL: {
                newOutput = _GreaterEqual(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_ADD: {
                newOutput = _Add(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_SUB: {
                newOutput = _Subtract(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_FLOOR_DIV: {
                newOutput = _FloorDiv(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_DIV: {
                newOutput = _Divide(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_FLOOR_MOD: {
                newOutput = _FloorMod(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_LESS_EQUAL: {
                newOutput = _LessEqual(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_GREATER: {
                newOutput = _GreaterEqual(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_EQUAL: {
                newOutput = _Equal(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_NOT_EQUAL:{
                newOutput = _NotEqual(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_SQUARED_DIFFERENCE: {
                newOutput = _SquaredDifference(input0, input1);
                break;
            }
            case tflite::BuiltinOperator_MUL:
            case tflite::BuiltinOperator_LOGICAL_AND: {
                newOutput = _Multiply(input0, input1);
                break;
            }
            default: {
                LOG(ERROR) << "MNN Converter Not "
                              "Supported!!! BinaryOp: "
                           << tflite::EnumNameBuiltinOperator(opType);
            }
        }
        switch (activationType) {
            case tflite::ActivationFunctionType_RELU:
            case tflite::ActivationFunctionType_RELU_N1_TO_1:
                newOutput = _Relu(newOutput);
                break;
            case tflite::ActivationFunctionType_RELU6:
                newOutput = _Relu6(newOutput);
                break;
            case tflite::ActivationFunctionType_TANH:
                newOutput = _Tanh(newOutput);
                break;
            case tflite::ActivationFunctionType_SIGN_BIT:
                newOutput = _Sign(newOutput);
                break;
            default:
                break;
        }
        newOutput->setName(expr->name());
        return newOutput->expr().first;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("BinaryActivation", std::shared_ptr<TFliteExtraManager::Transform>(new BinaryActivationTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
