//
//  BinaryTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

using namespace tflite;

DECLARE_OP_COVERTER(BinaryTflite);

MNN::OpType BinaryTflite::opType(int quantizedModel) {
    return MNN::OpType_Extra;
}
MNN::OpParameter BinaryTflite::type(int quantizedModel) {
    return MNN::OpParameter_Extra;
}

void BinaryTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    auto extraOpParam = new MNN::ExtraT;
    extraOpParam->engine = "Tflite";
    extraOpParam->type = "BinaryActivation";
    extraOpParam->attr.resize(2);
    extraOpParam->attr[0].reset(new MNN::AttributeT);
    extraOpParam->attr[1].reset(new MNN::AttributeT);
    extraOpParam->attr[0]->key = "opType";
    extraOpParam->attr[0]->i = liteOpConverter::getOpCode(tfliteOpSet[tfliteOp->opcode_index].get());
    extraOpParam->attr[1]->key = "activationType";
    // MUL / SUB / DIV each carry their own options type; AsAddOptions() returns
    // null for them, which previously dropped the fused activation (e.g. RELU6).
    auto activationType = tflite::ActivationFunctionType_NONE;
    switch (tfliteOp->builtin_options.type) {
        case tflite::BuiltinOptions_MulOptions: {
            const auto option = tfliteOp->builtin_options.AsMulOptions();
            if (nullptr != option) {
                activationType = option->fused_activation_function;
            }
            break;
        }
        case tflite::BuiltinOptions_SubOptions: {
            const auto option = tfliteOp->builtin_options.AsSubOptions();
            if (nullptr != option) {
                activationType = option->fused_activation_function;
            }
            break;
        }
        case tflite::BuiltinOptions_DivOptions: {
            const auto option = tfliteOp->builtin_options.AsDivOptions();
            if (nullptr != option) {
                activationType = option->fused_activation_function;
            }
            break;
        }
        default:
            break;
    }
    extraOpParam->attr[1]->i = activationType;
    dstOp->main.value = extraOpParam;
}
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_POW);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MAXIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MINIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER_EQUAL);
// REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_ADD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SUB);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_DIV);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_MOD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_NOT_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SQUARED_DIFFERENCE);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MUL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LOGICAL_AND);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_DIV);
