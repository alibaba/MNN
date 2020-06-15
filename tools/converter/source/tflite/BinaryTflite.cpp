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

MNN::OpType BinaryTflite::opType(bool quantizedModel) {
    return MNN::OpType_BinaryOp;
}
MNN::OpParameter BinaryTflite::type(bool quantizedModel) {
    return MNN::OpParameter_BinaryOp;
}

void BinaryTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    auto param = new MNN::BinaryOpT;
    switch (tfliteOpSet[tfliteOp->opcode_index]->builtin_code) {
        case tflite::BuiltinOperator_POW: {
            param->opType = MNN::BinaryOpOperation_POW;
            break;
        }
        case tflite::BuiltinOperator_MAXIMUM: {
            param->opType = MNN::BinaryOpOperation_MAXIMUM;
            break;
        }
        case tflite::BuiltinOperator_MINIMUM: {
            param->opType = MNN::BinaryOpOperation_MINIMUM;
            break;
        }
        case tflite::BuiltinOperator_LESS: {
            param->opType = MNN::BinaryOpOperation_LESS;
            break;
        }
        case tflite::BuiltinOperator_GREATER_EQUAL: {
            param->opType = MNN::BinaryOpOperation_GREATER_EQUAL;
            break;
        }
        case tflite::BuiltinOperator_ADD: {
            param->opType = MNN::BinaryOpOperation_ADD;
            break;
        }
        case tflite::BuiltinOperator_SUB: {
            param->opType = MNN::BinaryOpOperation_SUB;
            break;
        }
        case tflite::BuiltinOperator_FLOOR_DIV: {
            param->opType = MNN::BinaryOpOperation_FLOORDIV;
            break;
        }
        case tflite::BuiltinOperator_FLOOR_MOD: {
            param->opType = MNN::BinaryOpOperation_FLOORMOD;
            break;
        }
        case tflite::BuiltinOperator_LESS_EQUAL: {
            param->opType = MNN::BinaryOpOperation_LESS_EQUAL;
            break;
        }
        case tflite::BuiltinOperator_GREATER: {
            param->opType = MNN::BinaryOpOperation_GREATER;
            break;
        }
        case tflite::BuiltinOperator_EQUAL: {
            param->opType = MNN::BinaryOpOperation_EQUAL;
            break;
        }
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE: {
            param->opType = MNN::BinaryOpOperation_SquaredDifference;
            break;
        }
        case BuiltinOperator_MUL:
        case BuiltinOperator_LOGICAL_AND: {
            param->opType = MNN::BinaryOpOperation_MUL;
            break;
        }
        default: {
            LOG(ERROR) << "MNN Converter Not "
                          "Supported!!! BinaryOp: "
                       << tfliteOpSet[tfliteOp->opcode_index]->custom_code;
        }
    }
    dstOp->main.value = param;
}
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_POW);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MAXIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MINIMUM);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER_EQUAL);
//REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_ADD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SUB);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_DIV);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_FLOOR_MOD);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LESS_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_GREATER);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_EQUAL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_SQUARED_DIFFERENCE);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_MUL);
REGISTER_CONVERTER(BinaryTflite, BuiltinOperator_LOGICAL_AND);
