//
//  UnaryTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(UnaryTflite);
MNN::OpType UnaryTflite::opType(int quantizedModel) {
    return MNN::OpType_UnaryOp;
}
MNN::OpParameter UnaryTflite::type(int quantizedModel) {
    return MNN::OpParameter_UnaryOp;
}

static MNN::UnaryOpOperation _convert(tflite::BuiltinOperator op) {
#define MNNCONVERT(x, y) if (op == tflite::BuiltinOperator_##x) return MNN::UnaryOpOperation_##y;
    MNNCONVERT(ABS, ABS);
    MNNCONVERT(COS, COS);
    MNNCONVERT(CEIL, CEIL);
    MNNCONVERT(EXP, EXP);
    MNNCONVERT(FLOOR, FLOOR);
    MNNCONVERT(HARD_SWISH, HARDSWISH);
    MNNCONVERT(LOG, LOG);
    MNNCONVERT(NEG, NEG);
    MNNCONVERT(ROUND, ROUND);
    MNNCONVERT(RSQRT, RSQRT);
    MNNCONVERT(SQUARE, SQUARE);
    MNNCONVERT(SQRT, SQRT);
    MNNCONVERT(SIN, SIN);
#undef MNNCONVERT
    return (MNN::UnaryOpOperation)0;
}
void UnaryTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto param = new MNN::UnaryOpT;
    param->opType = _convert(tfliteOpSet[tfliteOp->opcode_index]->builtin_code);
    dstOp->main.value = param;
}

using namespace tflite;

#define MNNCONVERT(x, y) REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_##x);
    MNNCONVERT(ABS, ABS);
    MNNCONVERT(COS, COS);
    MNNCONVERT(CEIL, CEIL);
    MNNCONVERT(EXP, EXP);
    MNNCONVERT(FLOOR, FLOOR);
    MNNCONVERT(HARD_SWISH, HARDSWISH);
    MNNCONVERT(LOG, LOG);
    MNNCONVERT(NEG, NEG);
    MNNCONVERT(ROUND, ROUND);
    MNNCONVERT(RSQRT, RSQRT);
    MNNCONVERT(SQUARE, SQUARE);
    MNNCONVERT(SQRT, SQRT);
    MNNCONVERT(SIN, SIN);
#undef MNNCONVERT
