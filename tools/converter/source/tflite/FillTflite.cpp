//
//  FillTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(FillTflite);

MNN::OpType FillTflite::opType(int quantizedModel) {
    return MNN::OpType_Fill;
}
MNN::OpParameter FillTflite::type(int quantizedModel) {
    return MNN::OpParameter_Fill;
}

void FillTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                     const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                     const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                     const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    dstOp->main.value = nullptr;
}
DECLARE_OP_COVERTER(ZerosLikeTflite);
MNN::OpType ZerosLikeTflite::opType(int quantizedModel) {
    return MNN::OpType_ZerosLike;
}
MNN::OpParameter ZerosLikeTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}

void ZerosLikeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                          const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                          const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    dstOp->main.value = nullptr;
}
using namespace tflite;
REGISTER_CONVERTER(FillTflite, BuiltinOperator_FILL);
REGISTER_CONVERTER(ZerosLikeTflite, BuiltinOperator_ZEROS_LIKE);
