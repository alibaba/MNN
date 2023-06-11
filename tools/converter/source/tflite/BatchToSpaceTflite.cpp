//
//  BatchToSpaceTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(BatchToSpaceTflite);
MNN::OpType BatchToSpaceTflite::opType(int quantizedModel) {
    return MNN::OpType_Extra;
}
MNN::OpParameter BatchToSpaceTflite::type(int quantizedModel) {
    return MNN::OpParameter_Extra;
}

void BatchToSpaceTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto extraOpParam = new MNN::ExtraT;
    extraOpParam->engine = "Tflite";
    extraOpParam->type = "BatchToSpace";
    dstOp->main.value = extraOpParam;
}

DECLARE_OP_COVERTER(SpaceToBatchTflite);
MNN::OpType SpaceToBatchTflite::opType(int quantizedModel) {
    return MNN::OpType_Extra;
}
MNN::OpParameter SpaceToBatchTflite::type(int quantizedModel) {
    return MNN::OpParameter_Extra;
}

void SpaceToBatchTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto extraOpParam = new MNN::ExtraT;
    extraOpParam->engine = "Tflite";
    extraOpParam->type = "SpaceToBatch";
    dstOp->main.value = extraOpParam;
}

using namespace tflite;
REGISTER_CONVERTER(BatchToSpaceTflite, BuiltinOperator_BATCH_TO_SPACE_ND);
REGISTER_CONVERTER(SpaceToBatchTflite, BuiltinOperator_SPACE_TO_BATCH_ND);

