//
//  BatchMatMulTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(BatchMatMulTflite);
MNN::OpType BatchMatMulTflite::opType(int quantizedModel) {
    return MNN::OpType_BatchMatMul;
}
MNN::OpParameter BatchMatMulTflite::type(int quantizedModel) {
    return MNN::OpParameter_BatchMatMulParam;
}

void BatchMatMulTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    // Do nothing
    dstOp->main.value = new MNN::BatchMatMulParamT;
    auto src = tfliteOp->builtin_options.AsBatchMatMulOptions();
    if (nullptr != src) {
        dstOp->main.AsBatchMatMulParam()->adjX = src->adj_x;
        dstOp->main.AsBatchMatMulParam()->adjY = src->adj_y;
    }
}


using namespace tflite;
REGISTER_CONVERTER(BatchMatMulTflite, BuiltinOperator_BATCH_MATMUL);
