//
//  ExpandDimTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ExpandDimTflite);
MNN::OpType ExpandDimTflite::opType(bool quantizedModel) {
    return MNN::OpType_ExpandDims;
}
MNN::OpParameter ExpandDimTflite::type(bool quantizedModel) {
    return MNN::OpParameter_ExpandDims;
}

void ExpandDimTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
    dstOp->main.value = new MNN::ExpandDimsT;
}


using namespace tflite;
REGISTER_CONVERTER(ExpandDimTflite, BuiltinOperator_EXPAND_DIMS);
