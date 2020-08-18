//
//  ReshapeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReshapeTflite);

MNN::OpType ReshapeTflite::opType(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedReshape;
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeTflite::type(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedReshape;
    return MNN::OpParameter_Reshape;
}

void ReshapeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    auto reshapeParam     = new MNN::ReshapeT;
    reshapeParam->dimType = MNN::MNN_DATA_FORMAT_NHWC;

    dstOp->main.value = reshapeParam;

    // set input output index
    dstOp->inputIndexes.resize(2);
    dstOp->outputIndexes.resize(1);
    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->inputIndexes[1]  = tfliteOp->inputs[1];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(ReshapeTflite, BuiltinOperator_RESHAPE);
