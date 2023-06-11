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

MNN::OpType ReshapeTflite::opType(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedReshape;
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeTflite::type(int quantizedModel) {
    return MNN::OpParameter_Reshape;
}

void ReshapeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    auto reshapeParam     = new MNN::ReshapeT;
    reshapeParam->dimType = MNN::MNN_DATA_FORMAT_NHWC;

    dstOp->main.value = reshapeParam;
    auto reshape = tfliteOp->builtin_options.AsReshapeOptions();
    if (nullptr != reshape) {
        reshapeParam->dims = reshape->new_shape;
    }
}

using namespace tflite;
REGISTER_CONVERTER(ReshapeTflite, BuiltinOperator_RESHAPE);
