//
//  UnpackTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(UnpackTflite);

MNN::OpType UnpackTflite::opType(int quantizedModel) {
    return MNN::OpType_Unpack;
}
MNN::OpParameter UnpackTflite::type(int quantizedModel) {
    return MNN::OpParameter_Axis;
}

void UnpackTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    auto axisT = new MNN::AxisT;
    auto opt=tfliteOp->builtin_options.AsUnpackOptions();
    axisT->axis = opt->axis;
    dstOp->main.value = axisT;
}

REGISTER_CONVERTER(UnpackTflite, BuiltinOperator_UNPACK);
