//
//  OneHotTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(OneHotTflite);

MNN::OpType OneHotTflite::opType(int quantizedModel) {
    return MNN::OpType_OneHot;
}
MNN::OpParameter OneHotTflite::type(int quantizedModel) {
    return MNN::OpParameter_OneHotParam;
}

void OneHotTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    auto ohParam = new MNN::OneHotParamT;
    auto opt=tfliteOp->builtin_options.AsOneHotOptions();
    ohParam->axis = opt->axis;
    dstOp->main.value = ohParam;

}

REGISTER_CONVERTER(OneHotTflite, BuiltinOperator_ONE_HOT);
