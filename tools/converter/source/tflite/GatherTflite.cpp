//
//  GatherTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(GatherTflite);
MNN::OpType GatherTflite::opType(int quantizedModel) {
    return MNN::OpType_Gather;
}
MNN::OpParameter GatherTflite::type(int quantizedModel) {
    return MNN::OpParameter_Axis;
}

void GatherTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
     auto parameter  = new MNN::AxisT;
     auto opt=tfliteOp->builtin_options.AsGatherOptions();
     parameter->axis = opt->axis;
     dstOp->main.value = parameter;
}


using namespace tflite;
REGISTER_CONVERTER(GatherTflite, BuiltinOperator_GATHER);
