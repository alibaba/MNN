//
//  GatherTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/5.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(GatherTflite);
MNN::OpType GatherTflite::opType(bool quantizedModel) {
    return MNN::OpType_Gather;
}
MNN::OpParameter GatherTflite::type(bool quantizedModel) {
    return MNN::OpParameter_Gather;
}

void GatherTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
     auto parameter  = new MNN::GatherT;
     auto opt=tfliteOp->builtin_options.AsGatherOptions();
     parameter->axis = opt->axis;
     dstOp->main.value = parameter;
}


using namespace tflite;
REGISTER_CONVERTER(GatherTflite, BuiltinOperator_GATHER);
