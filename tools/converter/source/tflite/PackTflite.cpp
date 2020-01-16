//
//  PackTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(PackTflite);
MNN::OpType PackTflite::opType(bool quantizedModel) {
    return MNN::OpType_Pack;
}
MNN::OpParameter PackTflite::type(bool quantizedModel) {
    return MNN::OpParameter_PackParam;
}

void PackTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto param = new MNN::PackParamT;
  auto opt=tfliteOp->builtin_options.AsPackOptions();
  param->axis=opt->axis;
  dstOp->main.value = param;
}


using namespace tflite;
REGISTER_CONVERTER(PackTflite, BuiltinOperator_PACK);
