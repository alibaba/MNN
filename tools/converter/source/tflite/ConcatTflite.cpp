//
//  ConcatTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ConcatTflite);
MNN::OpType ConcatTflite::opType(int quantizedModel) {
    return MNN::OpType_Concat;
}
MNN::OpParameter ConcatTflite::type(int quantizedModel) {
    return MNN::OpParameter_Axis;
}

void ConcatTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    const auto& tfliteConcatOption = tfliteOp->builtin_options.AsConcatenationOptions();
    DCHECK(tfliteConcatOption->fused_activation_function == tflite::ActivationFunctionType_NONE);
    auto concatParamFloat  = new MNN::AxisT;
    concatParamFloat->axis = tfliteConcatOption->axis;
    dstOp->main.value      = concatParamFloat;
}

using namespace tflite;
REGISTER_CONVERTER(ConcatTflite, BuiltinOperator_CONCATENATION);
