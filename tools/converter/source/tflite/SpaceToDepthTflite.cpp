//
//  SpaceToDepthTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2019, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"
using namespace tflite;

DECLARE_OP_COVERTER(SpaceToDepthTflite);

MNN::OpType SpaceToDepthTflite::opType(bool quantizedModel) {
    return MNN::OpType_SpaceToDepth;
}
MNN::OpParameter SpaceToDepthTflite::type(bool quantizedModel) {
    return MNN::OpParameter_DepthSpaceParam;
}

void SpaceToDepthTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    auto spaceToDepthParam = new MNN::DepthSpaceParamT;
    auto opt=tfliteOp->builtin_options.AsSpaceToDepthOptions();
    spaceToDepthParam->blockSize = opt->block_size;
    dstOp->main.value = spaceToDepthParam;

}

REGISTER_CONVERTER(SpaceToDepthTflite, BuiltinOperator_SPACE_TO_DEPTH);
