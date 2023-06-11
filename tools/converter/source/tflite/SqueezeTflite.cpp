//
//  SqueezeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SqueezeTflite);

MNN::OpType SqueezeTflite::opType(int quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpType_Squeeze;
    return MNN::OpType_Squeeze;
}
MNN::OpParameter SqueezeTflite::type(int quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpParameter_SqueezeParam;
    return MNN::OpParameter_SqueezeParam;
}

void SqueezeTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    DCHECK(!quantizedModel);
    auto squeezeParam         = new MNN::SqueezeParamT;
    const auto &squeezeOption = tfliteOp->builtin_options.AsSqueezeOptions();
    squeezeParam->squeezeDims = squeezeOption->squeeze_dims;
    
    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);
    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
    dstOp->main.value = squeezeParam;
}

using namespace tflite;
REGISTER_CONVERTER(SqueezeTflite, BuiltinOperator_SQUEEZE);
