//
//  ResizeBilinear.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ResizeBilinear);

MNN::OpType ResizeBilinear::opType(bool quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpType_Interp;
    return MNN::OpType_Interp;
}
MNN::OpParameter ResizeBilinear::type(bool quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpParameter_Interp;
    return MNN::OpParameter_Interp;
}

void ResizeBilinear::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, bool quantizedModel) {
    DCHECK(!quantizedModel);
    auto resizeParam         = new MNN::InterpT;
    const auto& resizeOption = tfliteOp->builtin_options.AsResizeBilinearOptions();
    const auto& scaleTensor = tfliteTensors[tfliteOp->inputs[1]];
    auto scaleDataPtr       = reinterpret_cast<const int*>(tfliteModelBuffer[scaleTensor->buffer]->data.data());

    resizeParam->alignCorners = resizeOption->alignCorners;
    resizeParam->resizeType = 2;
    
    resizeParam->outputHeight = scaleDataPtr[1];
    resizeParam->outputWidth  = scaleDataPtr[0];
    
    resizeParam->widthScale = 1.0;
    resizeParam->heightScale = 1.0;

    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);
    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];

    dstOp->main.value = resizeParam;
}

using namespace tflite;
REGISTER_CONVERTER(ResizeBilinear, BuiltinOperator_RESIZE_BILINEAR);
