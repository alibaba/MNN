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
using namespace tflite;
DECLARE_OP_COVERTER(ResizeBilinear);

MNN::OpType ResizeBilinear::opType(int quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpType_Interp;
    return MNN::OpType_Interp;
}
MNN::OpParameter ResizeBilinear::type(int quantizedModel) {
    DCHECK(!quantizedModel);
    if (quantizedModel)
        return MNN::OpParameter_Interp;
    return MNN::OpParameter_Interp;
}

void ResizeBilinear::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    DCHECK(!quantizedModel);
    if (tfliteOp->inputs.size() < 2 || tfliteOp->outputs.empty()) {
        MNN_ERROR("[ERROR] Invalid TFLite Model: Resize has invalid inputs or outputs\n");
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    auto resizeParam         = new MNN::InterpT;
    const auto* scaleTensor  = tfliteAt(tfliteTensors, tfliteOp->inputs[1], "tensor");
    if (nullptr == scaleTensor) {
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    if (tfliteOp->opcode_index < 0 || tfliteOp->opcode_index >= static_cast<int>(tfliteOpSet.size()) ||
        tfliteOpSet[tfliteOp->opcode_index] == nullptr) {
        MNN_ERROR("[ERROR] Invalid TFLite Model: opcode index %d out of range (size %zu)\n", tfliteOp->opcode_index,
                  tfliteOpSet.size());
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    auto code = liteOpConverter::getOpCode(tfliteOpSet[tfliteOp->opcode_index].get());
    if (BuiltinOperator_RESIZE_NEAREST_NEIGHBOR == code) {
        const auto* nearest = tfliteOp->builtin_options.AsResizeNearestNeighborOptions();
        if (nullptr == nearest) {
            MNN_ERROR("[ERROR] Invalid TFLite Model: Resize missing options\n");
            dstOp->type = MNN::OpType_MAX;
            return;
        }
        resizeParam->resizeType   = 1;
        resizeParam->alignCorners = nearest->align_corners;
        if (nearest->half_pixel_centers) {
            resizeParam->ctm = MNN::CoordinateTransformationMode_HalfPixels;
        }
    } else if (BuiltinOperator_RESIZE_BILINEAR == code) {
        const auto* resizeOption = tfliteOp->builtin_options.AsResizeBilinearOptions();
        if (nullptr == resizeOption) {
            MNN_ERROR("[ERROR] Invalid TFLite Model: Resize missing options\n");
            dstOp->type = MNN::OpType_MAX;
            return;
        }
        resizeParam->resizeType   = 2;
        resizeParam->alignCorners = resizeOption->align_corners;
        if (resizeOption->half_pixel_centers) {
            resizeParam->ctm = MNN::CoordinateTransformationMode_HalfPixels;
        }
    } else {
        DCHECK(false);
    }
    const auto* scaleBuffer = tfliteAt(tfliteModelBuffer, static_cast<int>(scaleTensor->buffer), "buffer");
    if (nullptr == scaleBuffer || scaleBuffer->data.empty()) {
        MNN_ERROR("[ERROR] Invalid TFLite Model: Resize scale buffer is empty\n");
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    auto scaleDataPtr        = reinterpret_cast<const int *>(scaleBuffer->data.data());
    if (scaleBuffer->data.size() < 2 * sizeof(int)) {
        MNN_ERROR("[ERROR] Invalid TFLite Model: Resize scale buffer is too small\n");
        dstOp->type = MNN::OpType_MAX;
        return;
    }

    resizeParam->outputHeight = scaleDataPtr[0];
    resizeParam->outputWidth  = scaleDataPtr[1];

    resizeParam->widthScale  = 1.0;
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
REGISTER_CONVERTER(ResizeBilinear, BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
