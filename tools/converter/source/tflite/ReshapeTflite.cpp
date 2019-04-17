//
//  ReshapeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReshapeTflite);

MNN::OpType ReshapeTflite::opType(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedReshape;
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeTflite::type(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedReshape;
    return MNN::OpParameter_Reshape;
}

void ReshapeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    if (quantizedModel) {
        auto reshapeParamQuan         = new MNN::QuantizedReshapeT;
        reshapeParamQuan->modelFormat = MNN::ModeFormat_TFLITE;

        DCHECK(tfliteOp->inputs.size() == 2) << "tflite Reshape input ERROR";

        const auto& shapeTensor = tfliteTensors[tfliteOp->inputs[1]];
        DCHECK(shapeTensor->type == tflite::TensorType_INT32) << "ERROR";

        const auto& shapeData = tfliteModelBuffer[shapeTensor->buffer]->data;
        DCHECK(shapeTensor->shape[0] == shapeData.size() / 4) << "ERROR";

        auto dimPtr = reinterpret_cast<const int32_t*>(shapeData.data());
        std::vector<int> reshapDim(dimPtr, dimPtr + shapeTensor->shape[0]);
        reshapeParamQuan->dims = reshapDim;
        dstOp->main.value      = reshapeParamQuan;
    } else {
        auto reshapeParam     = new MNN::ReshapeT;
        reshapeParam->dimType = MNN::MNN_DATA_FORMAT_NHWC;

        const auto& shapeTensor = tfliteTensors[tfliteOp->inputs[1]];
        DCHECK(shapeTensor->type == tflite::TensorType_INT32) << "ERROR";

        const auto& shapeData = tfliteModelBuffer[shapeTensor->buffer]->data;
        DCHECK(shapeTensor->shape[0] == shapeData.size() / 4) << "ERROR";

        auto dimPtr = reinterpret_cast<const int32_t*>(shapeData.data());
        std::vector<int> reshapDim(dimPtr, dimPtr + shapeTensor->shape[0]);
        reshapeParam->dims = reshapDim;

        dstOp->main.value = reshapeParam;
    }

    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);
    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(ReshapeTflite, BuiltinOperator_RESHAPE);
