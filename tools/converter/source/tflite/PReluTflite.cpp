//
//  PReluTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(PReluTflite);
MNN::OpType PReluTflite::opType(bool quantizedModel) {
    return MNN::OpType_PReLU;
}
MNN::OpParameter PReluTflite::type(bool quantizedModel) {
    return MNN::OpParameter_PRelu;
}

void PReluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                      const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                      const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                      const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){

    DCHECK(quantizedModel == false) << "tflite PRelu not support quantizedModel yet ERROR! ";
    // 2 inputs: input tensor, slope
    const int inputSize = tfliteOp->inputs.size();
    DCHECK(inputSize == 2) << "tflite PRelu input ERROR! ";
    // input, slope index
    const int inputIndex    = tfliteOp->inputs[0];
    const int slopeIndex    = tfliteOp->inputs[1];
    const auto& inputTensor = tfliteTensors[inputIndex];
    const auto& slopeTensor = tfliteTensors[slopeIndex];
    // input & slope shape
    const auto& inputShape = inputTensor->shape;
    const auto& slopeShape = slopeTensor->shape;
    int slopeSize = 1;
    for (auto shape: slopeShape)
        slopeSize *= (int)shape;

    DCHECK(inputShape.size() >= 4 && inputShape[3] == slopeSize) << "tflite PRelu slope count differs from input tensor ERROR! ";

    auto PRelu   = new MNN::PReluT;

    std::vector<float> slopeData;
    slopeData.resize(slopeSize);
    auto originalSlopePtr = reinterpret_cast<const float*>(tfliteModelBuffer[slopeTensor->buffer]->data.data());
    convertDataFormatTflite(originalSlopePtr, slopeData.data(), 1, 1, slopeSize, 1);

    PRelu->slope = slopeData;
    PRelu->slopeCount = slopeSize;

    dstOp->main.value = PRelu;

    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(PReluTflite, BuiltinOperator_PRELU);

