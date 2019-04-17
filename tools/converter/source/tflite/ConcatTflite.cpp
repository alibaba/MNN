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
MNN::OpType ConcatTflite::opType(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedConcat;
    return MNN::OpType_Concat;
}
MNN::OpParameter ConcatTflite::type(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedConcat;
    return MNN::OpParameter_Axis;
}

void ConcatTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    const auto& tfliteConcatOption = tfliteOp->builtin_options.AsConcatenationOptions();
    if (quantizedModel) {
        auto concatParamQuan  = new MNN::QuantizedConcatT;
        concatParamQuan->axis = tfliteConcatOption->axis;

        for (int i = 0; i < tfliteOp->inputs.size(); i++) {
            const int inputIndex     = tfliteOp->inputs[i];
            const auto& inputTensor  = tfliteTensors[inputIndex];
            auto quantized_param_ptr = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            concatParamQuan->inputZeroPoint.push_back(inputTensor->quantization->zeroPoint[0]);
            concatParamQuan->inputScale.push_back(inputTensor->quantization->scale[0]);
        }

        const int outputIndex                 = tfliteOp->outputs[0];
        const auto& outputTensor              = tfliteTensors[outputIndex];
        concatParamQuan->outputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        concatParamQuan->outputQuantizedParam->zeroPoint = outputTensor->quantization->zeroPoint[0];
        concatParamQuan->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];
        concatParamQuan->activationType =
            static_cast<MNN::FusedActivation>(tfliteConcatOption->fused_activation_function);
        dstOp->main.value = concatParamQuan;
    } else {
        DCHECK(tfliteConcatOption->fused_activation_function == tflite::ActivationFunctionType_NONE);
        auto concatParamFloat  = new MNN::AxisT;
        concatParamFloat->axis = tfliteConcatOption->axis;
        dstOp->main.value      = concatParamFloat;
    }

    // set input output index
    dstOp->inputIndexes.resize(tfliteOp->inputs.size());
    dstOp->outputIndexes.resize(tfliteOp->outputs.size());
    for (int i = 0; i < tfliteOp->inputs.size(); i++) {
        dstOp->inputIndexes[i] = tfliteOp->inputs[i];
    }
    for (int i = 0; i < tfliteOp->outputs.size(); i++) {
        dstOp->outputIndexes[i] = tfliteOp->outputs[i];
    }
}

using namespace tflite;
REGISTER_CONVERTER(ConcatTflite, BuiltinOperator_CONCATENATION);
