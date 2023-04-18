//
//  BroadCastAdd.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(AddTflite);

MNN::OpType AddTflite::opType(int quantizedModel) {
    if (quantizedModel == 1)
        return MNN::OpType_QuantizedAdd;
    return MNN::OpType_Extra;
}

MNN::OpParameter AddTflite::type(int quantizedModel) {
    if (quantizedModel == 1)
        return MNN::OpParameter_QuantizedAdd;
    return MNN::OpParameter_Extra;
}

void AddTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    const auto& addOption = tfliteOp->builtin_options.AsAddOptions();
    if (quantizedModel == 1) {
        auto AddParam = new MNN::QuantizedAddT;

        DCHECK(tfliteOp->inputs.size() == 2) << "tflite Reshape input ERROR";

        // input1
        const int input1Index                     = tfliteOp->inputs[0];
        const auto& input1Tensor                  = tfliteTensors[input1Index];
        AddParam->input1QuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->input1QuantizedParam->zeroPoint = input1Tensor->quantization->zero_point[0];
        AddParam->input1QuantizedParam->scale     = input1Tensor->quantization->scale[0];

        // input1
        const int input2Index                     = tfliteOp->inputs[1];
        const auto& input2Tensor                  = tfliteTensors[input2Index];
        AddParam->input2QuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->input2QuantizedParam->zeroPoint = input2Tensor->quantization->zero_point[0];
        AddParam->input2QuantizedParam->scale     = input2Tensor->quantization->scale[0];

        // output
        const int outputIndex                     = tfliteOp->outputs[0];
        const auto& outputTensor                  = tfliteTensors[outputIndex];
        AddParam->outputQuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->outputQuantizedParam->zeroPoint = outputTensor->quantization->zero_point[0];
        AddParam->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];

        AddParam->activationType = static_cast<MNN::FusedActivation>(addOption->fused_activation_function);

        dstOp->main.value = AddParam;
    } else {
        auto extraOpParam = new MNN::ExtraT;
        extraOpParam->engine = "Tflite";
        extraOpParam->type = "BinaryActivation";
        extraOpParam->attr.resize(2);
        extraOpParam->attr[0].reset(new MNN::AttributeT);
        extraOpParam->attr[1].reset(new MNN::AttributeT);
        extraOpParam->attr[0]->key = "opType";
        extraOpParam->attr[0]->i = tflite::BuiltinOperator_ADD;
        extraOpParam->attr[1]->key = "activationType";
        extraOpParam->attr[1]->i = addOption->fused_activation_function;
        dstOp->main.value = extraOpParam;
    }
}

using namespace tflite;
REGISTER_CONVERTER(AddTflite, BuiltinOperator_ADD);
