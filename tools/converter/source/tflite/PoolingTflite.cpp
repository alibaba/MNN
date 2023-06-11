//
//  PoolingTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(PoolingTflite);

MNN::OpType PoolingTflite::opType(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedAvgPool;
    return MNN::OpType_Pooling;
}
MNN::OpParameter PoolingTflite::type(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedAvgPool;
    return MNN::OpParameter_Pool;
}

void PoolingTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    const auto& tflitePoolOption = tfliteOp->builtin_options.AsPool2DOptions();

    if (quantizedModel) {
        auto quantizedAvgPoolQuan         = new MNN::QuantizedAvgPoolT;
        quantizedAvgPoolQuan->modelFormat = MNN::ModeFormat_TFLITE;

        quantizedAvgPoolQuan->kernelX = tflitePoolOption->filter_width;
        ;
        quantizedAvgPoolQuan->kernelY = tflitePoolOption->filter_height;

        quantizedAvgPoolQuan->strideX = tflitePoolOption->stride_w;
        quantizedAvgPoolQuan->strideY = tflitePoolOption->stride_h;

        // output
        const int outputIndex    = tfliteOp->outputs[0];
        const auto& outputTensor = tfliteTensors[outputIndex];

        CalculateActivationRangeUint8((MNN::FusedActivation)tflitePoolOption->fused_activation_function,
                                      outputTensor->quantization, &quantizedAvgPoolQuan->outputActivationMin,
                                      &quantizedAvgPoolQuan->outputActivationMax);

        if (tflitePoolOption->padding == tflite::Padding_SAME) {
            quantizedAvgPoolQuan->padType = MNN::PoolPadType_SAME;
        } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
            quantizedAvgPoolQuan->padType = MNN::PoolPadType_VALID;
        }
        dstOp->main.value = quantizedAvgPoolQuan;
    } else {
        DCHECK(tflitePoolOption->fused_activation_function == tflite::ActivationFunctionType_NONE);
        auto poolParam     = new MNN::PoolT;
        poolParam->kernelX = tflitePoolOption->filter_width;
        poolParam->kernelY = tflitePoolOption->filter_height;
        poolParam->strideY = tflitePoolOption->stride_h;
        poolParam->strideX = tflitePoolOption->stride_w;
        if (tflitePoolOption->padding == tflite::Padding_SAME) {
            poolParam->padType = MNN::PoolPadType_SAME;
        } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
            poolParam->padType = MNN::PoolPadType_VALID;
        }

        poolParam->type    = MNN::PoolType_AVEPOOL;
        const auto opIndex = tfliteOp->opcode_index;
        auto opType        = tfliteOpSet[opIndex]->builtin_code;
        if (opType == tflite::BuiltinOperator_MAX_POOL_2D) {
            poolParam->type = MNN::PoolType_MAXPOOL;
        }

        poolParam->isGlobal = false;
        dstOp->main.value   = poolParam;
    }

    DCHECK(tfliteOp->inputs.size() == 1) << "Tflite pooling input ERROR";
    
    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);
    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(PoolingTflite, BuiltinOperator_AVERAGE_POOL_2D);
REGISTER_CONVERTER(PoolingTflite, BuiltinOperator_MAX_POOL_2D);
