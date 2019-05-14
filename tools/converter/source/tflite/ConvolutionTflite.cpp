//
//  ConvolutionTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(Conv2DTflite);

MNN::OpType Conv2DTflite::opType(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_TfQuantizedConv2D;
    return MNN::OpType_Convolution;
}

MNN::OpParameter Conv2DTflite::type(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_TfQuantizedConv2D;
    return MNN::OpParameter_Convolution2D;
}

void Conv2DTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    // 3|2 inputs: input tensor, weight, (bias)
    const int inputSize = tfliteOp->inputs.size();
    DCHECK(inputSize == 2 || inputSize == 3) << "tflite Conv2D input ERROR! ";
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsConv2DOptions();
    // weight index
    const int weightIndex    = tfliteOp->inputs[1];
    const auto& weightTensor = tfliteTensors[weightIndex];
    // co kh kw ci
    const auto& weightShape = weightTensor->shape;
    DCHECK(weightShape.size() == 4) << "Conv2D weight ERROR!";
    const int co         = weightShape[0];
    const int kh         = weightShape[1];
    const int kw         = weightShape[2];
    const int ci         = weightShape[3];
    const int weightSize = co * kh * kw * ci;
    if (quantizedModel) {
        auto conv2dParamQuan         = new MNN::TfQuantizedConv2DT;
        conv2dParamQuan->modelFormat = MNN::ModeFormat_TFLITE;
        conv2dParamQuan->common      = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        // filterOffset
        conv2dParamQuan->filterQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        if (weightTensor->quantization->zeroPoint.size() > 0) {
            conv2dParamQuan->filterQuantizedParam->zeroPoint = weightTensor->quantization->zeroPoint[0];
        } else {
            conv2dParamQuan->filterQuantizedParam->zeroPoint = 0;
        }
        if (weightTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->filterQuantizedParam->scale = weightTensor->quantization->scale[0];
        } else {
            conv2dParamQuan->filterQuantizedParam->scale = 0.0;
        }

        // input
        const int inputIndex                 = tfliteOp->inputs[0];
        const auto& inputTensor              = tfliteTensors[inputIndex];
        conv2dParamQuan->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        if (inputTensor->quantization->zeroPoint.size() > 0) {
            conv2dParamQuan->inputQuantizedParam->zeroPoint = inputTensor->quantization->zeroPoint[0];
        } else {
            conv2dParamQuan->inputQuantizedParam->zeroPoint = 0;
        }
        if (inputTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->inputQuantizedParam->scale = inputTensor->quantization->scale[0];
        } else {
            conv2dParamQuan->inputQuantizedParam->scale = 0.0;
        }

        // output
        const int outputIndex                 = tfliteOp->outputs[0];
        const auto& outputTensor              = tfliteTensors[outputIndex];
        conv2dParamQuan->outputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);

        if (outputTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->outputQuantizedParam->zeroPoint = outputTensor->quantization->zeroPoint[0];
        } else {
            conv2dParamQuan->outputQuantizedParam->zeroPoint = 0;
        }

        if (outputTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->outputQuantizedParam->scale = outputTensor->quantization->scale[0];
        } else {
            conv2dParamQuan->outputQuantizedParam->scale = 0.0;
        }

        // kernel size
        conv2dParamQuan->common->kernelX     = kw;
        conv2dParamQuan->common->kernelY     = kh;
        conv2dParamQuan->common->outputCount = co;

        // default
        conv2dParamQuan->common->group   = 1;
        conv2dParamQuan->common->dilateX = tfliteConvOption->dilation_w_factor;
        conv2dParamQuan->common->dilateY = tfliteConvOption->dilation_h_factor;
        conv2dParamQuan->depthMultiplier = 1;

        // stride
        conv2dParamQuan->common->strideX = tfliteConvOption->stride_w;
        conv2dParamQuan->common->strideY = tfliteConvOption->stride_h;

        const auto tflitePadMode = tfliteConvOption->padding;
        if (tflitePadMode == tflite::Padding_SAME) {
            conv2dParamQuan->common->padMode = MNN::PadMode_SAME;
        } else if (tflitePadMode == tflite::Padding_VALID) {
            conv2dParamQuan->common->padMode = MNN::PadMode_VALID;
        }

        // weight
        DCHECK(weightTensor->type == tflite::TensorType_UINT8) << "Data type ERROR";

        // nhwc->hwcn
        int out_size = kh * kw * ci;
        int in_size  = co;
        std::vector<uint8_t> filter_hwcn;
        filter_hwcn.resize(weightSize);
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                filter_hwcn[i * in_size + j] = tfliteModelBuffer[weightTensor->buffer]->data[i + j * out_size];
            }
        }
        conv2dParamQuan->weight = filter_hwcn;

        conv2dParamQuan->biasflag = (inputSize == 3);
        DCHECK(conv2dParamQuan->biasflag == true);
        const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
        if (inputSize == 3) {
            DCHECK(biasTensor->type == tflite::TensorType_INT32) << "Bias Type ERROR";
            const auto& biasData                = tfliteModelBuffer[biasTensor->buffer]->data;
            conv2dParamQuan->biasQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            conv2dParamQuan->biasQuantizedParam->zeroPoint = biasTensor->quantization->zeroPoint[0];
            conv2dParamQuan->biasQuantizedParam->scale     = biasTensor->quantization->scale[0];
            DCHECK(biasData.size() / 4 == co) << "Bias Data ERROR";
            auto biasDataPtr               = biasData.data();
            const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
            std::vector<int32_t> biasInt32Vec(realBiasDataPtr, realBiasDataPtr + co);
            conv2dParamQuan->bias = biasInt32Vec;
        }

        conv2dParamQuan->activationType = (MNN::FusedActivation)tfliteConvOption->fused_activation_function;
        dstOp->main.value               = conv2dParamQuan;
    } else {
        auto convolution2DFloat = new MNN::Convolution2DT;
        // weight
        std::vector<float> weightData;
        weightData.resize(weightSize);
        auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        convertDataFormatTflite(originalWeightPtr, weightData.data(), kh, kw, ci, co);
        convolution2DFloat->weight = weightData;
        // bias
        std::vector<float> biasData(co, 0.0f);
        if (inputSize == 3) {
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto biasDataPtr       = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            ::memcpy(biasData.data(), biasDataPtr, sizeof(float) * co);
        }
        convolution2DFloat->bias = biasData;

        convolution2DFloat->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        auto& common               = convolution2DFloat->common;

        common->relu             = false;
        common->relu6            = false;
        const auto acticationFun = tfliteConvOption->fused_activation_function;
        if (acticationFun == tflite::ActivationFunctionType_RELU) {
            common->relu = true;
        } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
            common->relu6 = true;
        } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
            DLOG(ERROR) << "MNN Convolution do not Support fused_activation_function: " << acticationFun;
        }

        common->group       = 1;
        common->outputCount = co;
        common->inputCount  = ci;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->dilateX     = tfliteConvOption->dilation_w_factor;
        common->dilateY     = tfliteConvOption->dilation_h_factor;
        common->strideX     = tfliteConvOption->stride_w;
        common->strideY     = tfliteConvOption->stride_h;
        common->padMode     = MNN::PadMode_SAME;
        if (tfliteConvOption->padding == tflite::Padding_VALID) {
            common->padMode = MNN::PadMode_VALID;
        }

        dstOp->main.value = convolution2DFloat;
    }

    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);

    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(Conv2DTflite, BuiltinOperator_CONV_2D);
