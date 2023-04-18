//
//  DepthwiseConv2DTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(DepthwiseConv2DTflite);

MNN::OpType DepthwiseConv2DTflite::opType(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedDepthwiseConv2D;
    return MNN::OpType_ConvolutionDepthwise;
}

MNN::OpParameter DepthwiseConv2DTflite::type(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_TfQuantizedConv2D;
    return MNN::OpParameter_Convolution2D;
}

void DepthwiseConv2DTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                                const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
                                int quantizedModel) {
    // 3|2 inputs: input tensor, weight, (bias)
    const int inputSize = tfliteOp->inputs.size();
    DCHECK(inputSize == 2 || inputSize == 3) << "tflite DepthiwiseConv2D input ERROR! ";
    // weight index
    const int weightIndex    = tfliteOp->inputs[1];
    const auto& weightTensor = tfliteTensors[weightIndex];
    // co kh kw ci
    const auto& weightShape = weightTensor->shape;
    DCHECK(weightShape.size() == 4) << "Conv2D weight ERROR!";
    // const int co = weightShape[0];
    const int kh                 = weightShape[1];
    const int kw                 = weightShape[2];
    const int ci                 = weightShape[3];
    const int weightSize         = kh * kw * ci;
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsDepthwiseConv2DOptions();
    if (quantizedModel) {
        auto depthwiseConv2dParamQuan         = new MNN::TfQuantizedConv2DT;
        depthwiseConv2dParamQuan->modelFormat = MNN::ModeFormat_TFLITE;
        depthwiseConv2dParamQuan->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);

        // filterOffset
        depthwiseConv2dParamQuan->filterQuantizedParam =
            std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        depthwiseConv2dParamQuan->filterQuantizedParam->zeroPoint = weightTensor->quantization->zero_point[0];
        depthwiseConv2dParamQuan->filterQuantizedParam->scale     = weightTensor->quantization->scale[0];

        // input
        const int inputIndex                          = tfliteOp->inputs[0];
        const auto& inputTensor                       = tfliteTensors[inputIndex];
        depthwiseConv2dParamQuan->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        depthwiseConv2dParamQuan->inputQuantizedParam->zeroPoint = inputTensor->quantization->zero_point[0];
        depthwiseConv2dParamQuan->inputQuantizedParam->scale     = inputTensor->quantization->scale[0];

        // output
        const int outputIndex    = tfliteOp->outputs[0];
        const auto& outputTensor = tfliteTensors[outputIndex];
        depthwiseConv2dParamQuan->outputQuantizedParam =
            std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        depthwiseConv2dParamQuan->outputQuantizedParam->zeroPoint = outputTensor->quantization->zero_point[0];
        depthwiseConv2dParamQuan->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];

        // kernel size
        depthwiseConv2dParamQuan->common->kernelX     = kw;
        depthwiseConv2dParamQuan->common->kernelY     = kh;
        depthwiseConv2dParamQuan->common->outputCount = ci;

        // default
        depthwiseConv2dParamQuan->common->group   = 1;
        depthwiseConv2dParamQuan->common->dilateX = tfliteConvOption->dilation_w_factor;
        depthwiseConv2dParamQuan->common->dilateY = tfliteConvOption->dilation_h_factor;

        depthwiseConv2dParamQuan->depthMultiplier = tfliteConvOption->depth_multiplier;
        // stride
        depthwiseConv2dParamQuan->common->strideX = tfliteConvOption->stride_w;
        depthwiseConv2dParamQuan->common->strideY = tfliteConvOption->stride_h;

        const auto tflitePadMode = tfliteConvOption->padding;
        if (tflitePadMode == tflite::Padding_SAME) {
            depthwiseConv2dParamQuan->common->padMode = MNN::PadMode_SAME;
        } else if (tflitePadMode == tflite::Padding_VALID) {
            depthwiseConv2dParamQuan->common->padMode = MNN::PadMode_VALID;
        }

        // weight
        DCHECK(weightTensor->type == tflite::TensorType_UINT8) << "Data type ERROR";
        depthwiseConv2dParamQuan->weight   = tfliteModelBuffer[weightTensor->buffer]->data;
        depthwiseConv2dParamQuan->biasflag = inputSize == 3;
        // have bias
        const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
        if (inputSize == 3) {
            DCHECK(biasTensor->type == tflite::TensorType_INT32) << "Bias Type ERROR";

            const auto& biasData = tfliteModelBuffer[biasTensor->buffer]->data;
            depthwiseConv2dParamQuan->biasQuantizedParam =
                std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            depthwiseConv2dParamQuan->biasQuantizedParam->zeroPoint = biasTensor->quantization->zero_point[0];
            depthwiseConv2dParamQuan->biasQuantizedParam->scale     = biasTensor->quantization->scale[0];

            auto shape = biasTensor->shape;

            DCHECK(biasData.size() / 4 == ci) << "Bias Data ERROR";
            auto biasDataPtr               = biasData.data();
            const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
            std::vector<int32_t> biasInt32Vec(realBiasDataPtr, realBiasDataPtr + ci);
            depthwiseConv2dParamQuan->bias = biasInt32Vec;
        }
        depthwiseConv2dParamQuan->activationType =
            static_cast<MNN::FusedActivation>(tfliteConvOption->fused_activation_function);
        dstOp->main.value = depthwiseConv2dParamQuan;
    } else {
        auto depthwiseConv2dParamFloat = new MNN::Convolution2DT;
        std::vector<float> weightData;
        weightData.resize(weightSize);
        auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        
        if(originalWeightPtr){
            convertDataFormatTflite(originalWeightPtr, weightData.data(), kh, kw, ci, 1);
            depthwiseConv2dParamFloat->weight = weightData;
        }
        // bias
        if (inputSize == 3) {
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto originalBiasPtr = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            if (originalBiasPtr) {
                std::vector<float> biasData(ci, 0.0f);
                ::memcpy(biasData.data(), originalBiasPtr, sizeof(float) * ci);
                depthwiseConv2dParamFloat->bias   = biasData;
            }
        }
        depthwiseConv2dParamFloat->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        auto& common                      = depthwiseConv2dParamFloat->common;

        common->relu       = false;
        common->relu6      = false;
        auto acticationFun = tfliteConvOption->fused_activation_function;
        if (acticationFun == tflite::ActivationFunctionType_RELU) {
            common->relu = true;
        } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
            common->relu6 = true;
        } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
            DLOG(ERROR) << "MNN Convolution do not Support fused_activation_function: " << acticationFun;
        }

        common->group       = ci;
        common->outputCount = ci;
        common->inputCount  = ci;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->dilateX     = tfliteConvOption->dilation_w_factor;
        common->dilateY     = tfliteConvOption->dilation_h_factor;
        common->strideX     = tfliteConvOption->stride_w;
        common->strideY     = tfliteConvOption->stride_h;
        common->padMode     = MNN::PadMode_SAME;
        if (tfliteConvOption->depth_multiplier > 1) {
            if (ci == tfliteConvOption->depth_multiplier) {
                // Special case, turn to convolution
                dstOp->type = MNN::OpType_Convolution;
                common->outputCount = tfliteConvOption->depth_multiplier;
                common->inputCount = 1;
                common->group = 1;
            } else {
                DLOG(ERROR) << "MNN don't support tflite's depth_multiplier, please turn to pb or onnx";
            }
        }

        if (tfliteConvOption->padding == tflite::Padding_VALID) {
            common->padMode = MNN::PadMode_VALID;
        }

        dstOp->main.value = depthwiseConv2dParamFloat;
    }
    
    // set input output index
    {
        auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        if(originalWeightPtr){
            dstOp->inputIndexes.resize(1);
            dstOp->outputIndexes.resize(1);
            dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
            dstOp->outputIndexes[0] = tfliteOp->outputs[0];
        } else if (inputSize == 3 && tfliteModelBuffer[tfliteTensors[tfliteOp->inputs[2]]->buffer]->data.data() != nullptr) {
            dstOp->inputIndexes.resize(2);
            dstOp->outputIndexes.resize(1);
            dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
            dstOp->inputIndexes[1]  = tfliteOp->inputs[1];
            dstOp->outputIndexes[0] = tfliteOp->outputs[0];
        } else {
            dstOp->inputIndexes.resize(inputSize);
            dstOp->outputIndexes.resize(1);
            dstOp->outputIndexes[0] = tfliteOp->outputs[0];
            for(int i = 0; i < inputSize; ++i){
                dstOp->inputIndexes[i] = tfliteOp->inputs[i];
            }
        }
    }
}

using namespace tflite;
REGISTER_CONVERTER(DepthwiseConv2DTflite, BuiltinOperator_DEPTHWISE_CONV_2D);
