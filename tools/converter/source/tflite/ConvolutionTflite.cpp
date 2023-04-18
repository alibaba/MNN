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
#include "core/OpCommonUtils.hpp"

DECLARE_OP_COVERTER(Conv2DTflite);

MNN::OpType Conv2DTflite::opType(int quantizedModel) {
    if (quantizedModel == 1)
        return MNN::OpType_TfQuantizedConv2D;
    return MNN::OpType_Convolution;
}

MNN::OpParameter Conv2DTflite::type(int quantizedModel) {
    if (quantizedModel == 1)
        return MNN::OpParameter_TfQuantizedConv2D;
    return MNN::OpParameter_Convolution2D;
}

void Conv2DTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    // 3|2 inputs: input tensor, weight, (bias)
    const int inputSize = tfliteOp->inputs.size();
    DCHECK(inputSize == 2 || inputSize == 3) << "tflite Conv2D input ERROR! ";
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsConv2DOptions();
    const int inputIndex     = tfliteOp->inputs[0];
    const int weightIndex    = tfliteOp->inputs[1];
    const int outputIndex    = tfliteOp->outputs[0];
    const auto& inputTensor  = tfliteTensors[inputIndex];
    const auto& weightTensor = tfliteTensors[weightIndex];
    const auto& outputTensor = tfliteTensors[outputIndex];
    // co kh kw ci
    const auto& weightShape = weightTensor->shape;
    DCHECK(weightShape.size() == 4) << "Conv2D weight ERROR!";
    const int co         = weightShape[0];
    const int kh         = weightShape[1];
    const int kw         = weightShape[2];
    const int ci         = weightShape[3];
    const int weightSize = co * kh * kw * ci;
    if (quantizedModel == 1) { // UINT8_QUANT
        auto conv2dParamQuan         = new MNN::TfQuantizedConv2DT;
        conv2dParamQuan->modelFormat = MNN::ModeFormat_TFLITE;
        conv2dParamQuan->common      = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        // filterOffset
        conv2dParamQuan->filterQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        if (weightTensor->quantization->zero_point.size() > 0) {
            conv2dParamQuan->filterQuantizedParam->zeroPoint = weightTensor->quantization->zero_point[0];
        } else {
            conv2dParamQuan->filterQuantizedParam->zeroPoint = 0;
        }
        if (weightTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->filterQuantizedParam->scale = weightTensor->quantization->scale[0];
        } else {
            conv2dParamQuan->filterQuantizedParam->scale = 0.0;
        }

        // input
        conv2dParamQuan->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        if (inputTensor->quantization->zero_point.size() > 0) {
            conv2dParamQuan->inputQuantizedParam->zeroPoint = inputTensor->quantization->zero_point[0];
        } else {
            conv2dParamQuan->inputQuantizedParam->zeroPoint = 0;
        }
        if (inputTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->inputQuantizedParam->scale = inputTensor->quantization->scale[0];
        } else {
            conv2dParamQuan->inputQuantizedParam->scale = 0.0;
        }

        // output
        conv2dParamQuan->outputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);

        if (outputTensor->quantization->scale.size() > 0) {
            conv2dParamQuan->outputQuantizedParam->zeroPoint = outputTensor->quantization->zero_point[0];
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
            conv2dParamQuan->biasQuantizedParam->zeroPoint = biasTensor->quantization->zero_point[0];
            conv2dParamQuan->biasQuantizedParam->scale     = biasTensor->quantization->scale[0];
            DCHECK(biasData.size() / 4 == co) << "Bias Data ERROR";
            auto biasDataPtr               = biasData.data();
            const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
            std::vector<int32_t> biasInt32Vec(realBiasDataPtr, realBiasDataPtr + co);
            conv2dParamQuan->bias = biasInt32Vec;
        }

        conv2dParamQuan->activationType = (MNN::FusedActivation)tfliteConvOption->fused_activation_function;
        dstOp->main.value               = conv2dParamQuan;
    } else if (quantizedModel == 2) { // INT8_QUANT
        std::unique_ptr<MNN::Convolution2DT> convolution2DQuant(new MNN::Convolution2DT);
        convolution2DQuant->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        auto& common               = convolution2DQuant->common;

        common->relu             = false;
        common->relu6            = false;
        const auto acticationFun = tfliteConvOption->fused_activation_function;
        if (acticationFun == tflite::ActivationFunctionType_RELU) {
            common->relu = true;
        } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
            common->relu6 = true;
        } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
            DLOG(ERROR) << "MNN Convolution do not Support fused_activation_function: " << acticationFun;
            dstOp->type = MNN::OpType_MAX;
            return;
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

        // weight
        if (tfliteModelBuffer[weightTensor->buffer]->data.data() == nullptr) {
            //MNN_ERROR("Has not const weight data for tflite convolution\n");
            dstOp->main.value = convolution2DQuant.release();
            return;
        }

        MNN_ASSERT(weightTensor->type == tflite::TensorType_INT8);
        MNN_ASSERT(weightTensor->quantization->scale.size() == co);
        MNN_ASSERT(tfliteModelBuffer[weightTensor->buffer]->data.size() == weightSize);
        convolution2DQuant->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        convolution2DQuant->quanParameter.reset(new MNN::IDSTQuanT);
        auto& quanParam = convolution2DQuant->quanParameter;
        quanParam->type = 4;
        quanParam->buffer.resize(weightSize);
        quanParam->alpha.resize(co);
        auto weight = reinterpret_cast<const int8_t*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        auto alpha = weightTensor->quantization->scale.data();
        float scaleIn = inputTensor->quantization->scale[0];
        float scaleOut = outputTensor->quantization->scale[0];
        // [co, kh, kw, ci] -> [co, ci, kh, kw]
        const int area = kh * kw;
        for (int i = 0; i < co; i ++) {
            for (int j = 0; j < ci; j++) {
                for (int k = 0; k < area; k++) {
                    quanParam->buffer[i * ci * area + j * area + k] = weight[i * area * ci + k * ci + j];
                }
            }
        }
        ::memcpy(quanParam->alpha.data(), alpha, co * sizeof(float));
        quanParam->scaleIn = scaleIn;
        quanParam->scaleOut = scaleOut;

        // bias
        convolution2DQuant->bias.resize(co);
        if (inputSize == 3) {
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto bias = reinterpret_cast<const int*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            // int to float
            for (int i = 0; i < co; i++) {
                convolution2DQuant->bias[i] = bias[i] * (scaleIn * alpha[i]);
            }
        }
        dstOp->main.value = convolution2DQuant.release();
    } else {
        std::unique_ptr<MNN::Convolution2DT> convolution2DFloat(new MNN::Convolution2DT);
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
            dstOp->type = MNN::OpType_MAX;
            return;
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

        // weight
        if (tfliteModelBuffer[weightTensor->buffer]->data.data() == nullptr) {
            //MNN_ERROR("Has not const weight data for tflite convolution\n");
            dstOp->main.value = convolution2DFloat.release();
            return;
        }
        std::vector<float> weightData;
        weightData.resize(weightSize);
        switch (weightTensor->type) {
            case tflite::TensorType_FLOAT32:
            {
                auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                convertDataFormatTflite(originalWeightPtr, weightData.data(), kh, kw, ci, co);
                break;
            }
            case tflite::TensorType_UINT8:
            {
                auto originalWeightPtr = reinterpret_cast<const int8_t*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
                convertDataFormatTfliteDequant<int8_t>(originalWeightPtr, weightData.data(), kh, kw, ci, co, weightTensor->quantization.get());
                break;
            }
            default:
                DLOG(ERROR) << "MNN Convolution do not Support weight type: " << weightTensor->type;
        }
        convolution2DFloat->weight = weightData;
        // bias
        std::vector<float> biasData(co, 0.0f);
        if (inputSize == 3) {
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto biasDataPtr       = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            ::memcpy(biasData.data(), biasDataPtr, sizeof(float) * co);
        }
        convolution2DFloat->bias = biasData;
        dstOp->main.value = convolution2DFloat.release();
    }

    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);

    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

DECLARE_OP_COVERTER(TransposeConvTflite);

MNN::OpType TransposeConvTflite::opType(int quantizedModel){
    return MNN::OpType_Deconvolution;
}

MNN::OpParameter TransposeConvTflite::type(int quantizedModel){
    return MNN::OpParameter_Convolution2D;
}

void TransposeConvTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors, const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer, const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel){


    DCHECK(!quantizedModel) << "TransposeConv not support quantized model";

    // 3|4 inputs: output shape, weight, input tensor, (bias)
    const int inputSize = tfliteOp->inputs.size();
    DCHECK(inputSize == 3 || inputSize == 4) << "tflite Conv2D input ERROR! ";
    /*
     enum Padding : byte { SAME, VALID }
     table TransposeConvOptions {
       padding:Padding;
       stride_w:int;
       stride_h:int;
     }
     */
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsTransposeConvOptions();
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
    {
        auto convolution2DFloat = new MNN::Convolution2DT;
        // weight
        std::vector<float> weightData;
        weightData.resize(weightSize);
        auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        convertDataFormatTflite(originalWeightPtr, weightData.data(), kh, kw, ci, co, true);
        convolution2DFloat->weight = weightData;
        // bias
        std::vector<float> biasData(co, 0.0f);
        if (inputSize == 4) {
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto biasDataPtr       = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            if(biasDataPtr){
                ::memcpy(biasData.data(), biasDataPtr, sizeof(float) * co);
            }
        }
        convolution2DFloat->bias = biasData;

        convolution2DFloat->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        auto& common               = convolution2DFloat->common;

        common->relu             = false;
        common->relu6            = false;

        common->group       = 1;
        common->outputCount = co;
        common->inputCount  = ci;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->dilateX     = 1;
        common->dilateY     = 1;
        common->strideX     = tfliteConvOption->stride_w;
        common->strideY     = tfliteConvOption->stride_h;
        common->padMode     = MNN::PadMode_SAME;
        common->hasOutputShape = true;

        dstOp->main.value = convolution2DFloat;
    }

    // set input output index
    dstOp->inputIndexes.resize(2);
    dstOp->outputIndexes.resize(1);

    dstOp->inputIndexes[0]  = tfliteOp->inputs[2];
    dstOp->inputIndexes[1]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}


DECLARE_OP_COVERTER(FullConnectedTflite);

MNN::OpType FullConnectedTflite::opType(int quantizedModel) {
    return MNN::OpType_Extra;
}

MNN::OpParameter FullConnectedTflite::type(int quantizedModel) {
    return MNN::OpParameter_Extra;
}

void FullConnectedTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    dstOp->main.value = new MNN::ExtraT;
    auto dstP = dstOp->main.AsExtra();
    dstP->engine = "Tflite";
    dstP->type = "FULL_CONNECT";
    const auto& option = tfliteOp->builtin_options.AsFullyConnectedOptions();
    dstP->attr.resize(3);
    dstP->attr[0].reset(new MNN::AttributeT);
    dstP->attr[0]->key = "keep_num_dims";
    dstP->attr[0]->b = option->keep_num_dims;

    dstP->attr[1].reset(new MNN::AttributeT);
    dstP->attr[1]->key = "weights_format";
    dstP->attr[1]->i = option->weights_format;

    dstP->attr[2].reset(new MNN::AttributeT);
    dstP->attr[2]->key = "fused_activation_function";
    dstP->attr[2]->i = option->fused_activation_function;
}


using namespace tflite;
REGISTER_CONVERTER(Conv2DTflite, BuiltinOperator_CONV_2D);
REGISTER_CONVERTER(TransposeConvTflite, BuiltinOperator_TRANSPOSE_CONV);
REGISTER_CONVERTER(FullConnectedTflite, BuiltinOperator_FULLY_CONNECTED);
