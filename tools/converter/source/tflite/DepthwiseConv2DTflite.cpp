//
//  DepthwiseConv2DTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <limits>

#include "TfliteUtils.hpp"
#include "liteOpConverter.hpp"
#include "core/IDSTEncoder.hpp"

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
static void _writeCommon(MNN::OpT* dstOp, Convolution2DCommonT* common, tflite::OperatorT* tfliteOp, int ci, int kw, int kh) {
    common->relu       = false;
    common->relu6      = false;
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsDepthwiseConv2DOptions();
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
}

void DepthwiseConv2DTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                                const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
                                int quantizedModel) {
    // 3|2 inputs: input tensor, weight, (bias)
    const int inputSize = tfliteOp->inputs.size();
    if (inputSize < 2 || tfliteOp->outputs.empty()) {
        MNN_ERROR("[ERROR] Invalid TFLite Model: DEPTHWISE_CONV_2D has invalid inputs or outputs\n");
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    // weight index
    const int weightIndex    = tfliteOp->inputs[1];
    const auto* weightTensor = tfliteAt(tfliteTensors, weightIndex, "tensor");
    if (nullptr == weightTensor) {
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    const auto* weightBuffer = tfliteAt(tfliteModelBuffer, static_cast<int>(weightTensor->buffer), "buffer");
    if (nullptr == weightBuffer) {
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    // co kh kw ci
    const auto& weightShape = weightTensor->shape;
    if (4 != weightShape.size()) {
        DLOG(ERROR) << "DEPTHWISE_CONV_2D weight shape is not 4-D";
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    // const int co = weightShape[0];
    const int kh                 = weightShape[1];
    const int kw                 = weightShape[2];
    const int ci                 = weightShape[3];
    if (kh <= 0 || kw <= 0 || ci <= 0) {
        DLOG(ERROR) << "DEPTHWISE_CONV_2D weight shape contains non-positive dimension";
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    const int32_t weightDims[3] = {kh, kw, ci};
    int weightSize = 0;
    if (!computeTfliteWeightSize(weightDims, 3, &weightSize)) {
        DLOG(ERROR) << "DEPTHWISE_CONV_2D weight size overflow: " << kh << "x" << kw << "x" << ci;
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    const auto& tfliteConvOption = tfliteOp->builtin_options.AsDepthwiseConv2DOptions();
    if (nullptr == tfliteConvOption) {
        DLOG(ERROR) << "DEPTHWISE_CONV_2D operator carries no DepthwiseConv2DOptions";
        dstOp->type = MNN::OpType_MAX;
        return;
    }
    if (weightTensor->type == tflite::TensorType_INT8) {
        quantizedModel = 2;
        dstOp->type = MNN::OpType_ConvolutionDepthwise;
        dstOp->main.type = MNN::OpParameter_Convolution2D;
    } else if (weightTensor->type == tflite::TensorType_UINT8) {
        quantizedModel = 1;
        dstOp->type = MNN::OpType_DepthwiseConvInt8;
        dstOp->main.type = MNN::OpParameter_TfQuantizedConv2D;
    } else {
        MNN_ASSERT(weightTensor->type == tflite::TensorType_FLOAT32);
        quantizedModel = 0;
        dstOp->type = MNN::OpType_ConvolutionDepthwise;
        dstOp->main.type = MNN::OpParameter_Convolution2D;
    }

    std::unique_ptr<MNN::Convolution2DCommonT> dstCommon(new MNN::Convolution2DCommonT);
    _writeCommon(dstOp, dstCommon.get(), tfliteOp.get(), ci, kw, kh);
    if (quantizedModel) {
        if (weightTensor->type == tflite::TensorType_INT8) {
            dstOp->type = OpType_ConvolutionDepthwise;
            dstOp->main.type = OpParameter_Convolution2D;
            auto depthwiseConv2dParamFloat = new MNN::Convolution2DT;
            depthwiseConv2dParamFloat->common = std::move(dstCommon);
            dstOp->main.value = depthwiseConv2dParamFloat;
            // Bias Turn to float
            auto outputCount = depthwiseConv2dParamFloat->common->outputCount;
            depthwiseConv2dParamFloat->bias.resize(ci);
            ::memset(depthwiseConv2dParamFloat->bias.data(), 0, outputCount * sizeof(float));
            if (inputSize == 3) {
                const auto* biasTensor = tfliteAt(tfliteTensors, tfliteOp->inputs[2], "tensor");
                if (nullptr == biasTensor) {
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
                const auto* biasBuffer = tfliteAt(tfliteModelBuffer, static_cast<int>(biasTensor->buffer), "buffer");
                const auto* biasQuant = biasTensor->quantization.get();
                if (nullptr == biasQuant || biasQuant->scale.empty()) {
                    DLOG(ERROR) << "DEPTHWISE_CONV_2D bias tensor carries no quantization scale";
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
                const std::vector<uint8_t> emptyData;
                const auto& biasData = biasBuffer == nullptr ? emptyData : biasBuffer->data;
                if (biasData.size() >= sizeof(int32_t) * outputCount) {
                    if (biasQuant->scale.size() == 1) {
                        auto scale = biasQuant->scale[0];
                        auto zero = biasQuant->zero_point.empty() ? 0 : biasQuant->zero_point[0];
                        auto biasDataPtr = biasData.data();
                        const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
                        for (int i = 0; i < outputCount; ++i) {
                            depthwiseConv2dParamFloat->bias[i] = (float)(realBiasDataPtr[i] - zero) * scale;
                        }
                    } else {
                        // per-channel quantization; require the vectors long enough to cover every output
                        if ((int)biasQuant->scale.size() < outputCount ||
                            (int)biasQuant->zero_point.size() < outputCount) {
                            DLOG(ERROR) << "DEPTHWISE_CONV_2D per-channel bias quantization too short (need "
                                        << outputCount << " got scale=" << biasQuant->scale.size()
                                        << " zp=" << biasQuant->zero_point.size() << ")";
                            dstOp->type = MNN::OpType_MAX;
                            return;
                        }
                        auto biasDataPtr = biasData.data();
                        const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
                        for (int i = 0; i < outputCount; ++i) {
                            depthwiseConv2dParamFloat->bias[i] =
                                (float)(realBiasDataPtr[i] - biasQuant->zero_point[i]) * biasQuant->scale[i];
                        }
                    }
                } else {
                    DLOG(ERROR) << "DEPTHWISE_CONV_2D bias buffer needs " << (sizeof(int32_t) * outputCount)
                                << " bytes, got " << biasData.size();
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
            }
            // Weight
            // Transpose first
            std::vector<int8_t> transposeWeight(kw * kh * ci);
            const auto& weightData = weightBuffer->data;
            auto weightDataPtr = (int8_t*)weightData.data();
            if (weightDataPtr == nullptr || weightData.size() < (size_t)kw * kh * ci) {
                DLOG(ERROR) << "DEPTHWISE_CONV_2D INT8 weight buffer is too small";
                dstOp->type = MNN::OpType_MAX;
                return;
            }
            for (int i=0; i<ci; ++i) {
                for (int j=0; j<kw*kh; ++j) {
                    transposeWeight[i*kw*kh+j] = weightDataPtr[i+j*ci];
                }
            }
            auto quan = IDSTEncoder::encode(nullptr, weightTensor->quantization->scale, kw * kh, ci, false, transposeWeight.data(), -128);
            depthwiseConv2dParamFloat->quanParameter = std::move(quan);
        } else {
            // For old uint8 model
            std::unique_ptr<MNN::TfQuantizedConv2DT> depthwiseConv2dParamQuan(new MNN::TfQuantizedConv2DT);
            depthwiseConv2dParamQuan->modelFormat = MNN::ModeFormat_TFLITE;
            depthwiseConv2dParamQuan->common = std::move(dstCommon);

            // filterOffset
            depthwiseConv2dParamQuan->filterQuantizedParam =
                std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            if (weightTensor->quantization->zero_point.size() > 0) {
                depthwiseConv2dParamQuan->filterQuantizedParam->zeroPoint = weightTensor->quantization->zero_point[0];
            } else {
                depthwiseConv2dParamQuan->filterQuantizedParam->zeroPoint = 0;
            }
            if (weightTensor->quantization->scale.size() > 0) {
                depthwiseConv2dParamQuan->filterQuantizedParam->scale = weightTensor->quantization->scale[0];
            } else {
                depthwiseConv2dParamQuan->filterQuantizedParam->scale = 0.0f;
            }

            // input
            const int inputIndex                          = tfliteOp->inputs[0];
            const auto* inputTensor = tfliteAt(tfliteTensors, inputIndex, "tensor");
            if (nullptr == inputTensor) {
                dstOp->type = MNN::OpType_MAX;
                return;
            }
            depthwiseConv2dParamQuan->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            if (inputTensor->quantization->zero_point.size() > 0) {
                depthwiseConv2dParamQuan->inputQuantizedParam->zeroPoint = inputTensor->quantization->zero_point[0];
            } else {
                depthwiseConv2dParamQuan->inputQuantizedParam->zeroPoint = 0;
            }
            if (inputTensor->quantization->scale.size() > 0) {
                depthwiseConv2dParamQuan->inputQuantizedParam->scale = inputTensor->quantization->scale[0];
            } else {
                depthwiseConv2dParamQuan->inputQuantizedParam->scale = 0.0f;
            }

            // output
            const int outputIndex    = tfliteOp->outputs[0];
            const auto* outputTensor = tfliteAt(tfliteTensors, outputIndex, "tensor");
            if (nullptr == outputTensor) {
                dstOp->type = MNN::OpType_MAX;
                return;
            }
            depthwiseConv2dParamQuan->outputQuantizedParam =
                std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
            if (outputTensor->quantization->zero_point.size() > 0) {
                depthwiseConv2dParamQuan->outputQuantizedParam->zeroPoint = outputTensor->quantization->zero_point[0];
            } else {
                depthwiseConv2dParamQuan->outputQuantizedParam->zeroPoint = 0;
            }
            if (outputTensor->quantization->scale.size() > 0) {
                depthwiseConv2dParamQuan->outputQuantizedParam->scale = outputTensor->quantization->scale[0];
            } else {
                depthwiseConv2dParamQuan->outputQuantizedParam->scale = 0.0f;
            }

            depthwiseConv2dParamQuan->depthMultiplier = tfliteConvOption->depth_multiplier;

            // weight
            DCHECK(weightTensor->type == tflite::TensorType_UINT8) << "Data type ERROR";
            depthwiseConv2dParamQuan->weight = weightBuffer->data;
            depthwiseConv2dParamQuan->biasflag = inputSize == 3;
            // have bias
            if (inputSize == 3) {
                const auto* biasTensor = tfliteAt(tfliteTensors, tfliteOp->inputs[2], "tensor");
                if (nullptr == biasTensor) {
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
                DCHECK(biasTensor->type == tflite::TensorType_INT32) << "Bias Type ERROR";

                const auto* biasBuffer = tfliteAt(tfliteModelBuffer, static_cast<int>(biasTensor->buffer), "buffer");
                const std::vector<uint8_t> emptyData;
                const auto& biasData = biasBuffer == nullptr ? emptyData : biasBuffer->data;
                depthwiseConv2dParamQuan->biasQuantizedParam =
                    std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
                const auto* biasQuant = biasTensor->quantization.get();
                if (nullptr == biasQuant || biasQuant->scale.empty() || biasQuant->zero_point.empty()) {
                    DLOG(ERROR) << "DEPTHWISE_CONV_2D bias tensor carries no quantization scale/zero point";
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
                depthwiseConv2dParamQuan->biasQuantizedParam->zeroPoint = biasQuant->zero_point[0];
                depthwiseConv2dParamQuan->biasQuantizedParam->scale = biasQuant->scale[0];

                auto shape = biasTensor->shape;

                if (biasData.size() >= sizeof(int32_t) * ci) {
                    auto biasDataPtr = biasData.data();
                    const int32_t* realBiasDataPtr = (int32_t*)biasDataPtr;
                    std::vector<int32_t> biasInt32Vec(realBiasDataPtr, realBiasDataPtr + ci);
                    depthwiseConv2dParamQuan->bias = biasInt32Vec;
                } else {
                    DLOG(ERROR) << "DEPTHWISE_CONV_2D bias buffer needs " << (sizeof(int32_t) * ci) << " bytes, got "
                                << biasData.size();
                    dstOp->type = MNN::OpType_MAX;
                    return;
                }
            }
            depthwiseConv2dParamQuan->activationType =
                static_cast<MNN::FusedActivation>(tfliteConvOption->fused_activation_function);
            dstOp->main.value = depthwiseConv2dParamQuan.release();
        }
    } else {
        std::unique_ptr<MNN::Convolution2DT> depthwiseConv2dParamFloat(new MNN::Convolution2DT);
        std::vector<float> weightData;
        weightData.resize(weightSize);
        auto originalWeightPtr = reinterpret_cast<const float*>(weightBuffer->data.data());

        if(originalWeightPtr){
            convertDataFormatTflite(originalWeightPtr, weightData.data(), kh, kw, ci, 1);
            depthwiseConv2dParamFloat->weight = weightData;
        }
        // bias
        if (inputSize == 3) {
            const auto* biasTensor = tfliteAt(tfliteTensors, tfliteOp->inputs[2], "tensor");
            const auto* biasBuffer = biasTensor == nullptr
                                         ? nullptr
                                         : tfliteAt(tfliteModelBuffer, static_cast<int>(biasTensor->buffer), "buffer");
            const std::vector<uint8_t> emptyData;
            const auto& biasRaw = biasBuffer == nullptr ? emptyData : biasBuffer->data;
            if (biasRaw.data() != nullptr && biasRaw.size() >= sizeof(float) * ci) {
                std::vector<float> biasData(ci, 0.0f);
                ::memcpy(biasData.data(), biasRaw.data(), sizeof(float) * ci);
                depthwiseConv2dParamFloat->bias   = biasData;
            } else if (!biasRaw.empty()) {
                DLOG(ERROR) << "DEPTHWISE_CONV_2D bias buffer needs " << (sizeof(float) * ci) << " bytes, got "
                            << biasRaw.size();
                dstOp->type = MNN::OpType_MAX;
                return;
            }
        }
        depthwiseConv2dParamFloat->common = std::move(dstCommon);
        dstOp->main.value = depthwiseConv2dParamFloat.release();
    }
    
    // set input output index
    {
        auto originalWeightPtr = reinterpret_cast<const float*>(weightBuffer->data.data());
        if(originalWeightPtr){
            dstOp->inputIndexes.resize(1);
            dstOp->outputIndexes.resize(1);
            dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
            dstOp->outputIndexes[0] = tfliteOp->outputs[0];
        } else if (inputSize == 3) {
            const auto* biasTensor = tfliteAt(tfliteTensors, tfliteOp->inputs[2], "tensor");
            const auto* biasBuffer = biasTensor == nullptr
                                         ? nullptr
                                         : tfliteAt(tfliteModelBuffer, static_cast<int>(biasTensor->buffer), "buffer");
            if (biasBuffer != nullptr && biasBuffer->data.data() != nullptr) {
                dstOp->inputIndexes.resize(2);
                dstOp->outputIndexes.resize(1);
                dstOp->inputIndexes[0] = tfliteOp->inputs[0];
                dstOp->inputIndexes[1] = tfliteOp->inputs[1];
                dstOp->outputIndexes[0] = tfliteOp->outputs[0];
            } else {
                dstOp->inputIndexes.resize(inputSize);
                dstOp->outputIndexes.resize(1);
                dstOp->outputIndexes[0] = tfliteOp->outputs[0];
                for (int i = 0; i < inputSize; ++i) {
                    dstOp->inputIndexes[i] = tfliteOp->inputs[i];
                }
            }
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
