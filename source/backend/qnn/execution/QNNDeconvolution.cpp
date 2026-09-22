//
//  QNNDeconvolution.cpp
//  MNN
//

#include "QNNDeconvolution.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "QnnOpDef.h"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNDeconvolution::onEncodeLegacy(
    const std::vector<Tensor *> &inputs,
    const std::vector<Tensor *> &outputs) {
    const auto *conv = mOp->main_as_Convolution2D();
    const auto *common = conv->common();
    const int inputChannels = inputs[0]->channel();
    const int outputChannels = outputs[0]->channel();
    const int kernelH = common->kernelY();
    const int kernelW = common->kernelX();
    const int group = common->group();

    int padTop = common->padY();
    int padBottom = common->padY();
    int padLeft = common->padX();
    int padRight = common->padX();
    if (common->pads() != nullptr && common->pads()->size() >= 4) {
        padTop = common->pads()->Get(0);
        padLeft = common->pads()->Get(1);
        padBottom = common->pads()->Get(2);
        padRight = common->pads()->Get(3);
    }

    std::vector<uint32_t> stride = {
        (uint32_t)common->strideY(), (uint32_t)common->strideX()};
    std::vector<uint32_t> padAmount = {
        (uint32_t)padTop, (uint32_t)padBottom,
        (uint32_t)padLeft, (uint32_t)padRight};
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2},
                            stride.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2},
                            padAmount.data());
    this->createParamScalar("group", (uint32_t)group);

    bool hasOutputPadding = false;
    if (common->outPads() != nullptr && common->outPads()->size() >= 2) {
        const int outputPadH = common->outPads()->Get(0);
        const int outputPadW = common->outPads()->Get(1);
        if (outputPadH > 0 || outputPadW > 0) {
            hasOutputPadding = true;
            std::vector<uint32_t> outputPadding = {
                (uint32_t)outputPadH, (uint32_t)outputPadW};
            this->createParamTensor("output_padding", QNN_DATATYPE_UINT_32,
                                    {2}, outputPadding.data());
        }
    }

    const float *sourceWeight = nullptr;
    int weightElements = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quantWeight;
    ConvolutionCommon::getConvParameters(&quantWeight, mBackend, mOp,
                                         &sourceWeight, &weightElements);
    const int outputChannelsPerGroup = outputChannels / group;
    std::vector<float> weight(weightElements);
    for (int i = 0; i < inputChannels; ++i) {
        for (int o = 0; o < outputChannelsPerGroup; ++o) {
            for (int h = 0; h < kernelH; ++h) {
                for (int w = 0; w < kernelW; ++w) {
                    const uint32_t src =
                        w + kernelW *
                                (h + kernelH *
                                         (o + outputChannelsPerGroup * i));
                    const uint32_t dst =
                        o + outputChannelsPerGroup *
                                (i + inputChannels * (w + kernelW * h));
                    weight[dst] = sourceWeight[src];
                }
            }
        }
    }
    const Qnn_DataType_t floatType = mBackend->getUseFP16()
                                         ? QNN_DATATYPE_FLOAT_16
                                         : QNN_DATATYPE_FLOAT_32;
    this->createStaticFloatTensor(
        "weight", floatType,
        {(uint32_t)kernelH, (uint32_t)kernelW,
         (uint32_t)inputChannels, (uint32_t)outputChannelsPerGroup},
        weight.data());

    std::vector<float> bias(outputChannels, 0.0f);
    if (conv->bias() != nullptr) {
        ::memcpy(bias.data(), conv->bias()->data(),
                 outputChannels * sizeof(float));
    }
    this->createStaticFloatTensor("bias", floatType,
                                  {(uint32_t)outputChannels}, bias.data());

    mNodeType = "TransposeConv2d";
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
    mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam()));
    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));
    if (hasOutputPadding) {
        mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam()));
    }
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                             mPackageName.c_str(), mNodeType.c_str(),
                             mParams, mInputs, mOutputs);
    return NO_ERROR;
}

ErrorCode QNNDeconvolution::onEncode(const std::vector<Tensor *> &inputs,
                                     const std::vector<Tensor *> &outputs) {
    if (!mBackend->isDedicatedQnnSession()) {
        return onEncodeLegacy(inputs, outputs);
    }
    if (inputs.size() != 1 || outputs.size() != 1) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    const auto *conv = mOp->main_as_Convolution2D();
    const auto *common = conv->common();
    if (common->group() != 1 || common->dilateX() != 1 || common->dilateY() != 1) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    const int inputChannels = inputs[0]->channel();
    const int outputChannels = outputs[0]->channel();
    const int kernelH = common->kernelY();
    const int kernelW = common->kernelX();
    const int strideH = common->strideY();
    const int strideW = common->strideX();

    const Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    // HTP and legacy DSP both use this path when the signed runtime metadata
    // requires an end-to-end quantized graph. Restricting it to DSP would
    // reject valid HTP S8 tensors before backend validation.
    const bool fixedPointGraph =
        mBackend->requiresQuantizedGraph() &&
        dataType == QNN_DATATYPE_SFIXED_POINT_8;
    if (!fixedPointGraph && dataType != QNN_DATATYPE_FLOAT_16 &&
        dataType != QNN_DATATYPE_FLOAT_32) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    int outputPadH = 0;
    int outputPadW = 0;
    if (common->outPads() != nullptr && common->outPads()->size() >= 2) {
        outputPadH = common->outPads()->data()[0];
        outputPadW = common->outPads()->data()[1];
    }

    const auto leadingPad = ConvolutionCommon::convolutionTransposePad(inputs[0], outputs[0], common);
    const int padLeft = leadingPad.first;
    const int padTop = leadingPad.second;
    const int totalPadH = (inputs[0]->height() - 1) * strideH + kernelH + outputPadH - outputs[0]->height();
    const int totalPadW = (inputs[0]->width() - 1) * strideW + kernelW + outputPadW - outputs[0]->width();
    const int padBottom = totalPadH - padTop;
    const int padRight = totalPadW - padLeft;
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0 ||
        outputPadH < 0 || outputPadW < 0) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    std::vector<uint32_t> stride = {(uint32_t)strideH, (uint32_t)strideW};
    std::vector<uint32_t> padAmount = {(uint32_t)padTop, (uint32_t)padBottom,
                                       (uint32_t)padLeft, (uint32_t)padRight};
    std::vector<uint32_t> outputPadding = {(uint32_t)outputPadH, (uint32_t)outputPadW};
    this->createParamTensor(QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING,
                            QNN_DATATYPE_UINT_32, {2}, outputPadding.data());
    this->createParamTensor(QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT,
                            QNN_DATATYPE_UINT_32, {2, 2}, padAmount.data());
    this->createParamTensor(QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE,
                            QNN_DATATYPE_UINT_32, {2}, stride.data());
    this->createParamScalar(QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP, (uint32_t)1);

    const float *sourceWeight = nullptr;
    int weightElements = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quantWeight;
    ConvolutionCommon::getConvParameters(&quantWeight, mBackend, mOp,
                                         &sourceWeight, &weightElements);
    const int expectedWeightElements = inputChannels * outputChannels * kernelH * kernelW;
    if (sourceWeight == nullptr || weightElements != expectedWeightElements) {
        MNN_ERROR("MNN_QNN: invalid deconvolution weight: got %d, expected %d.\n",
                  weightElements, expectedWeightElements);
        return INPUT_DATA_ERROR;
    }
    // MNN deconvolution weights are IOHW; QNN TransposeConv2d expects HWIO.
    std::vector<float> weight(expectedWeightElements);
    for (int i = 0; i < inputChannels; ++i) {
        for (int o = 0; o < outputChannels; ++o) {
            for (int h = 0; h < kernelH; ++h) {
                for (int w = 0; w < kernelW; ++w) {
                    const int src = w + kernelW * (h + kernelH * (o + outputChannels * i));
                    const int dst = o + outputChannels * (i + inputChannels * (w + kernelW * h));
                    weight[dst] = sourceWeight[src];
                }
            }
        }
    }

    std::vector<float> bias(outputChannels, 0.0f);
    if (conv->bias() != nullptr) {
        const int count = std::min(outputChannels, (int)conv->bias()->size());
        std::memcpy(bias.data(), conv->bias()->data(), count * sizeof(float));
    }

    if (fixedPointGraph) {
        // Match the HTP INT8 representation on legacy DSP: the QNN HWIO
        // filter's output-channel axis is 3, so each output channel retains
        // its own SFixed8 scale. The SFixed32 bias uses the corresponding
        // inputScale * weightScale[channel] domain.
        mWeightScaleOffsets.assign(outputChannels, Qnn_ScaleOffset_t{});
        for (int channel = 0; channel < outputChannels; ++channel) {
            float maxAbsWeight = 0.0f;
            for (int h = 0; h < kernelH; ++h) {
                for (int w = 0; w < kernelW; ++w) {
                    for (int i = 0; i < inputChannels; ++i) {
                        const size_t index =
                            channel + outputChannels *
                                (i + inputChannels * (w + kernelW * h));
                        maxAbsWeight =
                            std::max(maxAbsWeight, std::abs(weight[index]));
                    }
                }
            }
            mWeightScaleOffsets[channel].scale =
                std::max(maxAbsWeight / 127.0f, 1.0e-12f);
            mWeightScaleOffsets[channel].offset = 0;
        }
        std::vector<int8_t> quantWeight(weight.size());
        for (int h = 0; h < kernelH; ++h) {
            for (int w = 0; w < kernelW; ++w) {
                for (int i = 0; i < inputChannels; ++i) {
                    for (int channel = 0; channel < outputChannels;
                         ++channel) {
                        const size_t index =
                            channel + outputChannels *
                                (i + inputChannels * (w + kernelW * h));
                        const int quantized = static_cast<int>(std::round(
                            weight[index] /
                            mWeightScaleOffsets[channel].scale));
                        quantWeight[index] = static_cast<int8_t>(
                            std::max(-127, std::min(127, quantized)));
                    }
                }
            }
        }

        Qnn_QuantizeParams_t weightQuantize = DEFAULT_QUANTIZE_PARAMS;
        weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
        weightQuantize.quantizationEncoding =
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        Qnn_AxisScaleOffset_t weightAxisScaleOffset = {};
        weightAxisScaleOffset.axis = 3;
        weightAxisScaleOffset.numScaleOffsets = outputChannels;
        weightAxisScaleOffset.scaleOffset = mWeightScaleOffsets.data();
        weightQuantize.axisScaleOffsetEncoding = weightAxisScaleOffset;
        this->createStaticTensor(
            "weight", QNN_DATATYPE_SFIXED_POINT_8,
            {(uint32_t)kernelH, (uint32_t)kernelW,
             (uint32_t)inputChannels, (uint32_t)outputChannels},
            quantWeight.data(), weightQuantize);

        const auto inputQuant =
            mBackend->getNativeTensor(inputs[0])->v1.quantizeParams;
        mBiasScaleOffsets.assign(outputChannels, Qnn_ScaleOffset_t{});
        std::vector<int32_t> quantBias(outputChannels, 0);
        for (int channel = 0; channel < outputChannels; ++channel) {
            const float biasScale = std::max(
                inputQuant.scaleOffsetEncoding.scale *
                    mWeightScaleOffsets[channel].scale,
                1.0e-20f);
            mBiasScaleOffsets[channel].scale = biasScale;
            mBiasScaleOffsets[channel].offset = 0;
            const double quantized = std::round(
                static_cast<double>(bias[channel]) /
                static_cast<double>(biasScale));
            quantBias[channel] = static_cast<int32_t>(
                std::max(
                    static_cast<double>(std::numeric_limits<int32_t>::min()),
                    std::min(
                        static_cast<double>(
                            std::numeric_limits<int32_t>::max()),
                        quantized)));
        }
        Qnn_QuantizeParams_t biasQuantize = DEFAULT_QUANTIZE_PARAMS;
        biasQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
        biasQuantize.quantizationEncoding =
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        Qnn_AxisScaleOffset_t biasAxisScaleOffset = {};
        biasAxisScaleOffset.axis = 0;
        biasAxisScaleOffset.numScaleOffsets = outputChannels;
        biasAxisScaleOffset.scaleOffset = mBiasScaleOffsets.data();
        biasQuantize.axisScaleOffsetEncoding = biasAxisScaleOffset;
        this->createStaticTensor(
            "bias", QNN_DATATYPE_SFIXED_POINT_32,
            {(uint32_t)outputChannels}, quantBias.data(), biasQuantize);

        MNN_PRINT(
            "MNN_QNN_TRANSPOSE_CONV_QUANT_AUDIT: node=%s backend=%s "
            "weight=AXIS_SCALE_OFFSET axis=3 scales=%d "
            "bias=AXIS_SCALE_OFFSET axis=0\n",
            mNodeName.c_str(),
            mBackend->isDspBackend() ? "DSP_V66" : "HTP",
            outputChannels);
    } else {
        const Qnn_DataType_t staticDataType = mBackend->getUseFP16()
                                                  ? QNN_DATATYPE_FLOAT_16
                                                  : QNN_DATATYPE_FLOAT_32;
        this->createStaticFloatTensor(
            "weight", staticDataType,
            {(uint32_t)kernelH, (uint32_t)kernelW,
             (uint32_t)inputChannels, (uint32_t)outputChannels},
            weight.data());
        this->createStaticFloatTensor(
            "bias", staticDataType, {(uint32_t)outputChannels}, bias.data());
    }

    for (const auto &param : mParamTensorWrappers) {
        mParams.push_back(*(param->getNativeParam()));
    }
    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));

    mNodeType = QNN_OP_TRANSPOSE_CONV_2D;
    if (common->relu() || common->relu6()) {
        this->createStageTensor("activation_input", dataType, getNHWCShape(outputs[0]), outputs[0]);
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_deconv").c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = common->relu6() ? "ReluMinMax" : "Relu";
        if (common->relu6()) {
            const size_t activationParamStart = mParamScalarWrappers.size();
            this->createParamScalar("min_value", 0.0f);
            this->createParamScalar("max_value", 6.0f);
            for (size_t i = activationParamStart;
                 i < mParamScalarWrappers.size(); ++i) {
                mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
            }
        }
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_activation").c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
    } else {
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNDeconvolutionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution *onCreate(const std::vector<Tensor *> &inputs,
                                         const std::vector<Tensor *> &outputs,
                                         const MNN::Op *op,
                                         Backend *backend) const override {
        return new QNNDeconvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNDeconvolutionCreator, OpType_Deconvolution)
#endif
} // end namespace QNN
} // end namespace MNN
