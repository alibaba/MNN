//
//  TRTDepthwiseConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTDepthwiseConvolution.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "core/ConvolutionCommon.hpp"

using namespace std;

namespace MNN {

TRTDepthwiseConvolution::TRTDepthwiseConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                                                 const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTDepthwiseConvolution::onEncode(const std::vector<ITensor *> &xOp) {
    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();
    nvinfer1::DimsHW NVKSize(kernelY, kernelX);
    nvinfer1::DimsHW NVKDSize(conv2DCommon->dilateY(), conv2DCommon->dilateX());
    nvinfer1::DimsHW NVKSSize(conv2DCommon->strideY(), conv2DCommon->strideX());
    const float *source = nullptr;
    int weightSize      = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    if (nullptr != mOp->main_as_Convolution2D()->quanParameter()) {
        quanWeight = ConvolutionCommon::load(mOp->main_as_Convolution2D(), backend(), true);
        source     = quanWeight->weightFloat.get();
        weightSize = quanWeight->weightFloat.size();
    } else {
        if (nullptr != conv2D->weight()) {
            source     = conv2D->weight()->data();
            weightSize = conv2D->weight()->size();
        }
    }
    mTrtBackend->pushCache(quanWeight);
    TRTWeight weight{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(source)),
                     static_cast<size_t>(weightSize)};

    TRTWeight bias{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(conv2D->bias()->data())),
                   static_cast<size_t>(conv2D->bias()->size())};
    auto conv_layer =
        mTrtBackend->getNetwork()->addConvolution(*(xOp[0]), outputCount, NVKSize, weight.get(), bias.get());

    conv_layer->setStride(NVKSSize);
    conv_layer->setDilation(NVKDSize);
    conv_layer->setNbGroups(outputCount);
    auto pads = ConvolutionCommon::convolutionPad(mInputs[0], mOutputs[0], conv2DCommon);
    conv_layer->setPadding(nvinfer1::DimsHW{pads.second, pads.first});
    if (conv2DCommon->padMode() == PadMode_SAME) {
        conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }    
    conv_layer->setName(mOp->name()->str().c_str());
    auto relu  = conv2DCommon->relu();
    auto relu6 = conv2DCommon->relu6();

    if (relu) {
        mActivationLayer = mTrtBackend->getNetwork()->addActivation(*conv_layer->getOutput(0), ActivationType::kRELU);
    }

    if (relu6) {
        mActivationLayer = mTrtBackend->getNetwork()->addActivation(*conv_layer->getOutput(0), ActivationType::kCLIP);
        mActivationLayer->setAlpha(0.);
        mActivationLayer->setBeta(6.);
    }

    if (relu || relu6) {
        return {mActivationLayer->getOutput(0)};
    }
    return {conv_layer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTDepthwiseConvolution>> __dw_conv_op(OpType_ConvolutionDepthwise);

} // namespace MNN
