//
//  TRTDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTDeconvolution.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"
#include "plugin/PreluPlugin.hpp"

using namespace std;

namespace MNN {

TRTDeconvolution::TRTDeconvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTDeconvolution::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTDeconvolution in\n");
#endif
    auto opName       = mOp->name()->str();
    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX        = conv2DCommon->kernelX();
    auto kernelY        = conv2DCommon->kernelY();
    auto outputCount    = conv2DCommon->outputCount();
    const float *source = nullptr;
    int weightSize      = 0;

    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend(), conv2D, &source, &weightSize);

    nvinfer1::DimsHW NVKSize(kernelY, kernelX);
    nvinfer1::DimsHW NVKSSize(conv2DCommon->strideY(), conv2DCommon->strideX());

    TRTWeight weight{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(source)),
                     static_cast<size_t>(weightSize)};

    TRTWeight bias{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(conv2D->bias()->data())),
                   static_cast<size_t>(conv2D->bias()->size())};
    auto conv_layer =
        mTrtBackend->getNetwork()->addDeconvolution(*xOp[0], outputCount, NVKSize, weight.get(), bias.get());

    MNN_ASSERT(conv_layer != nullptr);
    conv_layer->setStride(NVKSSize);
    conv_layer->setNbGroups(1);
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

TRTCreatorRegister<TypedCreator<TRTDeconvolution>> __de_conv_op(OpType_Deconvolution);

} // namespace MNN
