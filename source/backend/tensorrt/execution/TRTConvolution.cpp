//
//  TRTConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTConvolution.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"
#include "plugin/PreluPlugin.hpp"
using namespace std;

namespace MNN {

TRTConvolution::TRTConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTConvolution::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTConvolution in\n");
#endif
    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX        = conv2DCommon->kernelX();
    auto kernelY        = conv2DCommon->kernelY();
    auto outputCount    = conv2DCommon->outputCount();
    int srcCount        = 0;
    const float *source = nullptr;
    int weightSize      = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    if (nullptr != mOp->main_as_Convolution2D()->quanParameter()) {
        quanWeight = ConvolutionCommon::load(mOp->main_as_Convolution2D(), backend(), true);
        srcCount   = quanWeight->weightFloat.size() / (outputCount * kernelX * kernelY);
        source     = quanWeight->weightFloat.get();
        weightSize = quanWeight->weightFloat.size();
    } else {
        if (nullptr != conv2D->weight()) {
            srcCount   = conv2D->weight()->size() / (outputCount * kernelX * kernelY);
            source     = conv2D->weight()->data();
            weightSize = conv2D->weight()->size();
        } else {
            srcCount = conv2D->common()->inputCount();
        }
    }
    int inputCount = srcCount;
    mTrtBackend->pushCache(quanWeight);

    nvinfer1::DimsHW NVKSize(kernelY, kernelX);
    nvinfer1::DimsHW NVKDSize(conv2DCommon->dilateY(), conv2DCommon->dilateX());
    nvinfer1::DimsHW NVKSSize(conv2DCommon->strideY(), conv2DCommon->strideX());

    TRTWeight weight{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(source)),
                     static_cast<size_t>(weightSize)};

    TRTWeight bias{nvinfer1::DataType::kFLOAT, static_cast<void *>(const_cast<float *>(conv2D->bias()->data())),
                   static_cast<size_t>(conv2D->bias()->size())};
    ITensor* input = xOp[0];
    auto originDim = xOp[0]->getDimensions();
    auto dims = originDim.nbDims;
    if (dims < 4) {
        auto shuffle =  mTrtBackend->getNetwork()->addShuffle(*(xOp[0]));
        auto dimReshape = originDim;
        dimReshape.nbDims = 4;
        for (int v=dims; v<4; ++v) {
            dimReshape.d[v] = 1;
        }
        shuffle->setReshapeDimensions(dimReshape);
        input = shuffle->getOutput(0);
    }
    auto conv_layer =
        mTrtBackend->getNetwork()->addConvolution(*input, outputCount, NVKSize, weight.get(), bias.get());

    MNN_ASSERT(conv_layer != nullptr);
    conv_layer->setStride(NVKSSize);
    conv_layer->setDilation(NVKDSize);
    conv_layer->setNbGroups(1);
    auto pads = ConvolutionCommon::convolutionPad(mInputs[0], mOutputs[0], conv2DCommon);
    conv_layer->setPadding(nvinfer1::DimsHW{pads.second, pads.first});

    if (conv2DCommon->padMode() == PadMode_SAME) {
        conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
    if (mOp->name()) {
        conv_layer->setName(mOp->name()->str().c_str());
    }
    auto output = conv_layer->getOutput(0);
    if (dims < 4) {
        auto dimReshape = originDim;
        dimReshape.d[1] = outputCount;
        dimReshape.d[2] = mOutputs[0]->length(2);
        auto shuffle =  mTrtBackend->getNetwork()->addShuffle(*output);
        shuffle->setReshapeDimensions(dimReshape);
        output = shuffle->getOutput(0);
    }
    auto relu  = conv2DCommon->relu();
    auto relu6 = conv2DCommon->relu6();

    if (relu) {
        mActivationLayer = mTrtBackend->getNetwork()->addActivation(*output, ActivationType::kRELU);
    }

    if (relu6) {
        mActivationLayer = mTrtBackend->getNetwork()->addActivation(*output, ActivationType::kCLIP);
        mActivationLayer->setAlpha(0.);
        mActivationLayer->setBeta(6.);
    }

    if (relu || relu6) {
        return {mActivationLayer->getOutput(0)};
    }
    return {output};
}

TRTCreatorRegister<TypedCreator<TRTConvolution>> __conv_op(OpType_Convolution);

} // namespace MNN
