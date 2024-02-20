//
//  NPUDeconvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUDeconvolutionDepthwise.hpp"
#include "NPUBackend.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"

using namespace std;

namespace MNN {

NPUDeconvolutionDepthwise::NPUDeconvolutionDepthwise(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUDeconvolutionDepthwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();

    std::vector<int64_t> pads;
    if (conv2DCommon->pads() != nullptr) {
        int32_t size = conv2DCommon->pads()->size() / 2;
        for (int32_t i = 0; i < size; i++) {
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i]));
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i+size]));
        }
    } else {
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
    }
    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    if (nullptr == filterDataPtr) {
        weightSize    = conv2D->weight()->size();
        filterDataPtr = conv2D->weight()->data();
    }

    int inputCount = weightSize / (kernelX * kernelY * outputCount);
    
    shared_ptr<hiai::op::ConvTranspose> deconv(new hiai::op::ConvTranspose(opName));
    
    auto xOp = mNpuBackend->getInputOps(mOp);
    // om input weight const op
    mConst_w = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::FORMAT_NCHW,
                             ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)filterDataPtr, weightSize * sizeof(float));

        mConst_w.set_attr_value(filter);
    }

    // om input bias const op
    mConst_b = hiai::op::Const(opName + "_b_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));

        mConst_b.set_attr_value(filter);
    }

    std::string padMode = "SPECIFIC"; // NOTSET
    if (PadMode_VALID == conv2DCommon->padMode()) {
        padMode = "VALID";
    } else if (PadMode_SAME == conv2DCommon->padMode()) {
        padMode = "SAME";
    }
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex];
    xOp = iops.back().first;
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*deconv).set_input_x(*xOp.get());
    } else {
        (*deconv).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    (*deconv)
        .set_input_filter(mConst_w)
        .set_input_bias(mConst_b)
        .set_attr_strides(ge::AttrValue::LIST_INT({conv2DCommon->strideY(), conv2DCommon->strideX()}))
        .set_attr_dilations(ge::AttrValue::LIST_INT({conv2DCommon->dilateY(), conv2DCommon->dilateX()}))
        .set_attr_groups(conv2DCommon->group())
        .set_attr_pads(pads) // 上下左右
        .set_attr_pad_mode(padMode);

    vector<int64_t> outputpads;
    if (conv2DCommon->outPads() != nullptr) {
        int32_t size = conv2DCommon->outPads()->size();
        for (int32_t i = 0; i < size; i++) {
            outputpads.push_back(static_cast<int64_t>(conv2DCommon->outPads()->data()[i]));
        }
        (*deconv).SetAttr("output_padding", ge::AttrValue::CreateFrom(outputpads));
    }

    shared_ptr<hiai::op::Activation> relu_conv(new hiai::op::Activation(opName + "_Relu"));
    mRelu_conv = relu_conv;

    auto relu  = conv2DCommon->relu();
    auto relu6 = conv2DCommon->relu6();
    if (relu || relu6) {
        (*mRelu_conv)
            .set_input_x(*deconv.get())
            .set_attr_mode(relu?1:14);
    }

    if (relu || relu6) {
        mNpuBackend->setOutputOps(mOp, {deconv, mRelu_conv}, outputs);
    }else{
        mNpuBackend->setOutputOps(mOp, {deconv}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUDeconvolutionDepthwise>> __depthwise_deconv_op(OpType_DeconvolutionDepthwise);

} // namespace MNN
