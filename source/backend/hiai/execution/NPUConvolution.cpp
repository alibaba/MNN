//
//  NPUConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConvolution.hpp"
#include "NPUBackend.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"

using namespace std;

namespace MNN {

NPUConvolution::NPUConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();

    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2D->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2D->quanParameter(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", mOp->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (nullptr == filterDataPtr) {
        weightSize    = conv2D->weight()->size();
        filterDataPtr = conv2D->weight()->data();
    }

    int inputCount = weightSize / (kernelX * kernelY * outputCount);
    
    shared_ptr<ge::op::Convolution> conv(new ge::op::Convolution(opName));
    
    auto xOp = mNpuBackend->getInputOps(mOp);

    // om input weight const op
    mConst_w = ge::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::FORMAT_NCHW,
                             ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)filterDataPtr, weightSize * sizeof(float));

        mConst_w.set_attr_value(filter);
    }

    // om input bias const op
    mConst_b = ge::op::Const(opName + "_b_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));

        mConst_b.set_attr_value(filter);
    }

    int padMode = 0; // NOTSET
    if (PadMode_VALID == conv2DCommon->padMode()) {
        padMode = 5;
    } else if (PadMode_SAME == conv2DCommon->padMode()) {
        padMode = 6;
    }

    (*conv)
        .set_input_x(*xOp.get())
        .set_input_w(mConst_w)
        .set_input_b(mConst_b)
        .set_attr_kernel(ge::AttrValue::LIST_INT({kernelY, kernelX}))
        .set_attr_mode(1) // 后续会废除，不再使用
        .set_attr_stride(ge::AttrValue::LIST_INT({conv2DCommon->strideY(), conv2DCommon->strideX()}))
        .set_attr_dilation(ge::AttrValue::LIST_INT({conv2DCommon->dilateY(), conv2DCommon->dilateX()}))
        .set_attr_group(conv2DCommon->group())
        .set_attr_pad(ge::AttrValue::LIST_INT(
            {conv2DCommon->padY(), conv2DCommon->padY(), conv2DCommon->padX(), conv2DCommon->padX()})) // 上下左右
        .set_attr_pad_mode(padMode)
        .set_attr_num_output(outputCount);

    shared_ptr<ge::op::Activation> relu_conv(new ge::op::Activation(opName + "_Relu"));
    mRelu_conv = relu_conv;

    auto relu  = conv2DCommon->relu();
    auto relu6 = conv2DCommon->relu6();
    if (relu || relu6) {
        (*mRelu_conv)
            .set_input_x(*conv.get())
            .set_attr_mode(relu?1:14);
    }

    if (relu || relu6) {
        mNpuBackend->setOutputOps(mOp, {conv, mRelu_conv}, outputs);
    }else{
        mNpuBackend->setOutputOps(mOp, {conv}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvolution>> __conv_op(OpType_Convolution);

} // namespace MNN
