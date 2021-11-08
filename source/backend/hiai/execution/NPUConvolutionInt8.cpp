//
//  NPUConvolutionInt8.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConvolutionInt8.hpp"
#include "NPUBackend.hpp"
#include "../3rdParty/include/graph/op/all_ops.h"
#include <core/TensorUtils.hpp>

using namespace std;

namespace MNN {

NPUConvolutionInt8::NPUConvolutionInt8(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUConvolutionInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();
    auto quantizedParams = conv2D->symmetricQuan();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();
    int weightSize  = quantizedParams->weight()->size();
    int inputCount = weightSize / (kernelX * kernelY * outputCount);
    
    auto int32ToInt8Scale = quantizedParams->scale()->data();
    

    auto xOp = mNpuBackend->getInputOps(mOp);

    int padMode = 0; // NOTSET
    vector<int64_t> pad = {conv2DCommon->padY(), conv2DCommon->padY(), conv2DCommon->padX(), conv2DCommon->padX()};
    if (PadMode_VALID == conv2DCommon->padMode()) {
        padMode = 5;
    } else if (PadMode_SAME == conv2DCommon->padMode()) {
        padMode = 6;
        pad = {0,0,0,0};
    }

    if(outputCount > 10000){
        vector<float> filterData(weightSize, 0);
        vector<float> biasData(outputCount, 0);
        int inSize = inputCount*kernelY*kernelX;
        for(int oc = 0; oc < outputCount; oc++){
            for(int is = 0; is < inSize; is++){
                filterData[oc*inSize + is] = int32ToInt8Scale[oc] * quantizedParams->weight()->data()[oc*inSize + is];
            }
        }

        for(int oc = 0; oc < outputCount; oc++){
                biasData[oc] = int32ToInt8Scale[oc] * quantizedParams->bias()->data()[oc];
        }

        // om input weight const op
        mConst_w = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::FORMAT_NCHW,
                                ge::DT_FLOAT); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)filterData.data(), filterData.size()*sizeof(float));

            mConst_w.set_attr_value(filter);
        }
        // om input bias const op
        mConst_b = ge::op::Const(opName + "_b_const");
        {
            ge::TensorDesc fdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr bias = std::make_shared<ge::Tensor>();
            bias->SetTensorDesc(fdesc);
            bias->SetData((uint8_t *)biasData.data(), biasData.size()* sizeof(float));
            mConst_b.set_attr_value(bias);
        }
        shared_ptr<ge::op::Convolution> conv(new ge::op::Convolution(opName));

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
    }else{
        vector<float> filter_scale(int32ToInt8Scale, int32ToInt8Scale + quantizedParams->scale()->size());
        // om input weight const op
        mConst_w = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::FORMAT_NCHW,
                                ge::DT_INT8); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)quantizedParams->weight()->data(), weightSize);

            mConst_w.set_attr_value(filter);
        }
        // om input bias const op
        mConst_b = ge::op::Const(opName + "_b_const");
        {
            ge::TensorDesc fdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr bias = std::make_shared<ge::Tensor>();
            bias->SetTensorDesc(fdesc);
            bias->SetData((uint8_t *)quantizedParams->bias()->data(), quantizedParams->bias()->size()*
                                                                    sizeof(int32_t));

            mConst_b.set_attr_value(bias);
        }

        shared_ptr<ge::op::QuantizedConvolution> conv(new ge::op::QuantizedConvolution(opName));
        (*conv)
            .set_input_x(*xOp.get())
            .set_input_filter(mConst_w)
            .set_input_bias(mConst_b)
            .set_attr_kernel(ge::AttrValue::LIST_INT({kernelY, kernelX}))
            .set_attr_stride(ge::AttrValue::LIST_INT({conv2DCommon->strideY(), conv2DCommon->strideX()}))
            .set_attr_dilation(ge::AttrValue::LIST_INT({conv2DCommon->dilateY(), conv2DCommon->dilateX()}))
            .set_attr_group(conv2DCommon->group())
            .set_attr_pad(ge::AttrValue::LIST_INT(pad)) // 上下左右
            .set_attr_pad_mode(padMode)
            .set_attr_filter_quant_type(1)
            .set_attr_x_quant_type(1)
            .set_attr_x_quant_offset(127)
            // .set_attr_x_quant_offset(0)
            .set_attr_x_quant_scale(1.0)
            .set_attr_filter_quant_scales(filter_scale)
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
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvolutionInt8>> __conv_int8_op(OpType_ConvInt8);

} // namespace MNN
