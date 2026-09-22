//
//  NPUDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUDeconvolution.hpp"
#include "NPUBackend.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"

using namespace std;

namespace MNN {

NPUDeconvolution::NPUDeconvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUDeconvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();
    const bool explicitHiAISession = mNpuBackend->isExplicitHiAISession();
    if (explicitHiAISession && conv2DCommon->group() != 1) {
        MNN_ERROR("HiAI V320 Deconvolution only supports group=1: %s\n", opName.c_str());
        return NOT_SUPPORT;
    }

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
    int biasSize               = 0;
    const float *biasDataPtr   = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    if (!explicitHiAISession) {
        weightSize = conv2D->weight()->size();
        filterDataPtr = conv2D->weight()->data();
        biasDataPtr = conv2D->bias()->data();
        biasSize = conv2D->bias()->size();
    } else if (inputs.size() == 3 && conv2D->weight() == nullptr) {
        const bool weightIsConst =
            TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::Usage::CONSTANT;
        const bool biasIsConst =
            TensorUtils::getDescribe(inputs[2])->usage == Tensor::InsideDescribe::Usage::CONSTANT;
        if (!weightIsConst || !biasIsConst || inputs[1]->host<float>() == nullptr ||
            inputs[2]->host<float>() == nullptr) {
            MNN_ERROR("HiAI Deconvolution requires constant weight and bias inputs: %s\n", opName.c_str());
            return NOT_SUPPORT;
        }
        filterDataPtr = inputs[1]->host<float>();
        weightSize    = inputs[1]->elementSize();
        biasDataPtr   = inputs[2]->host<float>();
        biasSize      = inputs[2]->elementSize();
    } else {
        if (conv2D->quanParameter() != nullptr) {
            quanCommon = ConvolutionCommon::load(mOp, backend(), true);
            if (quanCommon == nullptr || quanCommon->weightFloat.get() == nullptr) {
                MNN_ERROR("HiAI Deconvolution failed to decode quantized weight: %s\n", opName.c_str());
                return INPUT_DATA_ERROR;
            }
            filterDataPtr = quanCommon->weightFloat.get();
            weightSize    = quanCommon->weightFloat.size();
        } else if (conv2D->weight() != nullptr) {
            filterDataPtr = conv2D->weight()->data();
            weightSize    = conv2D->weight()->size();
        }
        if (conv2D->bias() != nullptr) {
            biasDataPtr = conv2D->bias()->data();
            biasSize    = conv2D->bias()->size();
        }
    }

    const int kernelSize = kernelX * kernelY;
    if (explicitHiAISession &&
        (filterDataPtr == nullptr || weightSize <= 0 || kernelSize <= 0 || outputCount <= 0 ||
         weightSize % (kernelSize * outputCount) != 0 || biasDataPtr == nullptr || biasSize != outputCount)) {
        MNN_ERROR("HiAI Deconvolution has invalid weight or bias data: %s\n", opName.c_str());
        return INPUT_DATA_ERROR;
    }

    int inputCount = weightSize / (kernelSize * outputCount);
    if (explicitHiAISession &&
        (inputs[0]->channel() != inputCount || outputs[0]->channel() != outputCount)) {
        MNN_ERROR("HiAI Deconvolution channel shape mismatch: %s\n", opName.c_str());
        return INPUT_DATA_ERROR;
    }
    
    shared_ptr<hiai::op::ConvTranspose> deconv(new hiai::op::ConvTranspose(opName));

    if (explicitHiAISession) {
        mOutputShape = hiai::op::Const(opName + "_shape_const");
        const std::vector<int32_t> shape = {
            outputs[0]->batch(), outputCount, outputs[0]->height(), outputs[0]->width()};
        ge::TensorDesc desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
        tensor->SetTensorDesc(desc);
        tensor->SetData(reinterpret_cast<const uint8_t*>(shape.data()),
                        shape.size() * sizeof(int32_t));
        mOutputShape.set_attr_value(tensor);
    }
    
    auto xOp = mNpuBackend->getInputOps(mOp);
    // om input weight const op
    mConst_w = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({inputCount, outputCount, kernelY, kernelX}), ge::FORMAT_NCHW,
                             ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)filterDataPtr, weightSize * sizeof(float));

        mConst_w.set_attr_value(filter);
    }

    // om input bias const op
    mConst_b = hiai::op::Const(opName + "_b_const");
    {
        ge::TensorDesc fdesc(explicitHiAISession
                                 ? ge::Shape({outputCount})
                                 : ge::Shape({1, outputCount, 1, 1}),
                             ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)biasDataPtr, biasSize * sizeof(float));

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
    if (explicitHiAISession) {
        (*deconv).set_input_output_shape(mOutputShape);
    }
    (*deconv).set_input_filter(mConst_w)
        .set_input_bias(mConst_b)
        .set_attr_strides(ge::AttrValue::LIST_INT({conv2DCommon->strideY(), conv2DCommon->strideX()}))
        .set_attr_dilations(ge::AttrValue::LIST_INT({conv2DCommon->dilateY(), conv2DCommon->dilateX()}))
        .set_attr_groups(conv2DCommon->group())
        .set_attr_pads(pads) // 上下左右
        .set_attr_pad_mode(padMode);
    if (!explicitHiAISession) {
        vector<int64_t> outputpads;
        if (conv2DCommon->outPads() != nullptr) {
            const int32_t size = conv2DCommon->outPads()->size();
            for (int32_t i = 0; i < size; ++i) {
                outputpads.push_back(static_cast<int64_t>(conv2DCommon->outPads()->data()[i]));
            }
#if defined(GRAPH_API_EXPORT)
            (*deconv).SetAttr("output_padding", ge::AttrValue::CreateFrom(outputpads));
#else
            (*deconv).SetAttr("output_padding",
                              ge::AttrValue::CreateFrom<ge::AttrValue::LIST_INT>(outputpads));
#endif
        }
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

NPUCreatorRegister<TypedCreator<NPUDeconvolution>> __deconv_op(OpType_Deconvolution);

} // namespace MNN
