//
//  QNNConvolution.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNConvolution.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNConvolution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto conv2D     = mOp->main_as_Convolution2D();
    auto common     = conv2D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    int n;
    int ih, iw, ic;
    int oh, ow, oc; 
    int kernelH, kernelW;
    int strideH, strideW;
    int padTop, padBottom, padLeft, padRight;
    int dilationH, dilationW;
    int group;

    // compute shape
{
    n = inputs[0]->batch();
    ih = inputs[0]->height(); iw = inputs[0]->width(); ic = inputs[0]->channel();
    oh = outputs[0]->height(); ow = outputs[0]->width(); oc = outputs[0]->channel();
    kernelH = common->kernelY(); kernelW = common->kernelX();
    strideH = common->strideY(); strideW = common->strideX();
    auto pads = ConvolutionCommon::convolutionPadFull(inputs[0], outputs[0], common);
    padTop = std::get<1>(pads); padBottom = std::get<3>(pads); padLeft = std::get<0>(pads); padRight = std::get<2>(pads);
    dilationH = common->dilateY(); dilationW = common->dilateX();
    group = 1;
}

    // create all tensors and params
{
    std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
    std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft, (uint32_t)padRight};
    std::vector<uint32_t> dilationData = {(uint32_t)dilationH, (uint32_t)dilationW};
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());
    this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {2}, (void *)dilationData.data());
    this->createParamScalar("group", (uint32_t)group);
    if (common->relu6()) {
        this->createParamScalar("min_value", 0.0f);
        this->createParamScalar("max_value", 6.0f);
    }

    this->createWeight(dataType, oc, ic, kernelH, kernelW, group);
    this->createBias(dataType, oc);
    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]));
    }
}

    // add nodes
{
    if (common->relu() || common->relu6()) {
        // Stage one
        {
            mNodeType = "Conv2d";
            std::string name = mNodeName + "_conv";
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
    
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
    
            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        // Stage two
        {
            mNodeType.clear();
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = common->relu6() ? "ReluMinMax" : "Relu";
            std::string name = mNodeName + "_relu";
            if (common->relu6()) {
                mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam())); // min_value
                mParams.push_back(*(mParamScalarWrappers[2]->getNativeParam())); // max_value
            }
            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

    } else {
        mNodeType = "Conv2d";
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
        mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

}
    return NO_ERROR;
}

void QNNConvolution::createWeight(Qnn_DataType_t dataType, int oc, int ic, int kernelH, int kernelW, int group) {
    std::vector<float> weightData;
    const float* source = nullptr;
    int weightElementNum = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);
    // oc ic h w ---> h w ic oc
    weightData.resize(weightElementNum);
    convertWeight(source, (float *) weightData.data(), oc, ic, kernelH, kernelW);
    this->createStaticFloatTensor("weight", dataType, {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, weightData.data());
}


void QNNConvolution::createBias(Qnn_DataType_t dataType, int oc) {
    int biasElementNum = oc;
    std::vector<float> biasData;
    biasData.resize(biasElementNum, 0);
    auto bias = mOp->main_as_Convolution2D()->bias();
    if (nullptr != bias) {
        ::memcpy((void *)biasData.data(), (void *)bias->data(), biasElementNum * sizeof(float));
    }
    this->createStaticFloatTensor("bias", dataType, {(uint32_t)oc}, biasData.data());
}

// oc ic h w ---> h w ic oc
void QNNConvolution::convertWeight(const float * src, float * dst, int oc, int ic, int kernelH, int kernelW) {
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < ic; i++) {
            for (int h = 0; h < kernelH; h++) {
                for (int w = 0; w < kernelW; w++) {
                    uint32_t srcOffset = w + kernelW * (h + kernelH * (i + ic * o));
                    uint32_t dstOffset = o + oc * (i + ic * (w + kernelW * h));
                    dst[dstOffset] = src[srcOffset];
                }
            }
        }
    }
}

class QNNConvolutionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (inputs.size() > 1) {
            return nullptr;
        }

        auto conv = op->main_as_Convolution2D()->common();
        if (conv->group() != 1) {
            return nullptr;
        }
        return new QNNConvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConvolutionCreator, OpType_Convolution)

} // end namespace QNN
} // end namespace MNN
