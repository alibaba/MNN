//
//  QNNConvDepthwise.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNConvDepthwise.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNConvDepthwise::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
}

    // create all tensors and params
{
    std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
    std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft, (uint32_t)padRight};
    std::vector<uint32_t> dilationData = {(uint32_t)dilationH, (uint32_t)dilationW};
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());
    this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {2}, (void *)dilationData.data());
    if (common->relu6()) {
        this->createParamScalar("min_value", 0.0f);
        this->createParamScalar("max_value", 6.0f);
    }

    this->createWeight(dataType, oc, kernelH, kernelW);
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
                mNodeType = "DepthWiseConv2d";
                std::string name = mNodeName + "_convDepthwise";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation

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
                    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // min_value
                    mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam())); // max_value
                }
                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
                mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
    
        } else {
            mNodeType = "DepthWiseConv2d";
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
    
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }

    return NO_ERROR;
}



void QNNConvDepthwise::createWeight(Qnn_DataType_t dataType, int oc, int kernelH, int kernelW) {
    std::vector<float> weightData;
    const float* source = nullptr;
    int weightElementNum = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);
    // oc ic h w ---> h w ic oc
    weightData.resize(weightElementNum);
    convertWeight(source, (float *) weightData.data(), oc, kernelH, kernelW);
    this->createStaticFloatTensor("weight", dataType, {(uint32_t)kernelH, (uint32_t)kernelW, 1, (uint32_t)oc}, weightData.data());
}


void QNNConvDepthwise::createBias(Qnn_DataType_t dataType, int oc) {
    int biasElementNum = oc;
    std::vector<float> biasData;
    biasData.resize(biasElementNum, 0);
    auto bias = mOp->main_as_Convolution2D()->bias();
    if (nullptr != bias) {
        ::memcpy((void *)biasData.data(), (void *)bias->data(), biasElementNum * sizeof(float));
    }
    this->createStaticFloatTensor("bias", dataType, {(uint32_t)oc}, biasData.data());
}


// oc, h, w ---> h, w, oc
void QNNConvDepthwise::convertWeight(const float * src, float * dst, int oc, int kernelH, int kernelW) {
    for (int c = 0; c < oc; c++) {
        for (int h = 0; h < kernelH; h++) {
            for (int w = 0; w < kernelW; w++) {
                int srcOffset = w + kernelW * (h + kernelH * c);
                int dstOffset = c + oc * (w + kernelW * h);
                dst[dstOffset] = src[srcOffset];
            }
        }
    }
} 

class QNNConvDepthwiseCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNConvDepthwise(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConvDepthwiseCreator, OpType_ConvolutionDepthwise)

} // end namespace QNN
} // end namespace MNN
