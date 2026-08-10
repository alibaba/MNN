//
//  QNNDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2025/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNDeconvolution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNDeconvolution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto conv2D = mOp->main_as_Convolution2D();
    auto common = conv2D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    int n;
    int ih, iw, ic;
    int oh, ow, oc;
    int kernelH, kernelW;
    int strideH, strideW;
    int padTop, padBottom, padLeft, padRight;
    int group;

    // compute shape
    {
        n = inputs[0]->batch();
        ih = inputs[0]->height();
        iw = inputs[0]->width();
        ic = inputs[0]->channel();
        oh = outputs[0]->height();
        ow = outputs[0]->width();
        oc = outputs[0]->channel();
        kernelH = common->kernelY();
        kernelW = common->kernelX();
        strideH = common->strideY();
        strideW = common->strideX();
        padTop = common->padY();
        padBottom = common->padY();
        padLeft = common->padX();
        padRight = common->padX();
        if (common->pads() != nullptr && common->pads()->size() >= 4) {
            padTop = common->pads()->Get(0);
            padLeft = common->pads()->Get(1);
            padBottom = common->pads()->Get(2);
            padRight = common->pads()->Get(3);
        }
        group = common->group();
    }

    // create parameters
    bool hasOutputPadding = false;
    {
        std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
        std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft,
                                               (uint32_t)padRight};
        this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void*)strideData.data());
        this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void*)padAmountData.data());
        this->createParamScalar("group", (uint32_t)group);
        // output_padding
        if (common->outPads() != nullptr && common->outPads()->size() >= 2) {
            int outPadH = common->outPads()->Get(0);
            int outPadW = common->outPads()->Get(1);
            if (outPadH > 0 || outPadW > 0) {
                hasOutputPadding = true;
                std::vector<uint32_t> outputPaddingData = {(uint32_t)outPadH, (uint32_t)outPadW};
                this->createParamTensor("output_padding", QNN_DATATYPE_UINT_32, {2}, (void*)outputPaddingData.data());
            }
        }
    }

    // create weight and bias
    {
        std::vector<float> weightData;
        const float* source = nullptr;
        int weightElementNum = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
        ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);
        // For deconv: MNN stores weight as [ic, oc/group, kH, kW]
        // QNN TransposeConv2d expects weight in HWIO format: [kH, kW, ic, oc/group]
        int ocPerGroup = oc / group;
        weightData.resize(weightElementNum);
        for (int i = 0; i < ic; i++) {
            for (int o = 0; o < ocPerGroup; o++) {
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        uint32_t srcOffset = w + kernelW * (h + kernelH * (o + ocPerGroup * i));
                        uint32_t dstOffset = o + ocPerGroup * (i + ic * (w + kernelW * h));
                        weightData[dstOffset] = source[srcOffset];
                    }
                }
            }
        }
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if (mBackend->getUseFP16()) {
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        this->createStaticFloatTensor("weight", floatDatatype,
                                      {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic, (uint32_t)ocPerGroup},
                                      weightData.data());

        // create bias
        auto bias = conv2D->bias();
        int biasElementNum = oc;
        std::vector<float> biasData(biasElementNum, 0.0f);
        if (nullptr != bias) {
            ::memcpy(biasData.data(), bias->data(), biasElementNum * sizeof(float));
        }
        this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)biasElementNum}, biasData.data());
    }

    // add TransposeConv2d node
    {
        mNodeType = "TransposeConv2d";
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
        if (hasOutputPadding) {
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // output_padding
        }

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));      // input
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
    }
    return NO_ERROR;
}

class QNNDeconvolutionCreator : public QnnBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new QNNDeconvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNDeconvolutionCreator, OpType_Deconvolution)
#endif
} // end namespace QNN
} // end namespace MNN