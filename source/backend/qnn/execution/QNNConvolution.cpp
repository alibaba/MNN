//
//  QNNConvolution.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNConvolution.hpp"
#include <cmath>

namespace MNN {
namespace QNN {
static std::pair<int, int> closest_factors(int n) {
    int a = static_cast<int>(std::sqrt(n));
    for (; a >= 1; --a) {
        if (n % a == 0) {
            int b = n / a;
            return {a, b};
        }
    }
    return {1, n};  
}
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
        group = common->group();
    }

    const float * weightSource = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (mOp->main_as_Convolution2D()->quanParameter()) {
        bool forceFloat = (common->kernelX() == 1 && common->kernelY() == 1) ? false : true;

        quanCommon = ConvolutionCommon::load(mOp, this->backend(), forceFloat);
        if (quanCommon->weightFloat.get() == nullptr) {
            // [TODO] Support asymmetric and other quantBits.
            // isQuantWeight && symmetric quantization && int8 quantization && 1x1 conv
            if (quanCommon->asymmetric || quanCommon->canUseInt4) {
                return NOT_SUPPORT;
            }
            return this->onEncodeQuant(inputs[0], outputs[0], n, ih, iw, ic, oc, quanCommon);
        } else {
            weightSource = quanCommon->weightFloat.get();
        }
    } else {
        int weightElementNum;
        ConvolutionCommon::getConvParameters(&quanCommon, mBackend, mOp, &weightSource, &weightElementNum);
    }

    #ifdef QNN_VERBOSE
    MNN_PRINT("n:%d, ih:%d, iw:%d, ic:%d, oh:%d, ow:%d, oc:%d, kernelH:%d, kernelW:%d, dilationH:%d, dilationW:%d, strideH:%d, strideW:%d, group:%d, pad:%d %d %d %d\n", n, ih, iw, ic, oh, ow, oc, kernelH, kernelW, dilationH, \
        dilationW, strideH, strideW, group, padTop, padBottom, padLeft, padRight);
    #endif
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

        this->createWeight(dataType, oc, ic, kernelH, kernelW, group, weightSource);
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
            bool isMatmul = ih==1 && iw==1 && oh==1 && ow==1 && kernelH==1 && kernelW==1 && strideH==1 && \
                strideW==1 && dilationH==1 && dilationW==1 && group==1 && \
                padTop==0 && padBottom==0 && padLeft==0 && padRight==0;
            if(isMatmul && n > 1) {
                auto num = closest_factors(n);
                this->createStageTensor("InputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, ic}));
                this->createStageTensor("OutputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, oc}));
                #ifdef QNN_VERBOSE
                MNN_PRINT("Matmul2Conv, start reshape batch:%d -> %dx%d\n", n, num.first, num.second);
                #endif
                // reshape input
                {
                    std::string name = mNodeName + "_input_reshape";
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    mNodeType = "Reshape";

                    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
                    mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
                // conv2d
                {
                    std::string name = mNodeName;
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    mNodeType = "Conv2d";

                    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                    mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                    mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
                    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group

                    mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // input0
                    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

                    mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }

                // reshape output
                {
                    std::string name = mNodeName + "_output_reshape";
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    mNodeType = "Reshape";

                    mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
                    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // input0
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
                return NO_ERROR;
            }


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


ErrorCode QNNConvolution::onEncodeQuant(Tensor * input, Tensor * output, int n, int h, int w, int ic, int oc, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) {
    // create parameters and stage tensors
    {
        bool transposeWeightFlag = true;
        this->createParamScalar("transpose_in1", transposeWeightFlag);

        std::vector<uint32_t> tempInputShape = {(uint32_t) n * h * w , (uint32_t) ic};
        std::vector<uint32_t> tempOutputShape = {(uint32_t) n * h * w , (uint32_t) oc};
        this->createStageTensor("tempInput", QNN_DATATYPE_FLOAT_16, tempInputShape);
        this->createStageTensor("tempOutput", QNN_DATATYPE_FLOAT_16, tempOutputShape);

        // create weight
        const int8_t * source = quanCommon->weight.get();
        std::vector<int8_t> quantWeightData(oc * ic, 0);
        ::memcpy(quantWeightData.data(), source, oc * ic * sizeof(int8_t));

        float * dequantAlpha = quanCommon->alpha.get();

        Qnn_QuantizeParams_t weightQuantize{};
        weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
        weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        Qnn_AxisScaleOffset_t weightAxisScaleOffsetEncoding{};
        weightAxisScaleOffsetEncoding.axis = 0;
        weightAxisScaleOffsetEncoding.numScaleOffsets = oc;
        std::vector<Qnn_ScaleOffset_t> scaleOffsetData(oc);
        for (int i = 0; i < oc; i++) {
            if (quanCommon->asymmetric) {
                scaleOffsetData[i].scale = dequantAlpha[2 * i + 1];
                // scaleOffsetData[i].offset = (int) dequantAlpha[2 * i + 0];
                scaleOffsetData[i].offset = 0;
            } else {
                scaleOffsetData[i].scale = dequantAlpha[i];
                scaleOffsetData[i].offset = 0;
            }
        }
        weightAxisScaleOffsetEncoding.scaleOffset = scaleOffsetData.data();
        weightQuantize.axisScaleOffsetEncoding = weightAxisScaleOffsetEncoding;

        this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t) oc, (uint32_t) ic}, (void *) quantWeightData.data(), weightQuantize);
    }

    // Stage One: reshape input
    {
        mNodeType = "Reshape";
        std::string name = mNodeName + "_reshapeOutput";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(mBackend->getNativeTensor(input)));
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Stage Two: matmul
    {
        mNodeType = "MatMul";
        std::string name = mNodeName + "_MatMul";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // tempInput
        // mInputs.push_back(*(mBackend->getNativeTensor(input)));
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // weight
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // tempOutput
        // mOutputs.push_back(*(mBackend->getNativeTensor(output)));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Stage Three: reshape output
    {
        mNodeType = "Reshape";
        std::string name = mNodeName + "_reshapeOutput";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(output)));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}


void QNNConvolution::createWeight(Qnn_DataType_t dataType, int oc, int ic, int kernelH, int kernelW, int group, const float * source) {
    std::vector<float> weightData;
    int weightElementNum = oc * ic / group * kernelH * kernelW;
    // oc ic/group h w ---> h w ic/group oc
    weightData.resize(weightElementNum);

    convertWeight(source, (float *) weightData.data(), oc, ic/group, kernelH, kernelW);
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
            MNN_ERROR("QNN only support single conv input\n");
            return nullptr;
        }
        return new QNNConvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConvolutionCreator, OpType_Convolution)

} // end namespace QNN
} // end namespace MNN
