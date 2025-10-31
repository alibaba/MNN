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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

void QNNConvDepthwise::isWeightQuantSupported(const Tensor *input, const int oc){
    Qnn_DataType_t dataType = mBackend->getNativeTensor(input)->v1.dataType;
    if(mOp->main_as_Convolution2D()->quanParameter() == nullptr){
        mWeightQuant = false;
        return;
    }else{
        bool hasBais = false;
        auto bias = mOp->main_as_Convolution2D()->bias();
        auto biasPtr = (float*)bias->data();
        for(int i = 0; i < oc; ++i){
            if(biasPtr[i] != 0.0f){
                hasBais = true;
                break;
            }
        }
        
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);
        if(quanCommon->asymmetric || dataType == QNN_DATATYPE_FLOAT_16 || dataType == QNN_DATATYPE_FLOAT_32){
            // not support asymmetric and mBlockSize > 1 results incorrect now
            mWeightQuant = false;
            return;
        }
        
        float inputScale = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.scale;
        int inputOffset = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.offset;
        if(inputOffset == 0){
            mWeightQuant = true;
        }else{
            if(hasBais){
                mWeightQuant = false;
            }else{
                mWeightQuant = true;
            }
        }
    }
}

ErrorCode QNNConvDepthwise::onEncodeQuantDequantDepthConv(Tensor *input, Tensor *output, const int n, const int ic, const int oc) {
    auto conv2D     = mOp->main_as_Convolution2D();
    auto common     = conv2D->common();
    Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32;
    if(mBackend->getUseFP16()){
        dataType = QNN_DATATYPE_FLOAT_16;
    }
    
    // create dequant input stage tensor
    this->createStageTensor("DequantInput", dataType, getNHWCShape(input)); // mTempTensorWrappers[2]
    this->createStageTensor("QuantOutput", dataType, getNHWCShape(output)); // mTempTensorWrappers[3]
    
    // add nodes
    {
        // dequant input
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "Dequantize";
            std::string name = mNodeName + "_dequant_input";
        
            mInputs.push_back(*(mBackend->getNativeTensor(input))); // input
            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // DequantInput
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        
        if (common->relu() || common->relu6()) {
            this->createStageTensor("ReluTensor", dataType, getNHWCShape(output)); // mTempTensorWrappers[4]
            // Stage one
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = "DepthWiseConv2d";
                std::string name = mNodeName + "_convDepthwise";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
        
                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // DequantInput
                mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
        
                mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // ReluTensor
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

            // Stage two
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = common->relu6() ? "Relu6" : "Relu";
                std::string name = mNodeName + "_relu";
                mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // ReluTensor
                mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

        } else {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "DepthWiseConv2d";
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
            
            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // DequantInput
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
            
            mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        
        // Quant output
        {
            auto QuantOutputTensor = mTempTensorWrappers[3]->getNativeTensor();
            if(mBackend->getUseFP16()){
                this->createStageTensor("CastOutput", QNN_DATATYPE_FLOAT_32, getNHWCShape(output));
                
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = "Cast";
                std::string name = mNodeName + "_Cast_Output";
                
                mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
                mOutputs.push_back(*(mTempTensorWrappers.back()->getNativeTensor())); // CastOutput
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                QuantOutputTensor = mTempTensorWrappers.back()->getNativeTensor();
            }
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = "Quantize";
                std::string name = mNodeName + "_Quant_Output";
                
                mInputs.push_back(*(QuantOutputTensor)); // stage tensor
                mOutputs.push_back(*(mBackend->getNativeTensor(output))); // output
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
        }
    }
    return NO_ERROR;
}

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
    
    isWeightQuantSupported(inputs[0], oc);
    // create all tensors and params
{
    std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
    std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft, (uint32_t)padRight};
    std::vector<uint32_t> dilationData = {(uint32_t)dilationH, (uint32_t)dilationW};
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());
    this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {2}, (void *)dilationData.data());

    this->createWeightAndBias(dataType, inputs[0], oc, kernelH, kernelW);
    // dequant input and quant output
    if(mWeightQuant == false && dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32){
        return this->onEncodeQuantDequantDepthConv(inputs[0], outputs[0], n, ic, oc);
    }
    
    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]), outputs[0]);
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
                mNodeType = common->relu6() ? "Relu6" : "Relu";
                std::string name = mNodeName + "_relu";
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



void QNNConvDepthwise::createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int kernelH, int kernelW) {
    if(mWeightQuant){
        Qnn_QuantizeParams_t weightQuantize{};
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);

        // [TODO] Support asymmetric and other quantBits.
        MNN_ASSERT(!quanCommon->asymmetric);

        // create weight
        const int8_t * source = quanCommon->weight.get();
        std::vector<int8_t> quantWeightData(oc * kernelH * kernelW, 0);
        if(quanCommon->canUseInt4){
            for (int c = 0; c < oc; c++) {
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        int srcOffset = w + kernelW * (h + kernelH * c);
                        int dstOffset = c + oc * (w + kernelW * h);
                        if(srcOffset % 2 == 0){
                            quantWeightData[dstOffset] = ((source[srcOffset / 2] >> 4) & 0x0f) - 8;
                        }else{
                            quantWeightData[dstOffset] = (source[srcOffset / 2] & 0x0f) - 8;
                        }
                    }
                }
            }
        }else{
            convertWeight(source, (int8_t *) quantWeightData.data(), oc, kernelH, kernelW);
        }

        mDequantAlpha = quanCommon->alpha.get();
        if(quanCommon->canUseInt4){
            weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
            Qnn_BwAxisScaleOffset_t weightBWAxisScaleOffsetEncoding{};
            weightBWAxisScaleOffsetEncoding.bitwidth = 4;
            weightBWAxisScaleOffsetEncoding.axis = 3;
            weightBWAxisScaleOffsetEncoding.numElements = oc;
            mScale.resize(oc);
            std::vector<int32_t> OffsetData(oc);
            for (int i = 0; i < oc; i++) {
                mScale[i] = mDequantAlpha[i];
            }
            weightBWAxisScaleOffsetEncoding.scales = mScale.data();
            weightQuantize.bwAxisScaleOffsetEncoding = weightBWAxisScaleOffsetEncoding;
            
            this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)kernelH, (uint32_t)kernelW, 1, (uint32_t)oc}, (void *) quantWeightData.data(), weightQuantize);
            std::function<void()> mReleaseWeightScaleOffset = [&](){
                std::vector<float>().swap(mScale);
            };
            mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
        }else{
            weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
            Qnn_AxisScaleOffset_t weightAxisScaleOffsetEncoding{};
            weightAxisScaleOffsetEncoding.axis = 3;
            weightAxisScaleOffsetEncoding.numScaleOffsets = oc;
            mScaleOffsetData.resize(oc);
            for (int i = 0; i < oc; i++) {
                mScaleOffsetData[i].scale = mDequantAlpha[i];
                mScaleOffsetData[i].offset = 0;
            }
            weightAxisScaleOffsetEncoding.scaleOffset = mScaleOffsetData.data();
            weightQuantize.axisScaleOffsetEncoding = weightAxisScaleOffsetEncoding;

            this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)kernelH, (uint32_t)kernelW, 1, (uint32_t)oc}, (void *) quantWeightData.data(), weightQuantize);
        
            std::function<void()> mReleaseWeightScaleOffset = [&](){
                std::vector<Qnn_ScaleOffset_t>().swap(mScaleOffsetData);
            };
            mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
        }
        // create bias
        {
            float inputScale = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.scale;
            int inputOffset = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.offset;
            std::vector<int> biasData;
            biasData.resize(oc, 0);

            Qnn_QuantizeParams_t biasQuantize{};
            biasQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            biasQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
            Qnn_AxisScaleOffset_t biasAxisScaleOffsetEncoding{};
            biasAxisScaleOffsetEncoding.axis = 0;
            biasAxisScaleOffsetEncoding.numScaleOffsets = oc;
            mBiasScaleOffsetData.resize(oc);

            auto bias = mOp->main_as_Convolution2D()->bias();
            auto biasPtr = (float*)bias->data();
            if (nullptr != bias) {
                for(int i = 0; i < oc; ++i){
                    float biasScale = inputScale * mDequantAlpha[i];
                    mBiasScaleOffsetData[i].scale = 0.f;
                    mBiasScaleOffsetData[i].offset = 0;
                    if(biasPtr[i] == 0.0f){
                        biasData[i] = 0;
                    } else{
                        biasData[i] = (int)(biasPtr[i] / (biasScale));
                    }
                }
            }
            biasAxisScaleOffsetEncoding.scaleOffset = mBiasScaleOffsetData.data();
            biasQuantize.axisScaleOffsetEncoding = biasAxisScaleOffsetEncoding;

            this->createStaticTensor("bias", QNN_DATATYPE_SFIXED_POINT_32, {(uint32_t)oc}, biasData.data(), biasQuantize);
            std::function<void()> mReleaseBiasScaleOffset = [&](){
                std::vector<Qnn_ScaleOffset_t>().swap(mBiasScaleOffsetData);
            };
            mBackend->pushReleaseFunc(mReleaseBiasScaleOffset);
        }
    }else{
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if(mBackend->getUseFP16()){
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        std::vector<float> weightData;
        const float* source = nullptr;
        int weightElementNum = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
        ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);
        // oc ic h w ---> h w ic oc
        weightData.resize(weightElementNum);
        convertWeight(source, (float *) weightData.data(), oc, kernelH, kernelW);
        this->createStaticFloatTensor("weight", floatDatatype, {(uint32_t)kernelH, (uint32_t)kernelW, 1, (uint32_t)oc}, weightData.data());
        
        // create bias
        std::vector<float> biasData;
        biasData.resize(oc, 0);
        auto bias = mOp->main_as_Convolution2D()->bias();
        if (nullptr != bias) {
            ::memcpy((void *)biasData.data(), (void *)bias->data(), oc * sizeof(float));
        }
        this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)oc}, biasData.data());
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
#endif
} // end namespace QNN
} // end namespace MNN
