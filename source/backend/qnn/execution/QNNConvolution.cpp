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

void QNNConvolution::isWeightQuantSupported(const Tensor *input, const int ic, const int oc){
    Qnn_DataType_t dataType = mBackend->getNativeTensor(input)->v1.dataType;
    if(mOp->main_as_Convolution2D()->quanParameter() == nullptr){
        mWeightQuant = false;
        return;
    }else{
        bool hasBias = false;
        auto bias = mOp->main_as_Convolution2D()->bias();
        auto biasPtr = (float*)bias->data();
        for(int i = 0; i < oc; ++i){
            if(biasPtr[i] != 0.0f){
                hasBias = true;
                break;
            }
        }
        
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);
        int totalCount = quanCommon->alpha.size();
        mBlockSize = totalCount / oc;
        if(quanCommon->asymmetric){
            // not support asymmetric and mBlockSize > 1 results incorrect now
            mWeightQuant = false;
            return;
        }
        
        if(dataType == QNN_DATATYPE_FLOAT_16 || dataType == QNN_DATATYPE_FLOAT_32){
            if(mIsMatMul && mBlockSize == 1){
                mWeightQuant = true;
            }else{
                mWeightQuant = false;
            }
            return;
        }
        
        float inputScale = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.scale;
        int inputOffset = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.offset;
        if(inputOffset == 0){
            mWeightQuant = true;
        }else{
            if(hasBias){
                mWeightQuant = false;
            }else{
                mWeightQuant = true;
            }
        }
        
        if(mBlockSize > 1 && mWeightQuant){
            if(mIs1x1Conv && hasBias == false && (ic / mBlockSize) >= 16){
                mWeightQuant = true;
            }else{
                mWeightQuant = false;
            }
        }
    }
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
    mIs1x1Conv = kernelW==1 && strideH==1 && \
                 strideW==1 && dilationH==1 && dilationW==1 && group==1 && \
                 padTop==0 && padBottom==0 && padLeft==0 && padRight==0;
    mIsMatMul = ih==1 && iw==1 && oh==1 && ow==1 && mIs1x1Conv;
    isWeightQuantSupported(inputs[0], ic, oc);
    
    if(mIsMatMul && mWeightQuant && (dataType == QNN_DATATYPE_FLOAT_16 || dataType == QNN_DATATYPE_FLOAT_32)){
        return onEncodeFpAIntBMatMul(inputs[0], outputs[0], n, ih, iw, ic, oc);
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
    }

    this->createWeightAndBias(dataType, inputs[0], oc, ic, kernelH, kernelW, group);
    // dequant input and quant output
    if(mWeightQuant == false && dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32){
        return this->onEncodeQuantDequantConv(inputs[0], outputs[0], n, ic, oc);
    }
    
    if (common->relu() || common->relu6()) {
        Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
        Qnn_ScaleOffset_t tScaleOffsetEncoding;
        auto quant = TensorUtils::getDescribe(outputs[0])->quantAttr.get();
        if(quant != nullptr && TensorUtils::getDescribe(outputs[0])->type == DataType_DT_INT8){
            quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            quantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = quant->zero;
            quantize.scaleOffsetEncoding = tScaleOffsetEncoding;
        }
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]), quantize);
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
            if(mIsMatMul && n > 1) {
                auto num = closest_factors(n);
                {
                    Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
                    Qnn_ScaleOffset_t tScaleOffsetEncoding;
                    auto quant = TensorUtils::getDescribe(inputs[0])->quantAttr.get();
                    if(quant != nullptr && TensorUtils::getDescribe(inputs[0])->type == DataType_DT_INT8){
                        quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
                        quantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                        tScaleOffsetEncoding.scale = quant->scale;
                        tScaleOffsetEncoding.offset = quant->zero;
                        quantize.scaleOffsetEncoding = tScaleOffsetEncoding;
                    }
                    this->createStageTensor("InputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, ic}), quantize);
                }
                {
                    Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
                    Qnn_ScaleOffset_t tScaleOffsetEncoding;
                    auto quant = TensorUtils::getDescribe(outputs[0])->quantAttr.get();
                    if(quant != nullptr && TensorUtils::getDescribe(outputs[0])->type == DataType_DT_INT8){
                        quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
                        quantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                        tScaleOffsetEncoding.scale = quant->scale;
                        tScaleOffsetEncoding.offset = quant->zero;
                        quantize.scaleOffsetEncoding = tScaleOffsetEncoding;
                    }
                    this->createStageTensor("OutputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, oc}), quantize);
                }
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

ErrorCode QNNConvolution::onEncodeQuantDequantConv(Tensor *input, Tensor *output, const int n, const int ic, const int oc) {
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
                mNodeType = "Conv2d";
                std::string name = mNodeName + "_conv";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
        
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
                mNodeType = common->relu6() ? "ReluMinMax" : "Relu";
                std::string name = mNodeName + "_relu";
                if (common->relu6()) {
                    mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam())); // min_value
                    mParams.push_back(*(mParamScalarWrappers[2]->getNativeParam())); // max_value
                }
                mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // ReluTensor
                mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

        } else {
            if(mIsMatMul && n > 1) {
                auto num = closest_factors(n);
                this->createStageTensor("InputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, ic})); // mTempTensorWrappers[4]
                this->createStageTensor("OutputReshapeTensor", dataType, std::vector<int>({1, num.first, num.second, oc})); // mTempTensorWrappers[5]
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

                    mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // DequantInput
                    mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // InputReshapeTensor
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

                    mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // InputReshapeTensor
                    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

                    mOutputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // OutputReshapeTensor
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }

                // reshape output
                {
                    std::string name = mNodeName + "_output_reshape";
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    mNodeType = "Reshape";

                    mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // OutputReshapeTensor
                    mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
            } else{
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = "Conv2d";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
                
                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // DequantInput
                mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
                
                mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QuantOutput
                mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
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

ErrorCode QNNConvolution::onEncodeFpAIntBMatMul(Tensor * input, Tensor * output, int n, int h, int w, int ic, int oc) {
    // create parameters and stage tensors
    auto conv2D     = mOp->main_as_Convolution2D();
    auto common     = conv2D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(input)->v1.dataType;
    {
        bool transposeWeightFlag = true;
        this->createParamScalar("transpose_in1", transposeWeightFlag);
        
        std::vector<uint32_t> tempInputShape = {(uint32_t) n * h * w , (uint32_t) ic};
        std::vector<uint32_t> tempOutputShape = {(uint32_t) n * h * w , (uint32_t) oc};
        this->createStageTensor("tempInput", dataType, tempInputShape);
        this->createStageTensor("tempOutput", dataType, tempOutputShape);

        // create weight and bias
        {
            Qnn_QuantizeParams_t weightQuantize{};
            std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);
            MNN_ASSERT(!quanCommon->asymmetric);
            const int8_t * source = quanCommon->weight.get();
            std::vector<int8_t> quantWeightData(oc * ic, 0);
            if(quanCommon->canUseInt4){
                for (int o = 0; o < oc; o++) {
                    for (int i = 0; i < ic; i++) {
                        uint32_t srcOffset = o * ic + i;
                        uint32_t dstOffset = srcOffset;
                        if(srcOffset % 2 == 0){
                            quantWeightData[dstOffset] = ((source[srcOffset / 2] >> 4) & 0x0f) - 8;
                        }else{
                            quantWeightData[dstOffset] = (source[srcOffset / 2] & 0x0f) - 8;
                        }
                    }
                }
            }else{
                ::memcpy(quantWeightData.data(), source, oc * ic * sizeof(int8_t));
            }
            mDequantAlpha = quanCommon->alpha.get();
            int totalCount = quanCommon->alpha.size();
            mBlockSize = totalCount / oc;
            int blockNum = ic / mBlockSize;
            if(quanCommon->canUseInt4){
                weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
                weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
                Qnn_BwAxisScaleOffset_t weightBWAxisScaleOffsetEncoding{};
                weightBWAxisScaleOffsetEncoding.bitwidth = 4;
                weightBWAxisScaleOffsetEncoding.axis = 0;
                weightBWAxisScaleOffsetEncoding.numElements = oc;
                mScale.resize(oc);
                std::vector<int32_t> OffsetData(oc);
                for (int i = 0; i < oc; i++) {
                    mScale[i] = mDequantAlpha[i];
                }
                weightBWAxisScaleOffsetEncoding.scales = mScale.data();
                weightQuantize.bwAxisScaleOffsetEncoding = weightBWAxisScaleOffsetEncoding;
                
                this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)oc, (uint32_t)ic}, (void *) quantWeightData.data(), weightQuantize);
                std::function<void()> mReleaseWeightScaleOffset = [&](){
                    std::vector<float>().swap(mScale);
                };
                mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
            }else{
                weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
                weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
                Qnn_AxisScaleOffset_t weightAxisScaleOffsetEncoding{};
                weightAxisScaleOffsetEncoding.axis = 0;
                weightAxisScaleOffsetEncoding.numScaleOffsets = oc;
                mScaleOffsetData.resize(oc);
                for (int i = 0; i < oc; i++) {
                    mScaleOffsetData[i].scale = mDequantAlpha[i];
                    mScaleOffsetData[i].offset = 0;
                }
                weightAxisScaleOffsetEncoding.scaleOffset = mScaleOffsetData.data();
                weightQuantize.axisScaleOffsetEncoding = weightAxisScaleOffsetEncoding;
                
                this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)oc, (uint32_t)ic}, (void *) quantWeightData.data(), weightQuantize);
                std::function<void()> mReleaseWeightScaleOffset = [&](){
                    std::vector<Qnn_ScaleOffset_t>().swap(mScaleOffsetData);
                };
                mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
            }
            //create bias
            this->createBias(dataType, oc, input, quanCommon);
        }
        
        if (common->relu6()) {
            this->createParamScalar("min_value", 0.0f);
            this->createParamScalar("max_value", 6.0f);
        }
        if (common->relu() || common->relu6()) {
            this->createStageTensor("ReluTensor", dataType, getNHWCShape(output));
        }
    }

    // Stage One: reshape input
    {
        mNodeType = "Reshape";
        std::string name = mNodeName + "_reshapeInput";
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
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // bias
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
        if (common->relu() || common->relu6()){
            mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); //ReluTensor
        }else{
            mOutputs.push_back(*(mBackend->getNativeTensor(output)));
        }
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    
    // Stage Four: relu or relu6
    if (common->relu() || common->relu6()){
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
        mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // ReluTensor
        mOutputs.push_back(*(mBackend->getNativeTensor(output))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}

bool QNNConvolution::createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int ic, int kernelH, int kernelW, int group) {
    if(mWeightQuant){
        Qnn_QuantizeParams_t weightQuantize{};
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);
        if(quanCommon->asymmetric) {
            MNN_ERROR("[Error]: Qnn weight quant only support symmetric currently\n");
            return false;
        }
        const int8_t * source = quanCommon->weight.get();
        std::vector<int8_t> quantWeightData(oc * (ic / group) * kernelH * kernelW, 0);
        if(quanCommon->canUseInt4){
            for (int o = 0; o < oc; o++) {
                for (int i = 0; i < ic/group; i++) {
                    for (int h = 0; h < kernelH; h++) {
                        for (int w = 0; w < kernelW; w++) {
                            uint32_t srcOffset = w + kernelW * (h + kernelH * (i + ic/group * o));
                            uint32_t dstOffset = o + oc * (i + ic/group * (w + kernelW * h));
                            if(srcOffset % 2 == 0){
                                quantWeightData[dstOffset] = ((source[srcOffset / 2] >> 4) & 0x0f) - 8;
                            }else{
                                quantWeightData[dstOffset] = (source[srcOffset / 2] & 0x0f) - 8;
                            }
                        }
                    }
                }
            }
        }else{
            convertWeight(source, (int8_t *) quantWeightData.data(), oc, ic/group, kernelH, kernelW);
        }
        mDequantAlpha = quanCommon->alpha.get();
        int totalCount = quanCommon->alpha.size();
        mBlockSize = totalCount / oc;
        // Todo: result is wrong, need to verify
        if(mBlockSize > 1){
            Qnn_QuantizeParams_t weightQuantize{};
            weightQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            weightQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;
            
            weightBlockwiseExpansionEncoding.axis = 3;
            weightBlockwiseExpansionEncoding.numBlocksPerAxis = mBlockSize;
            weightBlockwiseExpansionEncoding.blockScaleBitwidth = 4;
            weightBlockwiseExpansionEncoding.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;
            mBlockScale.resize(oc * mBlockSize);
            mScaleOffsetData.resize(oc);
            for (int i = 0; i < oc; i++) {
                float maxscale = -MAXFLOAT;
                for(int j = 0; j < mBlockSize; ++j){
                    if(mDequantAlpha[i * mBlockSize + j] > maxscale){
                        maxscale = mDequantAlpha[i * mBlockSize + j];
                    }
                }
                float blockScale = maxscale / 16.0f;
                for(int j = 0; j < mBlockSize; ++j){
                    int quantBlock = round(mDequantAlpha[i * mBlockSize + j] / blockScale);
                    mBlockScale[i * mBlockSize + j] = (uint8_t)std::min(std::max(quantBlock, 1), 16);
                }
                mScaleOffsetData[i].scale = blockScale;
                mScaleOffsetData[i].offset = 0;
            }
            weightBlockwiseExpansionEncoding.scaleOffsets = mScaleOffsetData.data();
            weightBlockwiseExpansionEncoding.blocksScale8 = mBlockScale.data();
            weightQuantize.blockwiseExpansion = &weightBlockwiseExpansionEncoding;
            this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, (void *) quantWeightData.data(), weightQuantize);
            std::function<void()> mReleaseWeightScaleOffset = [&](){
                std::vector<Qnn_ScaleOffset_t>().swap(mScaleOffsetData);
            };
            mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
            std::function<void()> mReleaseBlockScale = [&](){
                std::vector<uint8_t>().swap(mBlockScale);
            };
            mBackend->pushReleaseFunc(mReleaseBlockScale);
        }else if(quanCommon->canUseInt4){
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
            
            this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, (void *) quantWeightData.data(), weightQuantize);
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

            this->createStaticTensor("quantWeight", QNN_DATATYPE_SFIXED_POINT_8, {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, (void *) quantWeightData.data(), weightQuantize);
            std::function<void()> mReleaseWeightScaleOffset = [&](){
                std::vector<Qnn_ScaleOffset_t>().swap(mScaleOffsetData);
            };
            mBackend->pushReleaseFunc(mReleaseWeightScaleOffset);
        }
        this->createBias(dataType, oc, input, quanCommon);
    } else {
        std::vector<float> weightData;
        const float* source = nullptr;
        int weightElementNum = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
        ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);
        // oc ic h w ---> h w ic oc
        weightData.resize(weightElementNum);
        convertWeight(source, (float *) weightData.data(), oc, ic/group, kernelH, kernelW);
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if(mBackend->getUseFP16()){
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        this->createStaticFloatTensor("weight", floatDatatype, {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, weightData.data());
        this->createBias(dataType, oc, input, nullptr);
    }
    return NO_ERROR;
}


void QNNConvolution::createBias(Qnn_DataType_t dataType, int oc, const Tensor *input, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) {
    int biasElementNum = oc;
    if(dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32 && mWeightQuant){
        mDequantAlpha = quanCommon->alpha.get();
        float inputScale = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.scale;
        int inputOffset = mBackend->getNativeTensor(input)->v1.quantizeParams.scaleOffsetEncoding.offset;
        std::vector<int> biasData;
        biasData.resize(biasElementNum, 0);

        Qnn_QuantizeParams_t biasQuantize{};
        biasQuantize.encodingDefinition = QNN_DEFINITION_DEFINED;
        biasQuantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        Qnn_AxisScaleOffset_t biasAxisScaleOffsetEncoding{};
        biasAxisScaleOffsetEncoding.axis = 0;
        biasAxisScaleOffsetEncoding.numScaleOffsets = biasElementNum;
        mBiasScaleOffsetData.resize(biasElementNum);

        auto bias = mOp->main_as_Convolution2D()->bias();
        auto biasPtr = (float*)bias->data();
        if (nullptr != bias) {
            for(int i = 0; i < biasElementNum; ++i){
                float biasScale = inputScale * mDequantAlpha[i];
                mBiasScaleOffsetData[i].scale = biasScale;
                mBiasScaleOffsetData[i].offset = 0;
                if(fabs(biasPtr[i]) < 0.000001 || fabs(biasScale) < 0.000001){
                    biasData[i] = 0;
                } else{
                    biasData[i] = (int)(biasPtr[i] / biasScale);
                }
            }
        }
        
        biasAxisScaleOffsetEncoding.scaleOffset = mBiasScaleOffsetData.data();
        biasQuantize.axisScaleOffsetEncoding = biasAxisScaleOffsetEncoding;

        this->createStaticTensor("bias", QNN_DATATYPE_SFIXED_POINT_32, {(uint32_t)biasElementNum}, biasData.data(), biasQuantize);
        std::function<void()> mReleaseBiasScaleOffset = [&](){
            std::vector<Qnn_ScaleOffset_t>().swap(mBiasScaleOffsetData);
        };
        mBackend->pushReleaseFunc(mReleaseBiasScaleOffset);
    }else{
        std::vector<float> biasData;
        biasData.resize(biasElementNum, 0);
        auto bias = mOp->main_as_Convolution2D()->bias();
        if (nullptr != bias) {
            ::memcpy((void *)biasData.data(), (void *)bias->data(), biasElementNum * sizeof(float));
        }
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if(mBackend->getUseFP16()){
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)oc}, biasData.data());
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
