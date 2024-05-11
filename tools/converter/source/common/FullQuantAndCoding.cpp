//
//  FullQuantAndCoding.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include "CommonUtils.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "core/IDSTEncoder.hpp"

using namespace MNN;
using namespace MNN::Express;

void FullQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, std::unique_ptr<MNN::OpT>& op, Compression::Pipeline& proto, SubGraphProtoT* subgraph) {
    std::string outputTensorName = subgraph ? subgraph->tensors[op->outputIndexes[0]] : netT->tensorName[op->outputIndexes[0]];;
    auto opType = op->type;
    if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
        return;
    }
    if (op->inputIndexes.size() != 1) {
        return;
    }

    auto findQuantParameters = [&](Compression::Pipeline& proto, std::string outputTensorName) {
        for (const auto& algo : proto.algo()) {
            if (algo.type() == Compression::CompressionAlgo::QUANTIZE) {
                auto quantParams = algo.quant_params();
                for (const auto& layerProto : quantParams.layer()) {
                    const std::string& outputName = layerProto.output(0).name();
                    if ((outputName == outputTensorName) || (outputTensorName == outputName+"__matmul_converted")) {
                        return layerProto;
                    }
                }
            }
        }
        MNN::Compression::LayerQuantizeParams empty;
        return empty;
    };

    auto inputIndex = op->inputIndexes[0];
    int outputIndex = op->outputIndexes[0];
    auto quantParams = findQuantParameters(proto, outputTensorName);
    if (quantParams.weight_size() == 0) {
        return;
    }

    auto inputParams = quantParams.input(0);
    auto outputParams = quantParams.output(0);
    auto weightParams = quantParams.weight(0);
    auto& tensorDescribe = subgraph ? subgraph->extraTensorDescribe : netT->extraTensorDescribe;

    auto findInDescribe = [&] (int index) {
        for (int i = 0; i < tensorDescribe.size(); i++) {
            if (tensorDescribe[i]->index == index) {
                return true;
            }
        }
        return false;
    };

    if (!findInDescribe(inputIndex)) {
        std::unique_ptr<MNN::TensorDescribeT> inDescribe(new MNN::TensorDescribeT);
        inDescribe->index = inputIndex;
        std::unique_ptr<MNN::TensorQuantInfoT> inputQuantInfo(new MNN::TensorQuantInfoT);
        inputQuantInfo->zero = inputParams.zero_point();
        inputQuantInfo->scale = inputParams.scales(0);
        inputQuantInfo->min = inputParams.clamp_min();
        inputQuantInfo->max = inputParams.clamp_max();
        inputQuantInfo->type = MNN::DataType_DT_INT8;
        inDescribe->quantInfo = std::move(inputQuantInfo);
        tensorDescribe.emplace_back(std::move(inDescribe));
    }

    if (!findInDescribe(outputIndex)) {
        std::unique_ptr<MNN::TensorDescribeT> outDescribe(new MNN::TensorDescribeT);
        outDescribe->index = outputIndex;
        std::unique_ptr<MNN::TensorQuantInfoT> outputQuantInfo(new MNN::TensorQuantInfoT);
        outputQuantInfo->zero = outputParams.zero_point();
        outputQuantInfo->scale = outputParams.scales(0);
        outputQuantInfo->min = outputParams.clamp_min();
        outputQuantInfo->max = outputParams.clamp_max();
        outputQuantInfo->type = MNN::DataType_DT_INT8;
        outDescribe->quantInfo = std::move(outputQuantInfo);
        tensorDescribe.emplace_back(std::move(outDescribe));
    }

    auto convParams  = op->main.AsConvolution2D();
    auto weightFloat = convParams->weight;
    auto biasFloat   = convParams->bias;
    auto& common     = convParams->common;

    const int ko = common->outputCount;
    const int ki = common->inputCount / common->group;
    const int kh = common->kernelY;
    const int kw = common->kernelX;
    const int kernelNum = common->outputCount;
    int kernelSize = weightFloat.size() / kernelNum;

    VARP weightVar = _Const(weightFloat.data(), {ko, ki, kh, kw}, NCHW);
    VARP biasVar        = _Const(biasFloat.data(), {ko, 1, 1, 1}, NCHW);
    VARP inputScaleVar  = _Const(inputParams.scales(0), {}, NCHW);
    VARP outputScaleVar = _Const(outputParams.scales(0), {}, NCHW);

    float wClampMin = weightParams.clamp_min();
    float wClampMax = weightParams.clamp_max();

    std::vector<float> weightScaleVector(weightParams.scales().begin(), weightParams.scales().end());
    VARP weightScale = _Const(weightScaleVector.data(), {(int)weightScaleVector.size(), 1, 1, 1}, NCHW, halide_type_of<float>());
    auto quanWeightTemp = _Round(weightVar * _Reciprocal(weightScale));
    auto quanWeightClamp = MNN::Express::_Maximum(_Minimum(quanWeightTemp, _Scalar<float>(wClampMax)), _Scalar<float>(wClampMin));
    auto quanWeight = _Cast<int8_t>(quanWeightClamp);
    auto convScale  = _Reshape(_Reciprocal(outputScaleVar), {-1, 1, 1, 1}) * weightScale * inputScaleVar;

    std::vector<float> quantWeightFloat;
    std::vector<int8_t> quantWeights;
    std::vector<float> biasData;
    std::vector<float> scale;

    {
        auto info = quanWeight->getInfo();
        quantWeights.resize(info->size);
        quantWeightFloat.resize(info->size);
        auto ptr = quanWeight->readMap<int8_t>();
        for (int i = 0; i < quantWeightFloat.size(); i++) {
            quantWeightFloat[i] = ptr[i];
            quantWeights[i] = ptr[i];
        }
    }
    {
        auto biasinfo = biasVar->getInfo();
        biasData.resize(biasinfo->size);
        auto ptr = biasVar->readMap<float>();
        ::memcpy(biasData.data(), ptr, biasData.size() * sizeof(int32_t));

        auto info = weightScale->getInfo();
        scale.resize(info->size);
        MNN_ASSERT(scale.size() == biasData.size());
        auto ptrScale = weightScale->readMap<float>();
        ::memcpy(scale.data(), ptrScale, scale.size() * sizeof(float));
    }

    bool asymmetricQuantFlag = false;
    std::vector<float> fakeScales(kernelNum, 1.0f);
    convParams->quanParameter = IDSTEncoder::encode(quantWeightFloat.data(), fakeScales, kernelSize, kernelNum, asymmetricQuantFlag, quantWeights.data(), wClampMin);
    convParams->weight.clear();
    convParams->quanParameter->alpha = std::move(scale);
    convParams->quanParameter->scaleIn = inputParams.scales(0);
    convParams->quanParameter->scaleOut = outputParams.scales(0);

    convParams->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
    convParams->symmetricQuan->method = MNN::QuantizeAlgo(int(quantParams.method()));
    convParams->symmetricQuan->nbits = outputParams.bits();

    convParams->symmetricQuan->zeroPoint = inputParams.zero_point();
    convParams->symmetricQuan->outputZeroPoint = outputParams.zero_point();
    convParams->symmetricQuan->clampMin = outputParams.clamp_min();
    convParams->symmetricQuan->clampMax = outputParams.clamp_max();

    convParams->bias = std::move(biasData);
    // winogradAttr store:
    // 1. transformed weight and input scale
    // 2. winograd config (F(2,3)/F(4,3)/F(6,3)/...)
    if (quantParams.method() == MNN::Compression::LayerQuantizeParams::WinogradAware) {
        const auto& attr = quantParams.wino_params().units_attr();
        convParams->symmetricQuan->winogradAttr.assign(attr.begin(), attr.end());
    }
};

void fullQuantAndCoding(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto) {
    for (auto& op : netT->oplists) {
        FullQuantAndCoding(netT, op, proto, nullptr);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            FullQuantAndCoding(netT, op, proto, subgraph.get());
        }
    }
}
