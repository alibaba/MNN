//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <string>
#include <random>
#include <sstream>

#include "MNN_generated.h"
#include "half.hpp"
#include "logkit.h"
#include "writeFb.hpp"
#include "cpp/ConfigFile.hpp"
#include <MNN/MNNDefine.h>
#include "cli.hpp"
#include "MNN_compression.pb.h"
#include "MNN/expr/ExprCreator.hpp"
#include "cpp/IDSTEncoder.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/cpu/compute/SparseConvolutionTiledExecutor.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

static float findAbsMax(const float *weights, const int count) {
    float absMax = abs(weights[0]);
    for (int i = 1; i < count; i++) {
        float value = abs(weights[i]);
        if (value > absMax) {
            absMax = value;
        }
    }

    return absMax;
}

static std::vector<float> findMinMax(const float *weights, const int count) {
    float min = weights[0];
    float max = weights[0];

    for (int i = 1; i < count; i++) {
        float value = weights[i];
        if (value > max) {
            max = value;
        }
        if (value < min) {
            min = value;
        }
    }

    return {min, max};
}

static std::string uuid4() {
    static std::random_device              rd;
    static std::mt19937_64                 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
       ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
       ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
       ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
       ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
       ss << dis(gen);
    };
    return ss.str();
}

int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, const modelConfig& config) {
    auto RemoveParams = [](std::unique_ptr<MNN::OpT>& op) {
        const auto opType = op->type;
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_Deconvolution:
            case MNN::OpType_ConvolutionDepthwise: {
                auto param = op->main.AsConvolution2D();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_TfQuantizedConv2D: {
                auto param = op->main.AsTfQuantizedConv2D();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_MatMul: {
                auto param = op->main.AsMatMul();
                param->weight.clear();
                param->bias.clear();
                break;
            }
            case MNN::OpType_BatchNorm: {
                auto param = op->main.AsBatchNorm();
                param->slopeData.clear();
                param->meanData.clear();
                param->varData.clear();
                param->biasData.clear();
                param->Adata.clear();
                param->Bdata.clear();
                break;
            }
            case MNN::OpType_Scale: {
                auto param = op->main.AsScale();
                param->scaleData.clear();
                param->biasData.clear();
                break;
            }
            default:
                break;
        }
    };
    if (config.benchmarkModel) {
        for (auto& op : netT->oplists) {
            RemoveParams(op);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                RemoveParams(op);
            }
        }
    }

    auto CastParamsToHalf = [](std::unique_ptr<MNN::OpT>& op) {
        const auto opType = op->type;
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_ConvolutionDepthwise: {
                auto param           = op->main.AsConvolution2D();
                const int weightSize = param->weight.size();
                // const int biasSize = param->bias.size();
                std::vector<half_float::half> quantizedFp16Weight;
                quantizedFp16Weight.resize(weightSize);
                std::transform(param->weight.begin(), param->weight.end(), quantizedFp16Weight.begin(),
                               [](float w) { return half_float::half(w); });
                // std::vector<half_float::half> quantizedFp16Bias;
                // quantizedFp16Bias.resize(biasSize);
                // std::transform(param->bias.begin(), param->bias.end(), quantizedFp16Bias.begin(), [](float
                // b){return half_float::half(b); });
                param->weight.clear();
                // param->bias.clear();

                param->quanParameter.reset(new MNN::IDSTQuanT);
                param->quanParameter->type = 3;
                int8_t* halfWeight         = reinterpret_cast<int8_t*>(quantizedFp16Weight.data());
                param->quanParameter->buffer.assign(halfWeight, halfWeight + sizeof(half_float::half) * weightSize);
                break;
            }
            case MNN::OpType_Const: {
                auto blob = op->main.AsBlob();
                if (blob->dataType == MNN::DataType_DT_FLOAT) {
                    blob->dataType = MNN::DataType_DT_HALF;
                    blob->uint8s.resize(sizeof(half_float::half) * blob->float32s.size());
                    auto size = blob->float32s.size();
                    auto dst = (half_float::half*)blob->uint8s.data();
                    for (int i=0; i<size; ++i) {
                        dst[i] = blob->float32s[i];
                    }
                    blob->float32s.clear();
                }
                break;
            }
            default:
                break;
        }
    };
    if (config.saveHalfFloat) {
        for (auto& op : netT->oplists) {
            CastParamsToHalf(op);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                CastParamsToHalf(op);
            }
        }
    }

    auto AddSparseInfo = [&](std::unique_ptr<MNN::OpT>& op, Compression::Pipeline proto) {
        auto prune_algo_type = MNN::SparseAlgo_RANDOM;
        int sparseBlockOC = 1;
        int sparseBlockKernel = 1;

        for (const auto& algo : proto.algo()) {
            if (algo.type() == Compression::CompressionAlgo::PRUNE) {
                auto prune_type = algo.prune_params().type();
                prune_algo_type = MNN::SparseAlgo(prune_type);
                if (prune_type == Compression::PruneParams_PruneType_SIMD_OC) {
                    sparseBlockOC = algo.prune_params().simd_oc_pruner_params().oc_blocks(0);
                }
            }
        }

        const auto opType = op->type;
        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_ConvolutionDepthwise: {
                auto param = op->main.AsConvolution2D();
                if (param->weight.empty()) {
                    return;
                }

                int weightSize = param->weight.size();
                int biasSize = param->bias.size();
                size_t weightNNZElement, weightBlockNumber = 0;
                OpCommonUtils::statisticWeightSparsity(weightNNZElement, weightBlockNumber, param->weight.data(), biasSize, weightSize / biasSize, sparseBlockOC);
                float sparsity = 1. - float(weightNNZElement) / weightSize;
                if (sparsity < SPARSITY_THRESHOLD) {
                    return;
                }

                MNN::AttributeT* arg1(new MNN::AttributeT);
                arg1->key = "sparseBlockOC";
                arg1->i = sparseBlockOC;

                MNN::AttributeT* arg2(new MNN::AttributeT);
                arg2->key = "sparseBlockKernel";
                arg2->i = sparseBlockKernel;

                MNN::AttributeT* arg3(new MNN::AttributeT);
                arg3->key = "NNZElement";
                arg3->i = weightNNZElement;

                MNN::AttributeT* arg4(new MNN::AttributeT);
                arg4->key = "blockNumber";
                arg4->i = weightBlockNumber;

                flatbuffers::FlatBufferBuilder builder;
                std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
                auto sparseArg1 = MNN::CreateAttribute(builder, arg1);
                auto sparseArg2 = MNN::CreateAttribute(builder, arg2);
                auto sparseArg3 = MNN::CreateAttribute(builder, arg3);
                auto sparseArg4 = MNN::CreateAttribute(builder, arg4);

                argsVector.emplace_back(sparseArg1);
                argsVector.emplace_back(sparseArg2);
                argsVector.emplace_back(sparseArg3);
                argsVector.emplace_back(sparseArg4);

                auto sparseArgs = builder.CreateVectorOfSortedTables<MNN::Attribute>(&argsVector);
                auto sparseCom = MNN::CreateSparseCommon(builder, prune_algo_type, sparseArgs);
                builder.Finish(sparseCom);
                auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();
                
                param->sparseParameter.reset(sparseComPtr);
                
                break;
            }
            default:
                break;
        }
    };

    {
        std::string compressFileName = config.compressionParamsFile;
        Compression::Pipeline proto;
        if (compressFileName != "") {
            std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            }
        }
        for (auto& op : netT->oplists) {
            AddSparseInfo(op, proto);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                AddSparseInfo(op, proto);
            }
        }
    }

    auto FullQuantAndCoding = [&](std::unique_ptr<MNN::OpT>& op, Compression::Pipeline& proto, SubGraphProtoT* subgraph) {
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
                        if (outputName == outputTensorName) {
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
        const int kernelSize = weightFloat.size() / kernelNum;

        VARP weightVar      = _Const(weightFloat.data(), {ko, ki, kh, kw}, NCHW);
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
        convParams->quanParameter = IDSTEncoder::encode(quantWeightFloat, fakeScales, kernelSize, kernelNum, asymmetricQuantFlag, quantWeights.data(), wClampMin);
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
    };

    {
        std::string compressFileName = config.compressionParamsFile;
        if (compressFileName != "") {
            Compression::Pipeline proto;
            std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            }
            // set uuid from compress file
            if (proto.has_mnn_uuid()) {
                netT->mnn_uuid = proto.mnn_uuid();
            }
            for (auto& op : netT->oplists) {
                FullQuantAndCoding(op, proto, nullptr);
            }
            for (auto& subgraph : netT->subgraphs) {
                for (auto& op : subgraph->nodes) {
                    FullQuantAndCoding(op, proto, subgraph.get());
                }
            }
        }
    }

    auto WeightQuantAndCoding = [&](std::unique_ptr<MNN::OpT>& op) {
        const auto opType = op->type;
        // config.weightQuantBits only control weight quantization for float convolution
        // by default, do coding for convint8 and depthwiseconvint8, if there is any
        if ((config.weightQuantBits == 0) && (
            opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
            return;
        }
        
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
            opType != MNN::OpType_Deconvolution && opType != MNN::OpType_DeconvolutionDepthwise &&
            opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8) {
                return;
        }

        int bits = 8;
        if ((config.weightQuantBits > 0) && (
            opType != MNN::OpType_ConvInt8 && opType != MNN::OpType_DepthwiseConvInt8)) {
            bits = config.weightQuantBits;
        }
        // Bits must from 2-8
        bits = std::max(bits, 2);
        bits = std::min(bits, 8);

        auto param           = op->main.AsConvolution2D();
        auto& common = param->common;
        if (param->quanParameter.get() != nullptr) {
            return;
        }
        
        int weightSize = param->weight.size();
        if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
            weightSize = param->symmetricQuan->weight.size();
        }
        int kernelNum = common->outputCount;
        int kernelSize = weightSize / kernelNum;

        bool asymmetricQuantFlag = config.weightQuantAsymmetric;

        float threshold = (float)(1 << (bits - 1)) - 1.0f;
        float clampMin = -threshold;
        if (asymmetricQuantFlag) {
            clampMin = -threshold - 1;
        }
        std::vector<float> weightData, scales;
        std::vector<int8_t> quantWeights;

        switch (opType) {
            case MNN::OpType_Convolution:
            case MNN::OpType_ConvolutionDepthwise:
            case MNN::OpType_Deconvolution:
            case MNN::OpType_DeconvolutionDepthwise: {
                weightData = param->weight;
                
                if (asymmetricQuantFlag) {
                    scales.resize(kernelNum*2);
                    for (int k = 0; k < kernelNum; k++) {
                        int beginIndex = k * kernelSize;
                        auto minAndMax = findMinMax(weightData.data() + beginIndex, kernelSize);
                        float min = minAndMax[0];
                        float max = minAndMax[1];
                        float scale = (max - min) / (threshold - clampMin);

                        scales[2*k] = min;
                        scales[2*k+1] = scale;

                        for (int ii = 0; ii < kernelSize; ii++) {
                            float* ptr = weightData.data() + beginIndex;
                            int8_t quantValue = int8_t(std::round((ptr[ii] - min) / scale + clampMin));
                            quantWeights.emplace_back(quantValue);
                        }
                    }
                } else {
                    scales.resize(kernelNum);
                    for (int k = 0; k < kernelNum; k++) {
                        int beginIndex = k * kernelSize;
                        auto absMax = findAbsMax(weightData.data() + beginIndex, kernelSize);

                        scales[k] = absMax / threshold;

                        for (int ii = 0; ii < kernelSize; ii++) {
                            float* ptr = weightData.data() + beginIndex;
                            int8_t quantValue = int8_t(std::round(ptr[ii] / scales[k]));
                            quantWeights.emplace_back(quantValue);
                        }
                    }
                }
                
                break;
            }
            case MNN::OpType_ConvInt8:
            case MNN::OpType_DepthwiseConvInt8: {
                auto& int8Params = param->symmetricQuan;
                for (int i = 0; i < int8Params->weight.size(); i++) {
                    weightData.emplace_back(float(int8Params->weight[i]));
                }
                scales.resize(kernelNum, 1.0f);
                
                break;
            }
            default:
                break;
        }

        if (opType == MNN::OpType_ConvInt8 || opType == MNN::OpType_DepthwiseConvInt8) {
            param->quanParameter = IDSTEncoder::encode(weightData, scales, kernelSize, kernelNum, false, param->symmetricQuan->weight.data(), int(clampMin));
            param->symmetricQuan->weight.clear();
            param->quanParameter->alpha = {1.0f}; // fake scales
        } else {
            param->quanParameter = IDSTEncoder::encode(weightData, scales, kernelSize, kernelNum, asymmetricQuantFlag, quantWeights.data(), int(clampMin));
            param->weight.clear();
        }
    };

    {
        for (auto& op : netT->oplists) {
            WeightQuantAndCoding(op);
        }
        for (auto& subgraph : netT->subgraphs) {
            for (auto& op : subgraph->nodes) {
                WeightQuantAndCoding(op);
            }
        }
    }

    std::set<std::string> notSupportOps;
    auto CheckIfNotSupported = [&] (const std::unique_ptr<MNN::OpT>& op) {
        if (op->type == MNN::OpType_Extra) {
            if (op->main.AsExtra()->engine != "MNN") {
                notSupportOps.insert(op->main.AsExtra()->engine + "::" + op->main.AsExtra()->type);
            }
        }
    };
    for (auto& op : netT->oplists) {
        CheckIfNotSupported(op);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            CheckIfNotSupported(op);
        }
    }

    std::ostringstream notSupportInfo;
    if (!notSupportOps.empty()) {
        for (auto name : notSupportOps) {
            notSupportInfo << name << " | ";
        }
        auto opNames = notSupportInfo.str();
        LOG(FATAL) << "These Op Not Support: " << opNames.substr(0, opNames.size() - 2);
        return 1;
    }

    // dump input and output tensor name
    {
        std::set<int> inputIdx, outputIdx, realInput, realOutput;
        for (const auto& op : netT->oplists) {
            for (auto i : op->inputIndexes) {
                inputIdx.insert(i);
            }
            for (auto o : op->outputIndexes) {
                outputIdx.insert(o);
                if (op->type == OpType_Input) {
                    realInput.insert(o);
                }
            }
        }
        std::set_difference(outputIdx.begin(), outputIdx.end(), inputIdx.begin(), inputIdx.end(), std::inserter(realOutput, realOutput.begin()));
        std::cout << "inputTensors : [ ";
        for (int i : realInput) {
            std::cout << netT->tensorName[i] << ", ";
        }
        std::cout << "]\noutputTensors: [ ";
        if (netT->outputName.size() > 0) {
            for (auto& o : netT->outputName) {
                std::cout << o << ", ";
            }
        } else {
            for (int i : realOutput) {
                std::cout << netT->tensorName[i] << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    // set uuid with uuid4 if not set in compress file
    if (netT->mnn_uuid.empty()) {
        netT->mnn_uuid = uuid4();
    }
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();

    if (config.saveStaticModel && netT->usage != MNN::Usage_INFERENCE_STATIC) {
        std::map<std::string, std::vector<int>> inputConfig;
        // get config to set input size
        if (config.inputConfigFile.size() > 0) {
            ConfigFile conf(config.inputConfigFile);
            auto numOfInputs = conf.Read<int>("input_size");
            auto inputNames  = splitNames(numOfInputs, conf.Read<std::string>("input_names"));
            auto inputDims   = splitDims(numOfInputs, conf.Read<std::string>("input_dims"));
            for (int i = 0; i < numOfInputs; i++) {
                inputConfig.insert(std::make_pair(inputNames[i], inputDims[i]));
            }
        }
        const Net* net = flatbuffers::GetRoot<MNN::Net>(bufferOutput);
        converToStaticModel(net, inputConfig, MNNModelFile);
    } else {
        std::ofstream output(MNNModelFile, std::ofstream::binary);
        output.write((const char*)bufferOutput, sizeOutput);
    }

#ifdef MNN_DUMP_SUBGRAPH
    for (int i = 0; i < netT->subgraphs.size(); ++i) {
        std::unique_ptr<MNN::NetT> subnet(new MNN::NetT);
        auto& subgraph = netT->subgraphs[i];
        subnet->oplists = std::move(subgraph->nodes);
        subnet->tensorName = subgraph->tensors;
        subnet->sourceType = netT->sourceType;
        subnet->bizCode = netT->bizCode;

        flatbuffers::FlatBufferBuilder builder(1024);
        builder.ForceDefaults(true);
        auto len = MNN::Net::Pack(builder, subnet.get());
        builder.Finish(len);
        int output_size = builder.GetSize();
        auto* output_ptr = builder.GetBufferPointer();

        std::string filename =
            MNNModelFile + "_subgraph_" + std::to_string(i) + ".mnn";
        std::ofstream output(filename.c_str(), std::ofstream::binary);
        output.write((const char*)output_ptr, output_size);
    }
#endif
    return 0;
}
