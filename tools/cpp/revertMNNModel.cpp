//
//  revertMNNModel.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstdlib>
#include <random>
#include <ctime>
#include <fstream>
#include <iostream>

#include <string.h>
#include <stdlib.h>
#include <MNN/MNNDefine.h>
#include "revertMNNModel.hpp"
#include "core/CommonCompute.hpp"
#include "core/MemoryFormater.h"
#include "core/IDSTEncoder.hpp"
#include "core/ConvolutionCommon.hpp"

int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue) {
    const int channelStride     = size / channels;
    const int quantizedMaxValue = weightClampValue;

    for (int c = 0; c < channels; ++c) {
        const auto weightChannelStart    = weight + c * channelStride;
        auto quantizedWeightChannelStart = quantizedWeight + c * channelStride;
        auto minmaxValue                 = std::minmax_element(weightChannelStart, weightChannelStart + channelStride);
        const float dataAbsMax           = std::fmax(std::fabs(*minmaxValue.first), std::fabs(*minmaxValue.second));

        float scaleDataToInt8 = 1.0f;
        if (dataAbsMax == 0) {
            scale[c] = 0.0f;
        } else {
            scale[c]        = dataAbsMax / quantizedMaxValue;
            scaleDataToInt8 = quantizedMaxValue / dataAbsMax;
        }

        for (int i = 0; i < channelStride; ++i) {
            const int32_t quantizedInt8Value = static_cast<int32_t>(roundf(weightChannelStart[i] * scaleDataToInt8));
            quantizedWeightChannelStart[i] =
                std::min(quantizedMaxValue, std::max(-quantizedMaxValue, quantizedInt8Value));
        }
    }

    return 0;
}


Revert::Revert(const char* originalModelFileName) {
    std::ifstream inputFile(originalModelFileName, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    const auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read(buffer, size);
    inputFile.close();
    mMNNNet = MNN::UnPackNet(buffer);
    delete[] buffer;
    MNN_ASSERT(mMNNNet->oplists.size() > 0);
}

Revert::~Revert() {
}

void* Revert::getBuffer() const {
    return reinterpret_cast<void*>(mBuffer.get());
}

const size_t Revert::getBufferSize() const {
    return mBufferSize;
}

void Revert::writeExtraDescribeTensor(float* scale, float* offset) {
    int opCounts = static_cast<int32_t>(mMNNNet->oplists.size());
    for (int opIndex = 0; opIndex < opCounts; ++opIndex) {
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        describe->index = opIndex;
        describe->quantInfo.reset(new MNN::TensorQuantInfoT);
        describe->quantInfo->scale = *scale;
        describe->quantInfo->zero = *offset;
        describe->quantInfo->min = -127;
        describe->quantInfo->max = 127;
        describe->quantInfo->type = MNN::DataType_DT_INT8;
        mMNNNet->extraTensorDescribe.emplace_back(std::move(describe));
    }
    for (const auto& op: mMNNNet->oplists) {
        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise && opType != MNN::OpType_Deconvolution) {
            continue;
        }
        // Conv/ConvDepthwise/Deconv weight quant.
        const float inputScale = *scale;
        const float outputScale = *scale;
        const int outputChannel = static_cast<int32_t>(op->outputIndexes.size());

        auto param = op->main.AsConvolution2D();
        float* originWeight = param->weight.data();
        const int channels = param->common->outputCount;
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        param->symmetricQuan->nbits = 8;
        const int weightSize = static_cast<int32_t>(param->weight.size());
        param->common->inputCount = weightSize / (channels * param->common->kernelX * param->common->kernelY);
        std::vector<int8_t> quantizedWeight(weightSize);
        std::vector<float> quantizedWeightScale(channels);

        if (originWeight[0] == 0.f && originWeight[1] == 0.f) { // Process weight is null.
            // Initialize originWeight
            std::uniform_real_distribution<double> u(-200, 200);
            std::default_random_engine e(time(NULL));
            for (int i = 0; i < weightSize; ++i) {
                originWeight[i] = u(e);
            }
        }

        SymmetricQuantizeWeight(originWeight, weightSize, quantizedWeight.data(), quantizedWeightScale.data(), channels, 127.0f);
        param->quanParameter = IDSTEncoder::encode(param->weight.data(), quantizedWeightScale, weightSize/channels, channels, false, quantizedWeight.data(), -127.0f);
        param->quanParameter->scaleIn = *scale;
        param->quanParameter->scaleOut = *scale;
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }
        param->weight.clear();
    }
}

void Revert::packMNNNet() {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, mMNNNet.get());
    builder.Finish(offset);
    mBufferSize = builder.GetSize();
    mBuffer.reset(new uint8_t[mBufferSize], std::default_delete<uint8_t[]>());
    ::memcpy(mBuffer.get(), builder.GetBufferPointer(), mBufferSize);
    mMNNNet.reset();
}

void Revert::initialize(float spasity, int sparseBlockOC, bool rewrite, bool quantizedModel) {
    if (mMNNNet->bizCode == "benchmark" || rewrite) {
        randStart();
        bool useSparse = spasity > 0.5f;
        for (auto& op : mMNNNet->oplists) {
            const auto opType = op->type;
            switch (opType) {
                case MNN::OpType_Convolution:
                case MNN::OpType_Deconvolution:
                case MNN::OpType_ConvolutionDepthwise: {
                    auto param           = op->main.AsConvolution2D();
                    auto& convCommon     = param->common;
                    const int weightReduceStride = convCommon->kernelX * convCommon->kernelY * convCommon->inputCount;
                    const int oc = convCommon->outputCount / convCommon->group;
                    param->weight.resize(oc * weightReduceStride);
                    ::memset(param->weight.data(), 0, param->weight.size() * sizeof(float));
                    param->bias.resize(convCommon->outputCount);
                    ::memset(param->bias.data(), 0, param->bias.size() * sizeof(float));
                    if (useSparse) {
                        size_t weightNNZElement, weightBlockNumber = 0;
                        MNN::CommonCompute::fillRandValueAsSparsity(weightNNZElement, weightBlockNumber, param->weight.data(), oc, weightReduceStride, spasity, sparseBlockOC);

                        MNN::AttributeT* arg1(new MNN::AttributeT);
                        arg1->key = "sparseBlockOC";
                        arg1->i = sparseBlockOC;

                        MNN::AttributeT* arg2(new MNN::AttributeT);
                        arg2->key = "sparseBlockKernel";
                        arg2->i = 1;

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
                        MNN::SparseAlgo prune_algo_type;
                        if (sparseBlockOC == 4) {
                            prune_algo_type = MNN::SparseAlgo_SIMD_OC;
                        } else {
                            prune_algo_type = MNN::SparseAlgo_RANDOM;
                        }
                        auto sparseCom = MNN::CreateSparseCommon(builder, prune_algo_type, sparseArgs);
                        builder.Finish(sparseCom);
                        auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();
                        param->sparseParameter.reset(sparseComPtr);
                        MNN::CommonCompute::compressFloatWeightToSparse(op.get());
                    }
                    break;
                }
                case MNN::OpType_Scale: {
                    auto param = op->main.AsScale();
                    param->biasData.resize(param->channels);
                    param->scaleData.resize(param->channels);
                    fillRandValue(param->scaleData.data(), param->channels);
                    fillRandValue(param->biasData.data(), param->channels);
                    break;
                }
                default:
                    break;
            }
        }
    }
    if (quantizedModel) {
        int opsize =  mMNNNet->oplists.size();
        std::vector<float> scale(opsize);
        for (int i = 0;i < opsize; ++i) {
            scale[i] = ((i + 1) / (opsize + 100.0f));
        }
        float offset = 0;
        writeExtraDescribeTensor(scale.data(), &offset);
    }
    packMNNNet();
}

void Revert::fillRandValue(float * data, size_t size) {
    unsigned int seed = 1000;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform_dist(-2, 2);

    for (size_t i = 0; i < size; i++) {
        *data = uniform_dist(rng);
    }
    return;
}

void Revert::randStart() {
}
