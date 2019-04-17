//
//  revertMNNModel.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#include <string.h>
#include "MNNDefine.h"
#include "revertMNNModel.hpp"

const float MIN_VALUE = -2.0;
const float MAX_VALUE = 2.0;

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

void Revert::packMNNNet() {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, mMNNNet.get());
    builder.Finish(offset);
    mBufferSize = builder.GetSize();
    mBuffer.reset(new uint8_t[mBufferSize]);
    ::memcpy(mBuffer.get(), builder.GetBufferPointer(), mBufferSize);
    mMNNNet.reset();
}

void Revert::initialize() {
    if (mMNNNet->bizCode == "benchmark") {
        randStart();
        for (auto& op : mMNNNet->oplists) {
            const auto opType = op->type;
            switch (opType) {
                case MNN::OpType_Convolution:
                case MNN::OpType_Deconvolution:
                case MNN::OpType_ConvolutionDepthwise: {
                    auto param           = op->main.AsConvolution2D();
                    auto& convCommon     = param->common;
                    const int weightSize = convCommon->kernelX * convCommon->kernelY * convCommon->outputCount *
                                           convCommon->inputCount / convCommon->group;
                    param->weight.resize(weightSize);
                    ::memset(param->weight.data(), 0, param->weight.size() * sizeof(float));
                    param->bias.resize(convCommon->outputCount);
                    ::memset(param->bias.data(), 0, param->bias.size() * sizeof(float));
                    break;
                }
                case MNN::OpType_Scale: {
                    auto param = op->main.AsScale();
                    param->biasData.resize(param->channels);
                    param->scaleData.resize(param->channels);
                    for (int i = 0; i < param->channels; ++i) {
                        param->scaleData[i] = getRandValue();
                        param->biasData[i]  = getRandValue();
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    packMNNNet();
}

float Revert::getRandValue() {
    return MIN_VALUE + (MAX_VALUE - MIN_VALUE) * rand() / RAND_MAX;
}

void Revert::randStart() {
    srand((unsigned)time(NULL));
}
