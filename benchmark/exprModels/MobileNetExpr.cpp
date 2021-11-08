//
//  mobileNetExpr.cpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1704.04861.pdf https://arxiv.org/pdf/1801.04381.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include "MobileNetExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

// When we use MNNConverter to convert other mobilenet model to MNN model,
// {Conv3x3Depthwise + BN + Relu + Conv1x1 + BN + Relu} will be converted
// and optimized to {Conv3x3Depthwise + Conv1x1}
static VARP convBlock(VARP x, INTS channels, int stride) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int group = inputChannel;
    x = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
    x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}

VARP mobileNetV1Expr(MobileNetWidthType alpha, MobileNetResolutionType beta, int numClass) {
    int inputSize, poolSize; // MobileNet_224, MobileNet_192, MobileNet_160, MobileNet_128
    {
        auto inputSizeMap = std::map<MobileNetResolutionType, int>({
            {MobileNet_224, 224},
            {MobileNet_192, 192},
            {MobileNet_160, 160},
            {MobileNet_128, 128}
        });
        if (inputSizeMap.find(beta) == inputSizeMap.end()) {
            MNN_ERROR("MobileNetResolutionType (%d) not support, only support [MobileNet_224, MobileNet_192, MobileNet_160, MobileNet_128]\n", beta);
            return VARP(nullptr);
        }
        inputSize = inputSizeMap[beta];
        poolSize = inputSize / 32;
    }

    int channels[6]; // MobileNet_100, MobileNet_075, MobileNet_050, MobileNet_025
    {
        auto channelsMap = std::map<MobileNetWidthType, int>({
            {MobileNet_100, 32},
            {MobileNet_075, 24},
            {MobileNet_050, 16},
            {MobileNet_025, 8}
        });
        if (channelsMap.find(alpha) == channelsMap.end()) {
            MNN_ERROR("MobileNetWidthType (%d) not support, only support [MobileNet_100, MobileNet_075, MobileNet_050, MobileNet_025]\n", alpha);
            return VARP(nullptr);
        }
        channels[0] = channelsMap[alpha];
    }

    for (int i = 1; i < 6; ++i) {
        channels[i] = channels[0] * (1 << i);
    }

    auto x = _Input({1, 3, inputSize, inputSize}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, channels[0]}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
    x = convBlock(x, {channels[0], channels[1]}, 1);
    x = convBlock(x, {channels[1], channels[2]}, 2);
    x = convBlock(x, {channels[2], channels[2]}, 1);
    x = convBlock(x, {channels[2], channels[3]}, 2);
    x = convBlock(x, {channels[3], channels[3]}, 1);
    x = convBlock(x, {channels[3], channels[4]}, 2);
    x = convBlock(x, {channels[4], channels[4]}, 1);
    x = convBlock(x, {channels[4], channels[4]}, 1);
    x = convBlock(x, {channels[4], channels[4]}, 1);
    x = convBlock(x, {channels[4], channels[4]}, 1);
    x = convBlock(x, {channels[4], channels[4]}, 1);
    x = convBlock(x, {channels[4], channels[5]}, 2);
    x = convBlock(x, {channels[5], channels[5]}, 1);
    x = _AvePool(x, {poolSize, poolSize}, {1, 1}, VALID);
    x = _Conv(0.0f, 0.0f, x, {channels[5], numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    x = _Softmax(x, -1);
    return x;
}

static VARP bottleNeck(VARP x, INTS channels, int stride, int expansionRatio) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int expansionChannel = inputChannel * expansionRatio, group = expansionChannel;
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, expansionChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {expansionChannel, expansionChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
    y = _Conv(0.0f, 0.0f, y, {expansionChannel, outputChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    if (inputChannel != outputChannel || stride != 1) {
        x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    }
    y = _Add(x, y);
    return y;
}

static VARP bottleNeckBlock(VARP x, INTS channels, int stride, int expansionRatio, int number) {
    x = bottleNeck(x, {channels[0], channels[1]}, stride, expansionRatio);
    for (int i = 1; i < number; ++i) {
        x = bottleNeck(x, {channels[1], channels[1]}, 1, expansionRatio);
    }
    return x;
}

VARP mobileNetV2Expr(int numClass) {
    auto x = _Input({1, 3, 224, 224}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, 32}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
    x = bottleNeckBlock(x,   {32, 16}, 1, 1, 1);
    x = bottleNeckBlock(x,   {16, 24}, 2, 6, 2);
    x = bottleNeckBlock(x,   {24, 32}, 2, 6, 3);
    x = bottleNeckBlock(x,   {32, 64}, 2, 6, 4);
    x = bottleNeckBlock(x,   {64, 96}, 1, 6, 3);
    x = bottleNeckBlock(x,  {96, 160}, 2, 6, 3);
    x = bottleNeckBlock(x, {160, 320}, 1, 6, 1);
    x = _Conv(0.0f, 0.0f, x, {320, 1280}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    x = _AvePool(x, {7, 7}, {1, 1}, VALID);
    x = _Conv(0.0f, 0.0f, x, {1280, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    return x;
}
