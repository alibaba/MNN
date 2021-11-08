//
//  resnetExpr.cpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1512.03385.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <vector>
#include "ResNetExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

// When we use MNNConverter to convert other resnet model to MNN model,
// {Conv + BN + Relu} will be converted and optimized to {Conv}
static VARP residual(VARP x, INTS channels, int stride) {
    int inputChannel = x->getInfo()->dim[1], outputChannel = channels[1];
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {outputChannel, outputChannel}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    if (inputChannel != outputChannel || stride != 1) {
        x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    }
    y = _Add(x, y);
    return y;
}

static VARP residualBlock(VARP x, INTS channels, int stride, int number) {
    x = residual(x, {channels[0], channels[1]}, stride);
    for (int i = 1; i < number; ++i) {
        x = residual(x, {channels[1], channels[1]}, 1);
    }
    return x;
}

static VARP bottleNeck(VARP x, INTS channels, int stride) {
    int inputChannel = x->getInfo()->dim[1], narrowChannel = channels[1], outputChannel = channels[2];
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, narrowChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {narrowChannel, narrowChannel}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {narrowChannel, outputChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    if (inputChannel != outputChannel || stride != 1) {
        x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    }
    y = _Add(x, y);
    return y;
}

static VARP bottleNeckBlock(VARP x, INTS channels, int stride, int number) {
    x = bottleNeck(x, {channels[0], channels[1], channels[2]}, stride);
    for (int i = 1; i < number; ++i) {
        x = bottleNeck(x, {channels[2], channels[1], channels[2]}, 1);
    }
    return x;
}

VARP resNetExpr(ResNetType resNetType, int numClass) {
    std::vector<int> numbers;
    {
        auto numbersMap = std::map<ResNetType, std::vector<int>>({
            {ResNet18, {2, 2, 2, 2}},
            {ResNet34, {3, 4, 6, 3}},
            {ResNet50, {3, 4, 6, 3}},
            {ResNet101, {3, 4, 23, 3}},
            {ResNet152, {3, 8, 36, 3}}
        });
        if (numbersMap.find(resNetType) == numbersMap.end()) {
            MNN_ERROR("resNetType (%d) not support, only support [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]\n", resNetType);
            return VARP(nullptr);
        }
        numbers = numbersMap[resNetType];
    }
    std::vector<int> channels({64, 64, 128, 256, 512});
    {
        if (resNetType != ResNet18 && resNetType != ResNet34) {
            channels[0] = 16;
        }
    }
    std::vector<int> strides({1, 2, 2, 2});
    int finalChannel = channels[4] * 4;
    auto x = _Input({1, 3, 224, 224}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, 64}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    for (int i = 0; i < 4; ++i) {
        if (resNetType == ResNet18 || resNetType == ResNet34) {
            x = residualBlock(x, {channels[i], channels[i+1]}, strides[i], numbers[i]);
        } else {
            x = bottleNeckBlock(x, {channels[i] * 4, channels[i+1], channels[i+1] * 4}, strides[i], numbers[i]);
        }
    }
    x = _AvePool(x, {7, 7}, {1, 1}, VALID);
    x = _Conv(0.0f, 0.0f, x, {x->getInfo()->dim[1], numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    x = _Softmax(x, -1);
    return x;
}
