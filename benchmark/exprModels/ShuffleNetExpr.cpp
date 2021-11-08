//
//  ShuffleNetExpr.cpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1707.01083.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include "ShuffleNetExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

// bottleNeckChannel = outputChannel / narrowRatio
static VARP shuffleUnit(VARP x, int inputChannel, int outputChannel,
                        int group, int stride, int narrowRatio) {
    int bottleNeckChannel = outputChannel / narrowRatio;
    int branchChannel = outputChannel;
    if (stride != 1) {
        branchChannel = outputChannel - inputChannel;
    }
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, bottleNeckChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, group); // Group Conv
    y = _ChannelShuffle(y, group);
    y = _Conv(0.0f, 0.0f, y, {bottleNeckChannel, bottleNeckChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, bottleNeckChannel); // Depthwise Conv
    y = _Conv(0.0f, 0.0f, y, {bottleNeckChannel, branchChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, group); // Group Conv
    if (stride != 1) {
        x = _AvePool(x, {3, 3}, {2, 2}, SAME);
        y = _Concat({x, y}, 1); // concat on channel axis (NCHW)
    } else {
        y = _Add(x, y);
    }
    return y;
}

static VARP shuffleBlock(VARP x, int inputChannel, int outputChannel,
                         int group, int stride, int narrowRatio, int number) {
    x = shuffleUnit(x, inputChannel, outputChannel, group, stride, narrowRatio);
    for (int i = 1; i < number; ++i) {
        x = shuffleUnit(x, outputChannel, outputChannel, group, 1, narrowRatio);
    }
    return x;
}

VARP shuffleNetExpr(int group, int numClass) {
    std::map<int, std::vector<int>> groupMap = {
        {1, {144, 288, 576}},
        {2, {200, 400, 800}},
        {3, {240, 480, 960}},
        {4, {272, 544, 1088}},
        {8, {384, 768, 1536}}
    };
    int outputChannelStage2 = groupMap[group][0];
    int outputChannelStage3 = groupMap[group][1];
    int outputChannelStage4 = groupMap[group][2];
    // Net Construction
    auto x = _Input({1, 3, 224, 224}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, 24}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = shuffleBlock(x, 24, outputChannelStage2, group, 2, 4, 4);
    x = shuffleBlock(x, outputChannelStage2, outputChannelStage3, group, 2, 4, 8);
    x = shuffleBlock(x, outputChannelStage3, outputChannelStage4, group, 2, 4, 4);
    x = _AvePool(x, {7, 7}, {1, 1}, VALID);
    x = _Conv(0.0f, 0.0f, x, {outputChannelStage4, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // replace FC with Conv1x1
    return x;
}
