//
//  googLeNetExpr.cpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1409.4842.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GoogLeNetExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

// inception module, channels: [inputChannel, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj]
static VARP inception(VARP x, int inputChannelSet, int channel_1x1,
                      int channel_3x3_reduce, int channel_3x3,
                      int channel_5x5_reduce, int channel_5x5,
                      int channel_pool) {
    auto inputChannel = x->getInfo()->dim[1];
    auto y1 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y2 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_3x3_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    y2 = _Conv(0.0f, 0.0f, y2, {channel_3x3_reduce, channel_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    auto y3 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_5x5_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    y3 = _Conv(0.0f, 0.0f, y3, {channel_5x5_reduce, channel_5x5}, {5, 5}, SAME, {1, 1}, {1, 1}, 1);
    auto y4 = _MaxPool(x, {3, 3}, {1, 1}, SAME);
    y4 = _Conv(0.0f, 0.0f, y4, {inputChannel, channel_pool}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    return _Concat({y1, y2, y3, y4}, 1); // concat on channel axis (NCHW)
}

VARP googLeNetExpr(int numClass) {
    auto x = _Input({1, 3, 224, 224}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, 64}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = _Conv(0.0f, 0.0f, x, {64, 192}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = inception(x, 192, 64, 96, 128, 16, 32, 32);
    x = inception(x, 256, 128, 128, 192, 32, 96, 64);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = inception(x, 480, 192, 96, 208, 16, 48, 64);
    x = inception(x, 512, 160, 112, 224, 24, 64, 64);
    x = inception(x, 512, 128, 128, 256, 24, 64, 64);
    x = inception(x, 512, 112, 144, 288, 32, 64, 64);
    x = inception(x, 512, 256, 160, 320, 32, 128, 128);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = inception(x, 832, 256, 160, 320, 32, 128, 128);
    x = inception(x, 832, 384, 192, 384, 48, 128, 128);
    x = _AvePool(x, {7, 7}, {1, 1}, VALID);
    x = _Conv(0.0f, 0.0f, x, {1024, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // replace FC with Conv1x1
    x = _Softmax(x, -1);
    return x;
}
