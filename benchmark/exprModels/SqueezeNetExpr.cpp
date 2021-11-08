//
//  SqueezeNetExpr.cpp
//  MNN
//  Reference paper: https://arxiv.org/pdf/1602.07360.pdf
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SqueezeNetExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

// fire module in squeezeNet model
static VARP fireMoudle(VARP x, int inputChannel, int squeeze_1x1,
                       int expand_1x1, int expand_3x3) {
    x = _Conv(0.0f, 0.0f, x, {inputChannel, squeeze_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y1 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y2 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    return _Concat({y1, y2}, 1); // concat on channel axis (NCHW)
}

VARP squeezeNetExpr(int numClass) {
    auto x = _Input({1, 3, 224, 224}, NC4HW4);
    x = _Conv(0.0f, 0.0f, x, {3, 96}, {7, 7}, SAME, {2, 2}, {1, 1}, 1);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = fireMoudle(x, 96, 16, 64, 64);
    x = fireMoudle(x, 128, 16, 64, 64);
    x = fireMoudle(x, 128, 32, 128, 128);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = fireMoudle(x, 256, 32, 128, 128);
    x = fireMoudle(x, 256, 48, 192, 192);
    x = fireMoudle(x, 384, 48, 192, 192);
    x = fireMoudle(x, 384, 64, 256, 256);
    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
    x = fireMoudle(x, 512, 64, 256, 256);
    x = _Conv(0.0f, 0.0f, x, {512, numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    x = _AvePool(x, {14, 14}, {1, 1}, VALID);
    return x;
}
