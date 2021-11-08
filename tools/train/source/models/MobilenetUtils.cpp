//
//  MobilenetUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MobilenetUtils.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
int makeDivisible(int v, int divisor, int minValue) {
    if (minValue == 0) {
        minValue = divisor;
    }
    int newV = std::max(minValue, int(v + divisor / 2) / divisor * divisor);

    // Make sure that round down does not go down by more than 10%.
    if (newV < 0.9 * v) {
        newV += divisor;
    }

    return newV;
}

} // namespace Model
} // namespace Train
} // namespace MNN
