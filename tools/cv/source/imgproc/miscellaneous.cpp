//
//  miscellaneous.cpp
//  MNN
//
//  Created by MNN on 2021/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/miscellaneous.hpp"
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace CV {

VARP adaptiveThreshold(VARP src, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
    // TODO
    return nullptr;
}

VARP blendLinear(VARP src1, VARP src2, VARP weight1, VARP weight2) {
    return (src1 * weight1 + src2 * weight2) / (weight1 + weight2 + _Scalar<float>(1e-5));
}

void distanceTransform(VARP src, VARP& dst, VARP& labels, int distanceType, int maskSize, int labelType) {
    // TODO
}

int floodFill(VARP image, std::pair<int, int> seedPoint, float newVal) {
    // TODO
    return 0;
}

VARP integral(VARP src, VARP& sum, int sdepth) {
    // TODO
    return nullptr;
}

VARP threshold(VARP src, double thresh, double maxval, int type) {
    auto mask = _Threshold(src, thresh);
    switch (type) {
        case THRESH_BINARY:
            return mask * _Scalar<float>(maxval);
        case THRESH_BINARY_INV:
            return (_Scalar<float>(1.f) - mask) * _Scalar<float>(maxval);
        case THRESH_TRUNC:
            return mask * _Scalar<float>(thresh) + (_Scalar<float>(1.f) - mask) * src;
        case THRESH_TOZERO:
            return mask * src;
        case THRESH_TOZERO_INV:
            return (_Scalar<float>(1.f) - mask) * src;
        case THRESH_MASK:
        case THRESH_OTSU:
        case THRESH_TRIANGLE:
            MNN_ERROR("Don't support THRESH_MASK/THRESH_OTSU/THRESH_TRIANGLE.");
            break;
        default:
            break;
    }
    return nullptr;
}

} // CV
} // MNN
