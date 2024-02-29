//
//  miscellaneous.cpp
//  MNN
//
//  Created by MNN on 2021/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/miscellaneous.hpp"
#include "cv/imgproc/filter.hpp"
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>

namespace MNN {
namespace CV {

VARP adaptiveThreshold(VARP src, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
    auto origin_type = src->getInfo()->type;
    src = _Cast<float>(src);
    // get threshold
    VARP threshold;
    if (adaptiveMethod == ADAPTIVE_THRESH_MEAN_C) {
        threshold = boxFilter(src, -1, {blockSize, blockSize}, true, REFLECT);
    } else {
        threshold = GaussianBlur(src, {blockSize, blockSize}, 0, 0, REFLECT);
    }
    threshold = _Unsqueeze(threshold, {-1}) - _Scalar<float>(C);

    VARP dst;
    maxValue = maxValue > 255 ? 255 : maxValue;
    maxValue = maxValue < 0 ? 0 : maxValue;
    auto maxval = _Scalar<float>(maxValue);
    if (thresholdType == THRESH_BINARY) {
        dst = _Cast<float>(_Greater(src, threshold)) * maxval;
    } else {
        dst = _Cast<float>(_LessEqual(src, threshold)) * maxval;
    }
    return _Cast(dst, origin_type);
}

VARP blendLinear(VARP src1, VARP src2, VARP weight1, VARP weight2) {
    auto inputInfo = src1->getInfo();
    auto origin_type = inputInfo->type;
    if (origin_type.code != halide_type_float) {
        src1 = _Cast<float>(src1);
        src2 = _Cast<float>(src2);
        weight1 = _Cast<float>(weight1);
        weight2 = _Cast<float>(weight2);
        return _Cast((src1 * weight1 + src2 * weight2) / (weight1 + weight2 + _Scalar<float>(1e-5)), origin_type);
    }
    
    return _Cast((src1 * weight1 + src2 * weight2) / (weight1 + weight2 + _Scalar<float>(1e-5)), origin_type);
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
    auto origin_type = src->getInfo()->type;
    src = _Cast(src, halide_type_of<float>());
    auto mask = _Threshold(src, thresh);
    VARP dst;
    switch (type) {
        case THRESH_BINARY:
            dst = mask * _Scalar<float>(maxval);
            break;
        case THRESH_BINARY_INV:
            dst = (_Scalar<float>(1.f) - mask) * _Scalar<float>(maxval);
            break;
        case THRESH_TRUNC:
            dst = mask * _Scalar<float>(thresh) + (_Scalar<float>(1.f) - mask) * src;
            break;
        case THRESH_TOZERO:
            dst = mask * src;
            break;
        case THRESH_TOZERO_INV:
            dst = (_Scalar<float>(1.f) - mask) * src;
            break;
        case THRESH_MASK:
        case THRESH_OTSU:
        case THRESH_TRIANGLE:
            MNN_ERROR("Don't support THRESH_MASK/THRESH_OTSU/THRESH_TRIANGLE.");
            break;
        default:
            break;
    }
    return _Cast(dst, origin_type);
}

} // CV
} // MNN
