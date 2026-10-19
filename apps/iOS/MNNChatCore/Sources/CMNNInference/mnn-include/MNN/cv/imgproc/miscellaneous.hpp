//
//  miscellaneous.hpp
//  MNN
//
//  Created by MNN on 2021/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MISCELLANEOUS_HPP
#define MISCELLANEOUS_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace CV {
using namespace Express;

enum AdaptiveThresholdTypes {
    ADAPTIVE_THRESH_MEAN_C     = 0,
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
};

enum ThresholdTypes {
  THRESH_BINARY = 0,
  THRESH_BINARY_INV = 1,
  THRESH_TRUNC = 2,
  THRESH_TOZERO = 3,
  THRESH_TOZERO_INV = 4,
  THRESH_MASK = 7,
  THRESH_OTSU = 8,
  THRESH_TRIANGLE = 16
};

MNN_PUBLIC VARP adaptiveThreshold(VARP src, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C);

MNN_PUBLIC VARP blendLinear(VARP src1, VARP src2, VARP weight1, VARP weight2);

MNN_PUBLIC void distanceTransform(VARP src, VARP& dst, VARP& labels, int distanceType, int maskSize, int labelType = 0);

MNN_PUBLIC int floodFill(VARP image, std::pair<int, int> seedPoint, float newVal);

MNN_PUBLIC VARP integral(VARP src, int sdepth = -1);

MNN_PUBLIC VARP threshold(VARP src, double thresh, double maxval, int type);

} // CV
} // MNN
#endif // MISCELLANEOUS_HPP
