//
//  geometric.hpp
//  MNN
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GEOMETRIC_HPP
#define GEOMETRIC_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/ImageProcess.hpp>
#include "../types.hpp"

namespace MNN {
namespace CV {

enum InterpolationFlags {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_CUBIC = 2,
    INTER_AREA = 3,
    INTER_LANCZOS4 = 4,
    INTER_LINEAR_EXACT = 5,
    INTER_NEAREST_EXACT = 6,
    INTER_MAX = 7,
    WARP_FILL_OUTLIERS = 8,
    WARP_INVERSE_MAP = 16
};

enum BorderTypes {
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_REFLECT101 = BORDER_REFLECT_101,
    BORDER_DEFAULT = BORDER_REFLECT_101,
    BORDER_ISOLATED = 16
};

MNN_PUBLIC std::pair<VARP, VARP> convertMaps(VARP map1, VARP map2, int dstmap1type,
                                             bool interpolation = false);

MNN_PUBLIC Matrix getAffineTransform(const Point src[], const Point dst[]);

MNN_PUBLIC Matrix getPerspectiveTransform(const Point src[], const Point dst[]);

MNN_PUBLIC VARP getRectSubPix(VARP image, Size patchSize, Point center);

MNN_PUBLIC Matrix getRotationMatrix2D(Point center, double angle, double scale);

MNN_PUBLIC Matrix invertAffineTransform(Matrix M);

MNN_PUBLIC VARP remap(VARP src, VARP map1, VARP map2, int interpolation, int borderMode = BORDER_CONSTANT, int borderValue = 0);

MNN_PUBLIC VARP resize(VARP src, Size dsize, double fx = 0, double fy = 0,
                       int interpolation = INTER_LINEAR, int code = -1,
                       std::vector<float> mean = {}, std::vector<float> norm = {});

MNN_PUBLIC VARP warpAffine(VARP src, Matrix M, Size dsize,
                           int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, int borderValue = 0,
                           int code = -1, std::vector<float> mean = {}, std::vector<float> norm = {});

MNN_PUBLIC VARP warpPerspective(VARP src, Matrix M, Size dsize,
                                int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                                int borderValue = 0);

MNN_PUBLIC VARP undistortPoints(VARP src, VARP cameraMatrix, VARP distCoeffs);
} // CV
} // MNN
#endif // GEOMETRIC_HPP
