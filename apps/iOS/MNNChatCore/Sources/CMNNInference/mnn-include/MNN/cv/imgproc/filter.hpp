//
//  filter.hpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FILTER_HPP
#define FILTER_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "../types.hpp"

namespace MNN {
namespace CV {

MNN_PUBLIC VARP bilateralFilter(VARP src, int d, double sigmaColor, double sigmaSpace,
                                int borderType = REFLECT);

MNN_PUBLIC VARP blur(VARP src, Size ksize, int borderType = REFLECT);

MNN_PUBLIC VARP boxFilter(VARP src, int ddepth, Size ksize,
                          bool normalize = true, int borderType = REFLECT);

MNN_PUBLIC VARP dilate(VARP src, VARP kernel,
                       int iterations = 1, int borderType = CONSTANT);

MNN_PUBLIC VARP erode(VARP src, VARP kernel,
                      int iterations = 1, int borderType = CONSTANT);

MNN_PUBLIC VARP filter2D(VARP src, int ddepth, VARP kernel,
                         double delta = 0, int borderType = REFLECT);

MNN_PUBLIC VARP GaussianBlur(VARP src, Size ksize, double sigmaX,
                             double sigmaY = 0, int borderType = REFLECT);

MNN_PUBLIC std::pair<VARP, VARP> getDerivKernels(int dx, int dy, int ksize,
                                                 bool normalize = false);

MNN_PUBLIC VARP getGaborKernel(Size ksize, double sigma, double theta, double lambd,
                               double gamma, double psi = MNN_PI * 0.5);

MNN_PUBLIC VARP getGaussianKernel(int n, double sigma);

MNN_PUBLIC VARP getStructuringElement(int shape, Size ksize);

MNN_PUBLIC VARP Laplacian(VARP src, int ddepth, int ksize = 1,
                          double scale = 1, double delta = 0, int borderType = REFLECT);

MNN_PUBLIC VARP pyrDown(VARP src, Size dstsize = {}, int borderType = REFLECT);

MNN_PUBLIC VARP pyrUp(VARP src, Size dstsize = {}, int borderType = REFLECT);

MNN_PUBLIC VARP Scharr(VARP src, int ddepth, int dx, int dy,
                       double scale = 1, double delta = 0, int borderType = REFLECT);

MNN_PUBLIC VARP sepFilter2D(VARP src, int ddepth, VARP& kernelX, VARP& kernelY,
                            double delta = 0, int borderType = REFLECT);

MNN_PUBLIC VARP Sobel(VARP src, int ddepth, int dx, int dy, int ksize = 3,
                      double scale = 1, double delta = 0, int borderType = REFLECT);

MNN_PUBLIC std::pair<VARP, VARP> spatialGradient(VARP src, int ksize = 3,
                                                 int borderType = REFLECT);

MNN_PUBLIC VARP sqrBoxFilter(VARP src, int ddepth, Size ksize,
                             bool normalize = true, int borderType = REFLECT);
} // CV
} // MNN
#endif // FILTER_HPP
