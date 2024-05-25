//
//  geometric.cpp
//  MNN
//
//  Created by MNN on 2021/08/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/geometric.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include <cmath>

namespace MNN {
namespace CV {

std::pair<VARP, VARP> convertMaps(VARP map1, VARP map2, int dstmap1type, bool nninterpolation) {
    // just return src map
    return  { map1, map2 };
}

Matrix getAffineTransform(const Point src[], const Point dst[]) {
    Matrix M;
    M.setPolyToPoly(src, dst, 3);
    return M;
}

Matrix invertAffineTransform(Matrix M) {
    M.invert(&M);
    return M;
}

Matrix getPerspectiveTransform(const Point src[], const Point dst[]) {
    Matrix M;
    M.setPolyToPoly(src, dst, 4);
    return M;
}

VARP getRectSubPix(VARP image, Size patchSize, Point center) {
    // apply below affine:
    // 1, 0, center_x - (width - 1) / 2
    // 0, 1, center_y - (height - 1) / 2
    Matrix M;
    M.setTranslate(center.fX - (patchSize.width - 1) / 2, center.fY - (patchSize.height - 1) / 2);
    return warpAffine(image, M, patchSize, WARP_INVERSE_MAP);
}

Matrix getRotationMatrix2D(Point center, double angle, double scale) {
    Matrix M;
    // rotete with invert equal opencv rotate
    M.setRotate(angle, center.fX, center.fY);
    M.invert(&M);
    // add scale after rotate
    M.postScale(scale, scale, center.fX, center.fY);
    return M;
}

extern std::pair<CV::ImageFormat, CV::ImageFormat> getSrcDstFormat(int code);
extern int format2Channel(CV::ImageFormat format);

VARP remap(VARP src, VARP map1, VARP map2, int interpolation, int borderMode, int borderValue) {
    int oh, ow, oc;
    getVARPSize(map1, &oh, &ow, &oc);
    // src need float, NC4HW4, dims = 4
    auto original_type = src->getInfo()->type;
    src = _Convert(_Unsqueeze(src, {0}), NC4HW4);
    src = _Cast(src, halide_type_of<float>());
    // change remap matrix to gridsmaple matrix: y = (2 * x + 1) / num - 1
    map1 = (map1 * _Scalar<float>(2) + _Scalar<float>(1)) / _Scalar<float>(ow) - _Scalar<float>(1);
    map2 = (map2 * _Scalar<float>(2) + _Scalar<float>(1)) / _Scalar<float>(oh) - _Scalar<float>(1);
    // grid need shape = {n, h, w, 2}
    auto m1info = map1->getInfo();
    auto grid = _Stack({map1, map2}, -1);
    auto ginfo = grid->getInfo();
    grid = _Unsqueeze(grid, {0});
    ginfo = grid->getInfo();
    auto method = InterpolationMethod::BILINEAR;
    if (interpolation == 0) {
        method = InterpolationMethod::NEAREST;
    }
    auto dst = _GridSample(src, grid, method);
    dst = _Squeeze(_Convert(_Cast(dst, original_type), NHWC), {0});
    auto info = dst->getInfo();
    return dst;
}

VARP resize(VARP src, Size dsize, double fx, double fy, int interpolation, int code, std::vector<float> mean, std::vector<float> norm) {
    int ih, iw, ic;
    auto type = src->getInfo()->type;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    if (!oh && !ow) {
        oh = ih * fy;
        ow = iw * fx;
    }
    fx = static_cast<float>(iw) / ow;
    fy = static_cast<float>(ih) / oh;
    ImageProcess::Config config;
    // cvtColor
    int oc = ic;
    if (code >= 0) {
        auto format = getSrcDstFormat(code);
        config.sourceFormat = format.first;
        config.destFormat = format.second;
        oc = format2Channel(format.second);
    } else {
        ImageFormat format = RGB;
        if (ic == 1) {
            format = GRAY;
        } else if (ic == 4) {
            format = RGBA;
        }
        config.sourceFormat = format;
        config.destFormat = format;
    }
    // toFloat
    auto dstType = type;
    if (!mean.empty() || !norm.empty()) {
        for (int i = 0; i < mean.size() && i < 4; i++) {
            config.mean[i] = mean[i];
        }
        for (int i = 0; i < norm.size() && i < 4; i++) {
            config.normal[i] = norm[i];
        }
        dstType = halide_type_of<float>();
    }
    config.filterType = static_cast<Filter>(interpolation);
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    auto dest = Tensor::create({1, oh, ow, oc}, dstType);
    Matrix tr;
    tr.postScale(fx, fy);
    tr.postTranslate(0.5 * (fx - 1), 0.5 * (fy - 1));
    process->setMatrix(tr);
    process->convert(src->readMap<uint8_t>(), iw, ih, 0, dest->host<uint8_t>(), ow, oh, oc, 0, dstType);
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpAffine(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue, int code, std::vector<float> mean, std::vector<float> norm) {
    int ih, iw, ic;
    auto type = src->getInfo()->type;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    // auto dest = Tensor::create({1, oh, ow, ic}, type);
    ImageProcess::Config config;
    config.filterType = flags < 3 ? static_cast<Filter>(flags) : BILINEAR;
    switch (borderMode) {
        case BORDER_CONSTANT:
            config.wrap = ZERO;
            break;
        case BORDER_REPLICATE:
            config.wrap = REPEAT;
            break;
        case BORDER_TRANSPARENT:
            config.wrap = CLAMP_TO_EDGE;
            break;
        default:
            MNN_ERROR("Don't support borderMode!");
            break;
    }
    // cvtColor
    int oc = ic;
    if (code >= 0) {
        auto format = getSrcDstFormat(code);
        config.sourceFormat = format.first;
        config.destFormat = format.second;
        oc = format2Channel(format.second);
    } else {
        ImageFormat format = RGB;
        if (ic == 1) {
            format = GRAY;
        } else if (ic == 4) {
            format = RGBA;
        }
        config.sourceFormat = format;
        config.destFormat = format;
    }
    // toFloat
    auto dstType = type;
    if (!mean.empty() || !norm.empty()) {
        for (int i = 0; i < mean.size() && i < 4; i++) {
            config.mean[i] = mean[i];
        }
        for (int i = 0; i < norm.size() && i < 4; i++) {
            config.normal[i] = norm[i];
        }
        dstType = halide_type_of<float>();
    }
    auto dest = Tensor::create({1, oh, ow, oc}, dstType);
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    if (flags != WARP_INVERSE_MAP) {
        bool invert = M.invert(&M);
        MNN_ASSERT(invert);
    }
    process->setMatrix(M);
    process->setPadding(borderValue);
    process->convert(src->readMap<uint8_t>(), iw, ih, 0, dest->host<uint8_t>(), ow, oh, oc, 0, dstType);
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpPerspective(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue) {
    return warpAffine(src, M, dsize, flags, borderMode, borderValue);
}

VARP undistortPoints(VARP src, VARP cameraMatrix, VARP distCoeffs) {
    // Don't support distCoeffs
    auto dims = src->getInfo()->dim;
    int n = dims[0];
    auto dst  = _Input(dims, NCHW);
    auto iptr = src->readMap<float>();
    auto optr = dst->writeMap<float>();
    auto cptr = cameraMatrix->readMap<float>();
    double fx = cptr[0];
    double fy = cptr[4];
    double ifx = 1./fx;
    double ify = 1./fy;
    double cx = cptr[2];
    double cy = cptr[5];
    for (int i = 0; i < n; i++) {
        auto x = iptr[i * 2], y = iptr[i * 2 + 1];
        auto u = x;
        auto v = y;
        x = (x - cx)*ifx;
        y = (y - cy)*ify;
        optr[i * 2] = x;
        optr[i * 2 + 1] = y;
    }
    return dst;
}

} // CV
} // MNN