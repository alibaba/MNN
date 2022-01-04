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

namespace MNN {
namespace CV {

std::pair<VARP, VARP> convertMaps(VARP map1, VARP map2, int dstmap1type, bool nninterpolation) {
    MNN_ERROR("convertMaps NOT support NOW!");
    VARP dstmap1 = _Cast<float>(map1);
    VARP dstmap2 = _Cast<float>(map2);
    return  { dstmap1, dstmap2 };
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

VARP resize(VARP src, Size dsize, double fx, double fy, int interpolation) {
    int ih, iw, ic;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    if (!oh && !ow) {
        oh = ih * fy;
        ow = iw * fx;
    }
    fx = static_cast<float>(iw) / ow;
    fy = static_cast<float>(ih) / oh;
    auto dest = Tensor::create({1, oh, ow, ic}, halide_type_of<uint8_t>());
    ImageProcess::Config config;
    config.filterType = static_cast<Filter>(interpolation);
    config.sourceFormat = RGB;
    config.destFormat = RGB;
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    Matrix tr;
    tr.postScale(fx, fy);
    tr.postTranslate(0.5 * (fx - 1), 0.5 * (fy - 1));
    process->setMatrix(tr);
    process->convert(src->readMap<uint8_t>(), iw, ih, 0, dest->host<uint8_t>(), ow, oh, ic, 0, halide_type_of<uint8_t>());
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpAffine(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue) {
    int ih, iw, ic;
    getVARPSize(src, &ih, &iw, &ic);
    int oh = dsize.height, ow = dsize.width;
    auto dest = Tensor::create({1, oh, ow, ic}, halide_type_of<uint8_t>());
    ImageProcess::Config config;
    config.filterType = flags < 3 ? static_cast<Filter>(flags) : BILINEAR;
    config.sourceFormat = RGB;
    config.destFormat = RGB;
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
    std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
    if (flags != WARP_INVERSE_MAP) {
        bool invert = M.invert(&M);
        MNN_ASSERT(invert);
    }
    process->setMatrix(M);
    process->setPadding(borderValue);
    process->convert(src->readMap<uint8_t>(), iw, ih, 0, dest->host<uint8_t>(), ow, oh, ic, 0, halide_type_of<uint8_t>());
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP warpPerspective(VARP src, Matrix M, Size dsize, int flags, int borderMode, int borderValue) {
    return warpAffine(src, M, dsize, flags, borderMode, borderValue);
}

} // CV
} // MNN
