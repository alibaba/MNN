//
//  color.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include "cv/imgproc/color.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace CV {

std::pair<CV::ImageFormat, CV::ImageFormat> getSrcDstFormat(int code) {
    switch (code) {
#define CONVERT_SUFFIX(src, dst, suffix) \
        case COLOR_##src##2##dst##_##suffix: \
        return {CV::src##_##suffix, CV::dst}; \
            break;
#define CONVERT(src, dst) \
        case COLOR_##src##2##dst: \
            return {CV::src, CV::dst}; \
            break;
        // RGB -> *
        CONVERT(RGB, RGBA)
        CONVERT(RGB, BGRA)
        CONVERT(RGB, BGR)
        CONVERT(RGB, GRAY)
        CONVERT(RGB, YCrCb)
        CONVERT(RGB, YUV)
        CONVERT(RGB, XYZ)
        CONVERT(RGB, HSV)
        CONVERT(RGB, BGR555)
        CONVERT(RGB, BGR565)
        CONVERT(RGB, HSV_FULL)
        // BGR -> *
        // CONVERT(BGR, RGB) = CONVERT(RGB, BGR)
        CONVERT(BGR, GRAY)
        CONVERT(BGR, YCrCb)
        CONVERT(BGR, YUV)
        CONVERT(BGR, XYZ)
        CONVERT(BGR, HSV)
        CONVERT(BGR, BGR555)
        CONVERT(BGR, BGR565)
        CONVERT(BGR, HSV_FULL)
        // RGBA -> *
        CONVERT(RGBA, RGB)
        CONVERT(RGBA, BGR)
        CONVERT(RGBA, BGRA)
        CONVERT(RGBA, GRAY)
        // BGRA -> *
        // CONVERT(BGRA, RGB) = CONVERT(RGBA, RGB)
        // CONVERT(BGRA, BGR) = CONVERT(RGBA, BGR)
        // CONVERT(BGRA, BGRA) = CONVERT(RGBA, GRAY)
        CONVERT(BGRA, GRAY)
        // GRAY -> *
        CONVERT(GRAY, RGBA)
        CONVERT(GRAY, RGB)
        // CONVERT(GRAY, BGRA) = CONVERT(GRAY, RGBA)
        // CONVERT(GRAY, BGR) = CONVERT(GRAY, RGB)
        // YUV_NV21 -> *
        CONVERT_SUFFIX(YUV, RGB, NV21)
        CONVERT_SUFFIX(YUV, BGR, NV21)
        CONVERT_SUFFIX(YUV, RGBA, NV21)
        CONVERT_SUFFIX(YUV, BGRA, NV21)
        // YUV_NV12 -> *
        CONVERT_SUFFIX(YUV, RGB, NV12)
        CONVERT_SUFFIX(YUV, BGR, NV12)
        CONVERT_SUFFIX(YUV, RGBA, NV12)
        CONVERT_SUFFIX(YUV, BGRA, NV12)
        default:
            MNN_ASSERT(false);
    }
    return {CV::RGB, CV::RGB};
}

int format2Channel(CV::ImageFormat format) {
    switch (format) {
        case CV::RGB:
        case CV::BGR:
        case CV::YCrCb:
        case CV::YUV:
        case CV::HSV:
        case CV::XYZ:
        case CV::YUV_NV21:
        case CV::YUV_NV12:
        case CV::YUV_I420:
            return 3;
        case CV::BGR555:
        case CV::BGR565:
            return 2;
        case CV::GRAY:
            return 1;
        case CV::RGBA:
        case CV::BGRA:
            return 4;
        default:
            return 3;
    }
}

static VARP cvtImpl(VARP src, int code, int h, int w) {
    auto format = getSrcDstFormat(code);
    int oc = format2Channel(format.second);
    auto type = halide_type_of<uint8_t>();
    auto dest = Tensor::create({1, h, w, oc}, type);
    std::unique_ptr<CV::ImageProcess> process(CV::ImageProcess::create(format.first, format.second));
    process->convert(src->readMap<uint8_t>(), w, h, 0, dest);
    auto res = Express::Variable::create(Express::Expr::create(dest, true), 0);
    return _Squeeze(res, {0});
}

VARP cvtColor(VARP src, int code, int dstCn) {
    int h, w, c;
    getVARPSize(src, &h, &w, &c);
    return cvtImpl(src, code, h, w);
}

VARP cvtColorTwoPlane(VARP src1, VARP src2, int code) {
    int h, w, c;
    getVARPSize(src1, &h, &w, &c);
    auto src = _Concat({ _Reshape(src1, {-1}), _Reshape(src2, {-1})}, 0);
    return cvtImpl(src, code, h, w);
}

void demosaicing(VARP src, VARP& dst, int code, int dstCn) {
    dst = src;
    MNN_ERROR("demosaicing NOT support NOW!");
}

} // CV
} // MNN
