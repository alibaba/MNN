//
//  ImageProcessUtils.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <map>
#include "ImageProcessUtils.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "backend/cpu/CPUImageProcess.hpp"
#include "backend/cpu/compute/ImageProcessFunction.hpp"
#include <MNN/MNNForwardType.h>
#include "core/Backend.hpp"
#include <MNN/ImageProcess.hpp>

#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Execution.hpp"
#include "core/Backend.hpp"
#include "MNN_generated.h"

#ifdef _MSC_VER
#include "backend/cpu/x86_x64/cpu_id.h"
#endif

#define CACHE_SIZE 256
namespace MNN {
using namespace CV;
#define CHECKFORMAT(src, dst, func) if (source == src && dest == dst) return func
#define CHECKFORMAT_CORE(src, dst, func) if (source == src && dest == dst) return coreFunctions ? coreFunctions->func : func;

void registerBackend();
struct ImageProcessUtils::InsideProperty {
    CV::ImageProcess::Config config;
    // Image Format convert parameters.
    bool mDraw = false;
    // Image parameters.
    halide_type_t mDtype;
    int mStride = 0;
    int oc, oh, ow, ic, ih, iw;
    // Process functions.
    BLIT_FLOAT mBlitFloat = nullptr;
    BLITTER mBlitter = nullptr;
    SAMPLER mSampler = nullptr;
};
void ImageProcessUtils::destroy(ImageProcessUtils* pro) {
    if (nullptr != pro) {
        delete pro;
    }
}

ImageProcessUtils::~ImageProcessUtils() {
    delete mInside;
}

ImageProcessUtils::ImageProcessUtils(const CV::ImageProcess::Config& config, CoreFunctions* core) {
    mInside         = new InsideProperty;
    mInside->config = config;
    for (int i = 0; i < 4; ++i) {
        mInside->config.mean[i]   = config.mean[i];
        mInside->config.normal[i] = config.normal[i];
    }
    coreFunctions = core;
}

BLITTER ImageProcessUtils::choose(ImageFormat source, ImageFormat dest) {
    // YUV only different in sampler
    if ((ImageFormatType)source == ImageFormatType_YUV_NV12) {
        source = (ImageFormat)ImageFormatType_YUV_NV21;
    }
    if ((ImageFormatType)source == ImageFormatType_YUV_I420) {
        source = (ImageFormat)ImageFormatType_YUV_NV21;
    }
    CHECKFORMAT(RGBA, RGBA, MNNCopyC4);
    CHECKFORMAT_CORE(RGBA, BGRA, MNNRGBAToBGRA);
    CHECKFORMAT(RGBA, BGR, MNNRGBAToBGR);
    CHECKFORMAT(RGBA, RGB, MNNBGRAToBGR);
    CHECKFORMAT(RGBA, GRAY, MNNRGBAToGRAY);

    CHECKFORMAT_CORE(BGRA, RGBA, MNNRGBAToBGRA);
    CHECKFORMAT(BGRA, BGRA, MNNCopyC4);
    CHECKFORMAT(BGRA, BGR, MNNBGRAToBGR);
    CHECKFORMAT(BGRA, RGB, MNNRGBAToBGR);
    CHECKFORMAT(BGRA, GRAY, MNNBGRAToGRAY);

    CHECKFORMAT(RGB, RGB, MNNCopyC3);
    CHECKFORMAT(RGB, BGR, MNNRGBToBGR);
    CHECKFORMAT(RGB, GRAY, MNNRGBToGRAY);
    CHECKFORMAT(RGB, RGBA, MNNC3ToC4);
    CHECKFORMAT(RGB, YCrCb, MNNRGBToCrCb);
    CHECKFORMAT(RGB, YUV, MNNRGBToYUV);
    CHECKFORMAT(RGB, XYZ, MNNRGBToXYZ);
    CHECKFORMAT(RGB, HSV, MNNRGBToHSV);
    CHECKFORMAT(RGB, BGR555, MNNRGBToBGR555);
    CHECKFORMAT(RGB, BGR565, MNNRGBToBGR565);
    CHECKFORMAT(RGB, HSV_FULL, MNNRGBToHSV_FULL);

    CHECKFORMAT(BGR, BGR, MNNCopyC3);
    CHECKFORMAT(BGR, RGB, MNNRGBToBGR);
    CHECKFORMAT(BGR, GRAY, MNNBRGToGRAY);
    CHECKFORMAT(BGR, BGRA, MNNC3ToC4);
    CHECKFORMAT(BGR, YCrCb, MNNBGRToCrCb);
    CHECKFORMAT(BGR, YUV, MNNBGRToYUV);
    CHECKFORMAT(BGR, XYZ, MNNBGRToXYZ);
    CHECKFORMAT(BGR, HSV, MNNBGRToHSV);
    CHECKFORMAT(BGR, BGR555, MNNBGRToBGR555);
    CHECKFORMAT(BGR, BGR565, MNNBGRToBGR565);
    CHECKFORMAT(BGR, HSV_FULL, MNNBGRToHSV_FULL);

    CHECKFORMAT(GRAY, RGBA, MNNGRAYToC4);
    CHECKFORMAT(GRAY, BGRA, MNNGRAYToC4);
    CHECKFORMAT(GRAY, BGR, MNNGRAYToC3);
    CHECKFORMAT(GRAY, RGB, MNNGRAYToC3);
    CHECKFORMAT(GRAY, GRAY, MNNCopyC1);

    CHECKFORMAT(YUV_NV21, GRAY, MNNCopyC1);
    CHECKFORMAT_CORE(YUV_NV21, RGB, MNNNV21ToRGB);
    CHECKFORMAT_CORE(YUV_NV21, BGR, MNNNV21ToBGR);
    CHECKFORMAT_CORE(YUV_NV21, RGBA, MNNNV21ToRGBA);
    CHECKFORMAT_CORE(YUV_NV21, BGRA, MNNNV21ToBGRA);
    return nullptr;
}

BLITTER ImageProcessUtils::choose(int channelByteSize) {
    switch (channelByteSize) {
        case 4:
            return MNNC4blitH;
        case 3:
            return MNNC3blitH;
        case 1:
            return MNNC1blitH;
        default:
            return nullptr;
    }
}

SAMPLER ImageProcessUtils::choose(ImageFormat format, Filter type, bool identity) {
    ImageFormatType formatType = (ImageFormatType)format;
    FilterType filterType = (FilterType)type;
    if (identity) {
        switch (formatType) {
            case ImageFormatType_RGBA:
            case ImageFormatType_BGRA:
                return MNNSamplerC4Copy;
            case ImageFormatType_GRAY:
                return MNNSamplerC1Copy;

            case ImageFormatType_RGB:
            case ImageFormatType_BGR:
                return MNNSamplerC3Copy;
            case ImageFormatType_YUV_NV21:
                return MNNSamplerNV21Copy;
            case ImageFormatType_YUV_NV12:
                return MNNSamplerNV12Copy;
            case ImageFormatType_YUV_I420:
                return MNNSamplerI420Copy;
            default:
                break;
        }
    }
    if (FilterType_BILINEAR == filterType) {
        switch (formatType) {
            case ImageFormatType_RGBA:
            case ImageFormatType_BGRA:
                return coreFunctions->MNNSamplerC4Bilinear;
            case ImageFormatType_GRAY:
                return MNNSamplerC1Bilinear;

            case ImageFormatType_RGB:
            case ImageFormatType_BGR:
                return MNNSamplerC3Bilinear;
            default:
                break;
        }
    }

    // Nearest
    switch (formatType) {
        case ImageFormatType_RGBA:
        case ImageFormatType_BGRA:
            return coreFunctions->MNNSamplerC4Nearest;
        case ImageFormatType_GRAY:
            return MNNSamplerC1Nearest;

        case ImageFormatType_RGB:
        case ImageFormatType_BGR:
            return MNNSamplerC3Nearest;
        case ImageFormatType_YUV_NV12:
            return MNNSamplerNV12Nearest;
        case ImageFormatType_YUV_NV21:
            return MNNSamplerNV21Nearest;
        case ImageFormatType_YUV_I420:
            return MNNSamplerI420Nearest;
        default:
            break;
    }
    MNN_PRINT("Don't support sampler for format:%d, type:%d", format, type);
    return nullptr;
}

BLIT_FLOAT ImageProcessUtils::choose(ImageFormat format, int dstBpp) {
    ImageFormatType formatType = (ImageFormatType)format;
    if (4 == dstBpp) {
        switch (formatType) {
            case ImageFormatType_GRAY:
                return MNNC1ToFloatRGBA;
            case ImageFormatType_RGBA:
            case ImageFormatType_BGRA:
                return MNNC4ToFloatC4;
            case ImageFormatType_RGB:
            case ImageFormatType_BGR:
                return MNNC3ToFloatRGBA;
            default:
                break;
        }
    }
    switch (formatType) {
        case ImageFormatType_GRAY:
            return MNNC1ToFloatC1;
        case ImageFormatType_RGBA:
        case ImageFormatType_BGRA:
            return MNNC4ToFloatC4;
        case ImageFormatType_RGB:
        case ImageFormatType_BGR:
            return MNNC3ToFloatC3;
        default:
            break;
    }
    return nullptr;
}

ErrorCode ImageProcessUtils::selectImageProcer(bool identity, bool hasBackend, bool isdraw) {
    if (isdraw) {
        mInside->mBlitter = choose(mInside->ic * mInside->mDtype.bytes());
        return NO_ERROR;
    }
    // Choose sampler.
    mInside->mSampler = choose(mInside->config.sourceFormat, mInside->config.filterType, identity);
    if (nullptr == mInside->mSampler) {
        return INPUT_DATA_ERROR;
    }
    // Choose blitter.
    if ((ImageFormatType)mInside->config.sourceFormat != (ImageFormatType)mInside->config.destFormat) {
        mInside->mBlitter = choose(mInside->config.sourceFormat, mInside->config.destFormat);
        if (nullptr == mInside->mBlitter) {
            return INPUT_DATA_ERROR;
        }
    }
    // Choose float blitter.
    if (mInside->mDtype.code == halide_type_float) {
        mInside->mBlitFloat = ImageProcessUtils::choose(mInside->config.destFormat, mInside->oc);
        if (nullptr == mInside->mBlitFloat) {
            return INPUT_DATA_ERROR;
        }
    }
    return NO_ERROR;
}

ErrorCode ImageProcessUtils::resizeFunc(int ic, int iw, int ih, int oc, int ow, int oh, halide_type_t type, int stride) {
    bool identity = mTransform.isIdentity() && iw >= ow && ih >= oh;
    bool hasBackend = false;
    mInside->mDtype = type;
    mInside->ow = ow;
    mInside->oh = oh;
    mInside->oc = oc;
    mInside->iw = iw;
    mInside->ih = ih;
    mInside->ic = ic;
    mInside->mStride = stride;
    return selectImageProcer(identity, hasBackend, mInside->mDraw);
}

static int LEFT   = 1 << 0;
static int RIGHT  = 1 << 1;
static int TOP    = 1 << 2;
static int BOTTOM = 1 << 3;
inline static uint8_t _encode(const CV::Point& p, int iw, int ih) {
    uint8_t mask = 0;
    if (p.fX < 0) {
        mask |= LEFT;
    }
    if (p.fX > iw - 1) {
        mask |= RIGHT;
    }
    if (p.fY < 0) {
        mask |= TOP;
    }
    if (p.fY > ih - 1) {
        mask |= BOTTOM;
    }
    return mask;
}
static std::pair<int, int> _computeClip(CV::Point* points, int iw, int ih, const CV::Matrix& invert, int xStart, int count) {
    auto code1 = _encode(points[0], iw, ih);
    auto code2 = _encode(points[1], iw, ih);
    int sta    = 0;
    int end    = count;

    float x1     = points[0].fX;
    float x2     = points[1].fX;
    float y1     = points[0].fY;
    float y2     = points[1].fY;
    int code     = 0;
    int pIndex   = 0;
    float deltaY = y2 - y1;
    float deltaX = x2 - x1;
    if (deltaX > 0.01f || deltaX < -0.01f) {
        deltaY = (y2 - y1) / (x2 - x1);
    } else {
        deltaY = 0;
    }
    if (deltaY > 0.01f || deltaY < -0.01f) {
        deltaX = (x2 - x1) / (y2 - y1);
    } else {
        deltaX = 0;
    }
    while (code1 != 0 || code2 != 0) {
        if ((code1 & code2) != 0) {
            sta = end;
            break;
        }
        if (code1 != 0) {
            code   = code1;
            pIndex = 0;
        } else if (code2 != 0) {
            code   = code2;
            pIndex = 1;
        }
        if ((LEFT & code) != 0) {
            points[pIndex].fY = points[pIndex].fY + deltaY * (0 - points[pIndex].fX);
            points[pIndex].fX = 0;
        } else if ((RIGHT & code) != 0) {
            points[pIndex].fY = points[pIndex].fY + deltaY * (iw - 1 - points[pIndex].fX);
            points[pIndex].fX = iw - 1;
        } else if ((BOTTOM & code) != 0) {
            points[pIndex].fX = points[pIndex].fX + deltaX * (ih - 1 - points[pIndex].fY);
            points[pIndex].fY = ih - 1;
        } else if ((TOP & code) != 0) {
            points[pIndex].fX = points[pIndex].fX + deltaX * (0 - points[pIndex].fY);
            points[pIndex].fY = 0;
        }
        auto tmp = invert.mapXY(points[pIndex].fX, points[pIndex].fY);
        if (0 == pIndex) {
            code1 = _encode(points[pIndex], iw, ih);
            // FUNC_PRINT_ALL(tmp.fX, f);
            // sta = (int)::ceilf(tmp.fX) - xStart;
            sta = (int)::round(tmp.fX) - xStart;
        } else {
            code2 = _encode(points[pIndex], iw, ih);
            // FUNC_PRINT_ALL(tmp.fX, f);
            // end = (int)::ceilf(tmp.fX) - xStart + 1;
            end = (int)::floor(tmp.fX) - xStart + 1;
        }
    }
    if (end > count) {
        end = count;
    }
    if (sta > end) {
        sta = end;
    }
    return std::make_pair(sta, end);
}

ErrorCode ImageProcessUtils::transformImage(const uint8_t* source, uint8_t* dst, uint8_t* samplerDest, uint8_t* blitDest, int tileCount, int destBytes, const int32_t* regions) {
    CV::Point points[2];
    if (mInside->mStride == 0) {
        mInside->mStride = mInside->iw * mInside->ic;
    }
    for (int i = 0; i < mInside->oh; ++i) {
        int dy = mInside->mDraw ? regions[3 * i] : i;
        auto dstY = (uint8_t*)dst + dy * destBytes * mInside->ow * mInside->oc;
        for (int tIndex = 0; tIndex < tileCount; ++tIndex) {
            int xStart    = tIndex * CACHE_SIZE;
            int count     = std::min(CACHE_SIZE, mInside->ow - xStart);
            if (mInside->mDraw) {
                xStart = regions[3 * i + 1];
                count = regions[3 * i + 2] - xStart + 1;
            }
            auto dstStart = dstY + destBytes * mInside->oc * xStart;
          
            if (!mInside->mBlitFloat) {
                blitDest = dstStart;
            }
            if (!mInside->mBlitter) {
                samplerDest = blitDest;
            }

            // Sample
            if (!mInside->mDraw) {
                // Compute position
                points[0].fX = xStart;
                points[0].fY = dy;

                points[1].fX = xStart + count;
                points[1].fY = dy;
                mTransform.mapPoints(points, 2);
                float deltaY = points[1].fY - points[0].fY;
                float deltaX = points[1].fX - points[0].fX;

                int sta = 0;
                int end = count;

                // FUNC_PRINT(sta);
                if ((WrapType)mInside->config.wrap == WrapType_ZERO) {
                    // Clip: Cohen-Sutherland
                    auto clip    = _computeClip(points, mInside->iw, mInside->ih, mTransformInvert, xStart, count);
                    sta          = clip.first;
                    end          = clip.second;
                    points[0].fX = sta + xStart;
                    points[0].fY = dy;

                    mTransform.mapPoints(points, 1);
                    if (sta != 0 || end < count) {
                        if (mInside->ic > 0) {
                            if (sta > 0) {
                                ::memset(samplerDest, mPaddingValue, mInside->ic * sta);
                            }
                            if (end < count) {
                                ::memset(samplerDest + end * mInside->ic, mPaddingValue, (count - end) * mInside->ic);
                            }
                        } else {
                            // TODO, Only support NV12 / NV21
                            ::memset(samplerDest, mPaddingValue, count);
                            ::memset(samplerDest + count, 128, UP_DIV(count, 2) * 2);
                        }
                    }
                }
                points[1].fX = (deltaX) / (float)(count);
                points[1].fY = (deltaY) / (float)(count);

                mInside->mSampler(source, samplerDest, points, sta, end - sta, count, mInside->iw, mInside->ih, mInside->mStride);
            }
            // Convert format
            if (mInside->mBlitter) {
                mInside->mBlitter(samplerDest, blitDest, count);
            }
            // Turn float
            if (mInside->mBlitFloat) {
                mInside->mBlitFloat(blitDest, (float*)dstStart, mInside->config.mean, mInside->config.normal, count);
            }
        }
    }
    return NO_ERROR;
}

void ImageProcessUtils::setMatrix(const CV::Matrix& matrix) {
   mTransform = matrix;
   mTransform.invert(&mTransformInvert);
}

static int _getBpp(CV::ImageFormat format) {
    switch (format) {
        case CV::RGB:
        case CV::BGR:
        case CV::YCrCb:
        case CV::YUV:
        case CV::HSV:
        case CV::XYZ:
            return 3;
        case CV::RGBA:
        case CV::BGRA:
            return 4;
        case CV::GRAY:
            return 1;
        case CV::BGR555:
        case CV::BGR565:
            return 2;
        default:
            break;
    }
    return 0;
}

static CV::ImageFormat _correctImageFormat(int outputBpp, halide_type_t type, CV::ImageFormat format) {
    if (outputBpp != 4) {
        return format;
    }
    // TODO, use same judge for uint8 -> float
    if (type.code == halide_type_float) {
        return format;
    }

    static std::map<CV::ImageFormat, CV::ImageFormat> imageFormatTable = {{CV::RGB, CV::RGBA}, {CV::BGR, CV::BGRA}, {CV::GRAY, CV::RGBA}};
    if (imageFormatTable.find(format) != imageFormatTable.end()) {
        return imageFormatTable.find(format)->second;
    }
    return format;
}

ErrorCode ImageProcessUtils::execFunc(const uint8_t *source, int stride, void *dest) {
    uint8_t sampleDest[4 * 256];
    uint8_t blitDest[4 * 256];
    int destBytes = mInside->mDtype.bytes();
    int tileCount = UP_DIV(mInside->ow, 256);
    if (mInside->mDraw) {
        tileCount = 1;
    }

    return transformImage(source, (uint8_t*)dest, sampleDest, blitDest, tileCount, destBytes, nullptr);
}

void ImageProcessUtils::setDraw() {
    if (mInside) {
//        mInside->execution->setDraw();
        mInside->mDraw = true;
    }
}

void ImageProcessUtils::draw(uint8_t* img, int w, int h, int c, const int* regions, int num, uint8_t* color) {
    uint8_t blitDest[4 * 256];
    int destBytes = mInside->mDtype.bytes();
    mInside->oh = num;
    transformImage(img, img, color, blitDest, 1, destBytes, regions);
}
} // namespace MNN
