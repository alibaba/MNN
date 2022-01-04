//
//  CPUImageProcess.cpp
//  MNN
//
//  Created by MNN on 2021/10/27.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUImageProcess.hpp"
#include "compute/ImageProcessFunction.hpp"
#include <string.h>
#include <mutex>
#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#ifdef MNN_USE_SSE
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#include <map>
#include <utility>

namespace MNN {
#define CACHE_SIZE 256
#define CHECKFORMAT(src, dst, func) if (source == ImageFormatType_##src && dest == ImageFormatType_##dst) return func
#define CHECKFORMAT_CORE(src, dst, func) if (source == ImageFormatType_##src && dest == ImageFormatType_##dst) return coreFunctions ? coreFunctions->func : func;

BLITTER CPUImageProcess::choose(ImageFormatType source, ImageFormatType dest) {
    // YUV only different in sampler
    if (source == ImageFormatType_YUV_NV12) {
        source = ImageFormatType_YUV_NV21;
    }
    if (source == ImageFormatType_YUV_I420) {
        source = ImageFormatType_YUV_NV21;
    }
    CHECKFORMAT(RGBA, RGBA, MNNCopyC4);
    CHECKFORMAT_CORE(RGBA, BGRA, MNNRGBAToBGRA);
    CHECKFORMAT(RGBA, BGR, MNNRGBAToBGR);
    CHECKFORMAT(RGBA, RGB, MNNBGRAToBGR);
    CHECKFORMAT(RGBA, GRAY, MNNRGBAToGRAY);

    CHECKFORMAT_CORE(BGRA, RGBA, MNNRGBAToBGRA);
    CHECKFORMAT(BGRA, BGRA, MNNCopyC4);
    CHECKFORMAT(BGRA, BGR, MNNRGBAToBGR);
    CHECKFORMAT(BGRA, RGB, MNNBGRAToBGR);
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

SAMPLER CPUImageProcess::choose(ImageFormatType format, FilterType type, bool identity) {
    if (identity) {
        switch (format) {
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
    if (FilterType_BILINEAR == type) {
        switch (format) {
            case ImageFormatType_RGBA:
            case ImageFormatType_BGRA:
                return MNNSamplerC4Bilinear;
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
    switch (format) {
        case ImageFormatType_RGBA:
        case ImageFormatType_BGRA:
            return MNNSamplerC4Nearest;
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

BLIT_FLOAT CPUImageProcess::choose(ImageFormatType format, int dstBpp) {
    if (4 == dstBpp) {
        switch (format) {
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
    switch (format) {
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

ErrorCode CPUImageProcess::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0], output = outputs[0];
    ih = input->height();
    iw = input->width();
    ic = input->channel();
    oh = output->height();
    ow = output->width();
    oc = output->channel();
    dtype = output->getType();

    bool identity = transform.isIdentity() && iw >= ow && ih >= oh;
    // choose sampler
    sampler  = choose(sourceFormat, filterType, identity);
    if (nullptr == sampler) {
        return INPUT_DATA_ERROR;
    }
    // choose blitter
    if (sourceFormat != destFormat) {
        blitter = choose(sourceFormat, destFormat);
        if (nullptr == blitter) {
            return INPUT_DATA_ERROR;
        }
        if (backend()) {
            cacheBuffer.reset(Tensor::createDevice<uint8_t>(std::vector<int>{4 * CACHE_SIZE}));
            backend()->onAcquireBuffer(cacheBuffer.get(), Backend::DYNAMIC);
            samplerDest = cacheBuffer->host<uint8_t>();
        } else {
            samplerBuffer.reset(new uint8_t[4 * CACHE_SIZE]);
            samplerDest = samplerBuffer.get();
        }
    }
    // choose float blitter
    if (dtype.code == halide_type_float) {
        blitFloat = choose(destFormat, oc);
        if (nullptr == blitFloat) {
            return INPUT_DATA_ERROR;
        }
        if (backend()) {
            cacheBufferRGBA.reset(Tensor::createDevice<uint8_t>(std::vector<int>{4 * CACHE_SIZE}));
            backend()->onAcquireBuffer(cacheBufferRGBA.get(), Backend::DYNAMIC);
            blitDest = cacheBufferRGBA->host<uint8_t>();
        } else {
            blitBuffer.reset(new uint8_t[4 * CACHE_SIZE]);
            blitDest = blitBuffer.get();
        }
    }
    return NO_ERROR;
}

ErrorCode CPUImageProcess::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto source = inputs[0]->host<uint8_t>();
    auto dest = outputs[0]->host<void>();
    CV::Point points[2];
    int tileCount = UP_DIV(ow, CACHE_SIZE);
    auto destBytes = dtype.bytes();
    for (int dy = 0; dy < oh; ++dy) {
        auto dstY = (uint8_t*)dest + dy * destBytes * ow * oc;
        for (int tIndex = 0; tIndex < tileCount; ++tIndex) {
            int xStart    = tIndex * CACHE_SIZE;
            int count     = std::min(CACHE_SIZE, ow - xStart);
            auto dstStart = dstY + destBytes * oc * xStart;
          
            if (!blitFloat) {
                blitDest = dstStart;
            }
            if (!blitter) {
                samplerDest = blitDest;
            }

            // Sample
            {
                // Compute position
                points[0].fX = xStart;
                points[0].fY = dy;

                points[1].fX = xStart + count;
                points[1].fY = dy;
                transform.mapPoints(points, 2);
                float deltaY = points[1].fY - points[0].fY;
                float deltaX = points[1].fX - points[0].fX;

                int sta = 0;
                int end = count;

                // FUNC_PRINT(sta);
                if (wrap == WrapType_ZERO) {
                    // Clip: Cohen-Sutherland
                    auto clip    = _computeClip(points, iw, ih, transformInvert, xStart, count);
                    sta          = clip.first;
                    end          = clip.second;
                    points[0].fX = sta + xStart;
                    points[0].fY = dy;

                    transform.mapPoints(points, 1);
                    if (sta != 0 || end < count) {
                        if (ic > 0) {
                            if (sta > 0) {
                                ::memset(samplerDest, paddingValue, ic * sta);
                            }
                            if (end < count) {
                                ::memset(samplerDest + end * ic, paddingValue, (count - end) * ic);
                            }
                        } else {
                            // TODO, Only support NV12 / NV21
                            ::memset(samplerDest, paddingValue, count);
                            ::memset(samplerDest + count, 128, UP_DIV(count, 2) * 2);
                        }
                    }
                }
                points[1].fX = (deltaX) / (float)(count);
                points[1].fY = (deltaY) / (float)(count);

                sampler(source, samplerDest, points, sta, end - sta, count, iw, ih, iw * ic);
            }
            // Convert format
            if (blitter) {
                blitter(samplerDest, blitDest, count);
            }
            // Turn float
            if (blitFloat) {
                blitFloat(blitDest, (float*)dstStart, mean, normal, count);
            }
        }
    }

    return NO_ERROR;
}

class CPUImageProcessCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto process = op->main_as_ImageProcessParam();
        return new CPUImageProcess(backend, process);
    }
};

REGISTER_CPU_OP_CREATOR(CPUImageProcessCreator, OpType_ImageProcess);
} // namespace MNN
