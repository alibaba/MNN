//
//  ConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionTiledExecutor.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "core/BufferAllocator.hpp"
#include "core/MemoryFormater.h"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void ConvolutionTiledExecutor::initWeight(const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    // Swap k, ic
    int dims[4] = {
        depth,
        kernelSize,
        kernelSize,
        depth
    };
    for (int o=0; o<outputCount; ++o) {
        auto dO = cache + o * depth * kernelSize;
        auto sO = source + o * depth * kernelSize;
        MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
    }
    if (function->bytes < 4) {
        // Lowp
        function->MNNFp32ToLowp((float*)cache, (int16_t*)cache, outputCount * kernelSize * depth);
    }
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(Backend* b, const float* bias, size_t biasSize)
    : MNN::Execution(b) {

    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    mValid = mResource->copyBiasAlign(bias, biasSize);
    if (!mValid) {
        return;
    }
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, Backend* b) : mResource(res), Execution(b) {
}

ConvolutionTiledExecutor::~ConvolutionTiledExecutor() {
    // Do nothing
}
bool ConvolutionTiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvolutionTiledExecutor(mResource, bn);
    return true;
}

ErrorCode ConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode ConvolutionTiledImpl::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {

    MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
        mFunction.second((int)tId);
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

std::pair<size_t, std::pair<size_t, size_t>> ConvolutionTiledExecutor::computeBlitInfoSize(int eP, int ow, int kernelSize, int threadNumber) {
    auto maxLine       = UP_DIV(eP, ow) + 1;
    auto stride = kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *));
    auto total = threadNumber * stride;
    return std::make_pair(total, std::make_pair(stride, kernelSize * maxLine));
}

void ConvolutionTiledExecutor:: setIm2ColParameter(ConvolutionCommon::Im2ColParameter& dstIm2ColParamter, const Convolution2DCommon* convCommon, Tensor* input, Tensor* output, int padX, int padY, const CoreFunctions* floatCore, const CoreInt8Functions* int8Core, int pack) {
    // FIXME: Set int8 and float's pack as diff
    if (pack == 0) {
        pack = floatCore->pack;
    }
    
    const auto kernelCount = convCommon->kernelX() * convCommon->kernelY();

    dstIm2ColParamter.dilateX         = convCommon->dilateX();
    dstIm2ColParamter.dilateY         = convCommon->dilateY();
    dstIm2ColParamter.strideX         = convCommon->strideX();
    dstIm2ColParamter.strideY         = convCommon->strideY();
    dstIm2ColParamter.icDiv4          = UP_DIV(input->channel(), pack);;
    dstIm2ColParamter.kernelX         = convCommon->kernelX();
    dstIm2ColParamter.kernelY         = convCommon->kernelY();
    dstIm2ColParamter.padX = padX;
    dstIm2ColParamter.padY = padY;

    dstIm2ColParamter.ih = input->height();
    dstIm2ColParamter.iw = input->width();
    dstIm2ColParamter.oh = output->height();
    dstIm2ColParamter.ow = output->width();
    dstIm2ColParamter.srcZStep = input->stride(1) * pack * input->batch();
    dstIm2ColParamter.srcYStep = input->stride(2) * pack;
    dstIm2ColParamter.packCUnit = pack;
    dstIm2ColParamter.ic = input->channel();
    if (nullptr != int8Core) {
        // Compute Int8 Info and align ic
        int UNIT, SRC_UNIT, DynamicDestUnit;
        auto core = int8Core;
        core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DynamicDestUnit);
        if (SRC_UNIT > pack) {
            const auto srcCountUnit = UP_DIV(input->channel(), pack);
            dstIm2ColParamter.kernelCountUnit = UP_DIV(srcCountUnit * kernelCount, SRC_UNIT / pack);
            dstIm2ColParamter.ic = dstIm2ColParamter.icDiv4 * pack;
        } else {
            const auto srcCountUnit = UP_DIV(input->channel(), SRC_UNIT);
            dstIm2ColParamter.kernelCountUnit = srcCountUnit * kernelCount;
            dstIm2ColParamter.ic = srcCountUnit * SRC_UNIT;
        }
    }
    if (dstIm2ColParamter.iw == 1 && dstIm2ColParamter.ow == 1 && dstIm2ColParamter.oh > 1 && dstIm2ColParamter.kernelX == 1 && dstIm2ColParamter.padX == 0) {
        /* Convolution only work for Height. Swap x, y*/
        dstIm2ColParamter.ow = dstIm2ColParamter.oh;
        dstIm2ColParamter.oh = 1;
        dstIm2ColParamter.padX = dstIm2ColParamter.padY;
        dstIm2ColParamter.padY = 0;
        dstIm2ColParamter.strideX = dstIm2ColParamter.strideY;
        dstIm2ColParamter.strideY = 1; /* Don't need stride */
        dstIm2ColParamter.iw = dstIm2ColParamter.ih;
        dstIm2ColParamter.ih = 1;
        dstIm2ColParamter.dilateX = dstIm2ColParamter.dilateY;
        dstIm2ColParamter.dilateY = 1;
        dstIm2ColParamter.kernelX = dstIm2ColParamter.kernelY;
        dstIm2ColParamter.kernelY = 1;
    }
}
std::pair<int, bool> ConvolutionTiledExecutor::turnIm2ColToBlitInfo(float const ** srcPtr, int32_t* el, int start, int xC, const ConvolutionCommon::Im2ColParameter& p, const uint8_t* srcOrigin, int bytes) {
    /* Compute Pack position */
    int oyBegin   = start / p.ow;
    int oxBegin   = start % p.ow;
    int oyEnd     = (start + xC - 1) / p.ow;
    int remain    = xC;
    int number    = 0;
    bool needZero = false;
    int eStart    = 0;
    auto unit = p.packCUnit;

    for (int oyb = oyBegin; oyb <= oyEnd; ++oyb) {
        int step    = std::min(p.ow - oxBegin, remain);
        int oy      = oyb % p.oh;
        int ob      = oyb / p.oh;
        int sySta   = oy * p.strideY - p.padY;
        int kyStart = std::max(0, UP_DIV(-sySta, p.dilateY));
        int kyEnd   = std::min(p.kernelY, UP_DIV(p.ih - sySta, p.dilateY));
        if (kyEnd - kyStart < p.kernelY) {
            needZero = true;
        }
        auto srcStart = srcOrigin + ((ob * p.ih + sySta) * p.iw) * bytes * unit;
        for (int ky = kyStart; ky < kyEnd; ++ky) {
            auto lKYOffset = ky * p.kernelX * p.ic;
            auto srcKy     = srcStart + ky * p.dilateY * p.iw * bytes * unit;
            for (int kx = 0; kx < p.kernelX; ++kx) {
                /* Compute x range:*/
                /* 0 <= (oxBegin + x) * strideX - padX + dilateX * kx < src_width*/
                /* 0 <= x <= step*/
                int end = std::min(
                    step, (p.iw - oxBegin * p.strideX - p.dilateX * kx + p.padX + p.strideX - 1) / p.strideX);
                int sta = std::max(0, UP_DIV((p.padX - oxBegin * p.strideX - p.dilateX * kx), p.strideX));
                if (end - sta < step) {
                    needZero = true;
                }
                if (end > sta) {
                    auto lOffset = lKYOffset + (kx * p.ic);
                    auto srcKx   = srcKy + ((oxBegin + sta) * p.strideX + p.dilateX * kx - p.padX) * bytes * unit;
                    srcPtr[number]     = (const float*)srcKx;
                    el[4 * number + 0] = end - sta;
                    el[4 * number + 1] = p.ic;
                    el[4 * number + 2] = eStart + sta;
                    el[4 * number + 3] = lOffset;
                    number++;
                }
            }
        }
        oxBegin = 0;
        remain -= step;
        eStart += step;
    }
    return std::make_pair(number, needZero);
}

} // namespace MNN
