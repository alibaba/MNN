//
//  ConvolutionInt8Executor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionInt8Executor.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "ConvolutionIntFactory.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "Int8FunctionsOpt.h"
#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
#define SRC_UNIT 8

// One Tile Compute DST_XUNIT * outputCount 's number
#ifdef __aarch64__
#define DST_XUNIT 6
#else
#define DST_XUNIT 2
#endif

extern "C" {
void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                   size_t dst_step, size_t dst_depth_quad);
}
#ifndef MNN_USE_NEON
void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                   size_t dst_step, size_t dst_depth_quad) {
    dst_step /= sizeof(float);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto weight_dz = weight + src_depth_quad * dz * 32;
        auto dst_z     = dst + dz * dst_step;
        for (int w = 0; w < DST_XUNIT; ++w) {
            auto dst_x     = dst_z + 4 * w;
            int32_t dst[4] = {0, 0, 0, 0};
            auto src_x     = src + 8 * w;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                auto weight_sz = weight_dz + 32 * sz;
                auto src_z     = src_x + sz * DST_XUNIT * 8;
                for (int j = 0; j < 4; ++j) {
                    auto weight_j = weight_sz + j * 8;
                    for (int i = 0; i < 8; ++i) {
                        dst[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }
            for (int j = 0; j < 4; ++j) {
                dst_x[j] = dst[j];
            }
        }
    }
}
#endif

namespace MNN {
ConvolutionInt8Executor::ConvolutionInt8Executor(const Convolution2DCommon* convOp, Backend* b,
                                                 const ConvolutionIntFactory::Int8Common* common, const float* bias,
                                                 size_t biasSize)
    : MNN::CPUConvolution(convOp, b) {
    mBias.reset(ALIGN_UP4((int)biasSize));
    mBias.clear();
    auto biasDest = mBias.get();
    mAMin         = common->quan->aMin();
    mAMax         = common->quan->aMax();
    mQuanScale    = common->quan->quantScale();

    // The postTreat will contain scale_bias and biasRelu, so the bias will be add twice
    for (int i = 0; i < biasSize; ++i) {
        biasDest[i] = bias[i] * 0.5f;
    }
    int outputCount = (int)biasSize;
    mQuan           = common->quan;
    MNN_ASSERT(nullptr != mQuan);
    mAlpha.reset(ALIGN_UP4(common->alpha.size()));
    mAlpha.clear();
    ::memcpy(mAlpha.get(), common->alpha.get(), common->alpha.size() * sizeof(float));

    auto weightLength       = common->weight.size();
    mSrcCount               = (int)weightLength / mCommon->kernelX() / mCommon->kernelY() / outputCount;
    auto kx                 = mCommon->kernelX();
    auto ky                 = mCommon->kernelY();
    auto kernelCount        = kx * ky;
    auto srcCount           = mSrcCount;
    auto outputCountUnit    = UP_DIV(outputCount, UNIT);
    auto srcCountUnit       = UP_DIV(srcCount, UNIT);
    auto totalKernelCountD8 = UP_DIV(srcCountUnit * kx * ky, 2);
    mWeight.reset(Tensor::create<int8_t>(std::vector<int>{outputCountUnit, totalKernelCountD8, UNIT, SRC_UNIT}));
    auto dst = mWeight->host<int8_t>();
    for (int k = 0; k < kernelCount; ++k) {
        auto srcK = common->weight.get() + k;
        for (int y = 0; y < srcCount; ++y) {
            int yOutSide    = y / UNIT;
            int yInside     = y % UNIT;
            int yIndex      = yOutSide + k * srcCountUnit;
            int ySubOutside = yIndex / 2;
            int ySubInside  = yIndex % 2;

            auto dstY = dst + ySubOutside * mWeight->stride(1) + ySubInside * UNIT + yInside;
            auto srcY = srcK + y * kernelCount;
            for (int x = 0; x < outputCount; ++x) {
                int xOutSide = x / UNIT;
                int xInside  = x % UNIT;

                auto dstX = dstY + xOutSide * mWeight->stride(0) + xInside * SRC_UNIT;
                auto srcX = srcY + x * kernelCount * srcCount;

                dstX[0] = srcX[0];
            }
        }
    }
}

ErrorCode ConvolutionInt8Executor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    int tileCount           = UP_DIV(outputs[0]->width() * outputs[0]->height(), DST_XUNIT);
    auto outputCountUnit    = UP_DIV(outputs[0]->channel(), 4);
    int number              = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    number                  = std::min(number, tileCount);
    mIm2ColParamter.dilateX = mCommon->dilateX();
    mIm2ColParamter.dilateY = mCommon->dilateY();
    mIm2ColParamter.strideX = mCommon->strideX();
    mIm2ColParamter.strideY = mCommon->strideY();
    mIm2ColParamter.padX    = mPadX;
    mIm2ColParamter.padY    = mPadY;
    mIm2ColParamter.ih      = inputs[0]->height();
    mIm2ColParamter.iw      = inputs[0]->width();
    mIm2ColParamter.icDiv4  = UP_DIV(inputs[0]->channel(), 4);
    mIm2ColParamter.ow      = outputs[0]->width();
    mIm2ColParamter.oh      = outputs[0]->height();
    mIm2ColParamter.kernelX = mCommon->kernelX();
    mIm2ColParamter.kernelY = mCommon->kernelY();
    mIm2ColParamter.kernelCountUnit =
        UP_DIV(mIm2ColParamter.icDiv4 * mIm2ColParamter.kernelY * mIm2ColParamter.kernelX, 2);

    TensorUtils::copyShape(inputs[0], &mSrcCopyBuffer);
    mSrcCopyBuffer.buffer().dim[0].extent = 1;
    mSrcCopyBuffer.buffer().type          = halide_type_of<int8_t>();
    TensorUtils::setLinearLayout(&mTempBuffer);
    mTempBuffer.buffer().type          = halide_type_of<int8_t>();
    mTempBuffer.buffer().dimensions    = 3;
    mTempBuffer.buffer().dim[0].extent = number;
    mTempBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempBuffer.buffer().dim[2].extent = mWeight->length(1) * SRC_UNIT;
    TensorUtils::setLinearLayout(&mTempBuffer);

    mTempDstBuffer.buffer().type          = halide_type_of<float>();
    mTempDstBuffer.buffer().dimensions    = 3;
    mTempDstBuffer.buffer().dim[0].extent = number;
    mTempDstBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempDstBuffer.buffer().dim[2].extent = outputCountUnit * UNIT;
    TensorUtils::setLinearLayout(&mTempDstBuffer);

    bool success = backend()->onAcquireBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

typedef void (*im2ColFunction)(int8_t* colAddr, const int8_t* inputOrigin,
                               const CPUConvolution::Im2ColParameter* im2ColParameter, size_t xIndexStart,
                               size_t realDstCount);

static void _fastIm2Col(int8_t* colAddr, const int8_t* inputOrigin,
                        const CPUConvolution::Im2ColParameter* im2ColParameter, size_t xIndexStart,
                        size_t realDstCount) {
    int icDiv8   = im2ColParameter->icDiv4 / 2;
    int srcZStep = im2ColParameter->iw * im2ColParameter->ih * 4;
    inputOrigin += xIndexStart * UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + SRC_UNIT * i;
        auto inputK   = inputOrigin + UNIT * i;
        for (int sz = 0; sz < icDiv8; ++sz) {
            auto inputZ0      = inputK + srcZStep * (2 * sz + 0);
            auto inputZ1      = inputK + srcZStep * (2 * sz + 1);
            auto indexOutside = sz;

            auto dstK0         = colAddrI + indexOutside * SRC_UNIT * DST_XUNIT;
            auto dstK1         = dstK0 + UNIT;
            *((int32_t*)dstK0) = *((int32_t*)inputZ0);
            *((int32_t*)dstK1) = *((int32_t*)inputZ1);
        }
    }
}

static void _im2ColCommonZ1(int8_t* colAddr, const int8_t* inputOrigin,
                            const CPUConvolution::Im2ColParameter* im2ColParameter, size_t xIndexStart,
                            size_t realDstCount) {
    int col_buffer_size = im2ColParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, 0, col_buffer_size);
    auto ih            = im2ColParameter->ih;
    auto iw            = im2ColParameter->iw;
    auto kh            = im2ColParameter->kernelY;
    auto kw            = im2ColParameter->kernelX;
    auto dilateX       = im2ColParameter->dilateX;
    auto dilateY       = im2ColParameter->dilateY;
    auto icDiv4        = im2ColParameter->icDiv4;
    auto dstXStepInt32 = SRC_UNIT * DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2ColParameter->ow;
        int oy     = xIndex / im2ColParameter->ow;

        int sx = ox * im2ColParameter->strideX - im2ColParameter->padX;
        int sy = oy * im2ColParameter->strideY - im2ColParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2ColParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2ColParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2ColParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2ColParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + SRC_UNIT * i;
        auto inputOffset = inputOrigin + (sx + sy * iw) * UNIT + (sfx * dilateX) * UNIT + (sfy * dilateY) * iw * UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK       = inputOffset + (fx * dilateX) * UNIT + (fy * dilateY) * iw * UNIT;
                auto indexStart   = indexOffset + (fy * kw + fx) * icDiv4;
                auto indexInside  = indexStart % 2;
                auto indexOutside = indexStart / 2;
                auto dstK0        = (int32_t*)colAddrI + indexOutside * dstXStepInt32 + indexInside;
                dstK0[0]          = *((int32_t*)inputK);
            }
        }
    }
}

static void _im2ColCommon(int8_t* colAddr, const int8_t* inputOrigin,
                          const CPUConvolution::Im2ColParameter* im2ColParameter, size_t xIndexStart,
                          size_t realDstCount) {
    int col_buffer_size = im2ColParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, 0, col_buffer_size);
    auto ih            = im2ColParameter->ih;
    auto iw            = im2ColParameter->iw;
    auto kh            = im2ColParameter->kernelY;
    auto kw            = im2ColParameter->kernelX;
    auto dilateX       = im2ColParameter->dilateX;
    auto dilateY       = im2ColParameter->dilateY;
    auto icDiv4        = im2ColParameter->icDiv4;
    auto srcZStep      = iw * ih * UNIT;
    int icD4D2         = icDiv4 / 2;
    int remain         = icDiv4 - icD4D2 * 2;
    auto dstXStepInt32 = SRC_UNIT * DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2ColParameter->ow;
        int oy     = xIndex / im2ColParameter->ow;

        int sx = ox * im2ColParameter->strideX - im2ColParameter->padX;
        int sy = oy * im2ColParameter->strideY - im2ColParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2ColParameter->dilateX)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2ColParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2ColParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2ColParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + SRC_UNIT * i;
        auto inputOffset = inputOrigin + (sx + sy * iw) * UNIT + (sfx * dilateX) * UNIT + (sfy * dilateY) * iw * UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK             = inputOffset + (fx * dilateX) * UNIT + (fy * dilateY) * iw * UNIT;
                auto indexStart         = indexOffset + (fy * kw + fx) * icDiv4;
                auto indexInside        = indexStart % 2;
                auto indexInsideSecond  = (indexStart + 1) % 2;
                auto indexOutside       = indexStart / 2;
                auto indexOutsideSecond = (indexStart + 1) / 2;
                auto dstK0              = (int32_t*)colAddrI + indexOutside * dstXStepInt32 + indexInside;
                auto dstK1              = (int32_t*)colAddrI + indexOutsideSecond * dstXStepInt32 + indexInsideSecond;
                for (int sz = 0; sz < icD4D2; ++sz) {
                    dstK0[0] = *((int32_t*)inputK);
                    inputK += srcZStep;

                    dstK1[0] = *((int32_t*)inputK);
                    inputK += srcZStep;

                    dstK0 += dstXStepInt32;
                    dstK1 += dstXStepInt32;
                }
                if (remain) {
                    dstK0[0] = *((int32_t*)inputK);
                }
            }
        }
    }
}
ErrorCode ConvolutionInt8Executor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    //        AUTOTIME;
    auto input        = inputs[0];
    auto output       = outputs[0];
    auto weightOrigin = mWeight->host<int8_t>();
    auto dstZStep     = output->width() * output->height() * 4;
    int threadNumber  = 1;
    bool fastIm2Col = mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && mIm2ColParamter.icDiv4 % 2 == 0 &&
                      mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && mIm2ColParamter.padX == 0 &&
                      mIm2ColParamter.padY == 0;

    auto im2ColProc = _im2ColCommon;
    if (fastIm2Col) {
        im2ColProc = _fastIm2Col;
    } else if (input->channel() <= 4) {
        im2ColProc = _im2ColCommonZ1;
    }
    int batch            = input->batch();
    int width            = output->width();
    int height           = output->height();
    auto ocC4            = UP_DIV(output->channel(), 4);
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    int count            = width * height;
    float quantScale[] = {
        mQuanScale,
        mQuanScale,
        mQuanScale,
        mQuanScale
    };

    // MNN_PRINT("%s, %d, %d, %d,%d->%d,%d\n", layer->layer.layerId, layer->kernelSize[0], layer->kernelSize[1],
    // input->d1, input->d2, output->d1, output->d2);

    int inputTotalSize = mSrcCopyBuffer.elementSize();
    int8_t* srcCopy    = mSrcCopyBuffer.host<int8_t>();
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto srcOrigin = input->host<float>() + input->stride(0) * batchIndex;
        auto dstOrigin = output->host<float>() + output->stride(0) * batchIndex;

        MNNFloat2Int8(srcOrigin, srcCopy, inputTotalSize / 4, quantScale, mAMin, mAMax);
        int tileCount = UP_DIV(count, DST_XUNIT);

        threadNumber        = std::max(((CPUBackend*)backend())->threadNumber(), 1);
        threadNumber        = std::min(threadNumber, tileCount);
        auto outputOrigin   = output->host<float>() + batchIndex * output->stride(0);
        auto threadFunction = [&](int tId) {
            auto colAddr        = mTempBuffer.host<int8_t>() + tId * mTempBuffer.buffer().dim[0].stride;
            auto gemmOutputAddr = mTempDstBuffer.host<float>() + tId * mTempDstBuffer.buffer().dim[0].stride;

            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndexStart  = tIndex * DST_XUNIT;
                int realDstCount = ALIMIN(count - xIndexStart, DST_XUNIT);
                /*Im2Col Begin*/
                im2ColProc(colAddr, srcCopy, &mIm2ColParamter, xIndexStart, realDstCount);
                /*Im2Col End*/

                auto outputInTile = outputOrigin + xIndexStart * UNIT;
                // GEMM
                if (realDstCount == DST_XUNIT) {
                    MNNGemmInt8toFloat32_8x4_Unit(outputInTile, colAddr, weightOrigin, kernelCountUnit,
                                                  dstZStep * sizeof(float), ocC4);
                } else {
                    MNNGemmInt8toFloat32_8x4_Unit(gemmOutputAddr, colAddr, weightOrigin, kernelCountUnit,
                                                  UNIT * DST_XUNIT * sizeof(float), ocC4);
                    /*Copy Data to Real Output*/
                    for (int z = 0; z < ocC4; ++z) {
                        auto outputZ = outputInTile + z * dstZStep;
                        auto srcZ    = gemmOutputAddr + z * UNIT * DST_XUNIT;
                        ::memcpy(outputZ, srcZ, realDstCount * UNIT * sizeof(float));
                    }
                }
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            threadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        threadNumber = std::max(((CPUBackend*)backend())->threadNumber(), 1);
        threadNumber = std::min(threadNumber, ocC4);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int z = (int)tId; z < ocC4; z += threadNumber) {
                MNNScaleAndAddBias(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, mBias.get() + 4 * z,
                                   mAlpha.get() + 4 * z, width * height, 1);
                mPostFunction(dstOrigin + z * dstZStep, mBias.get() + 4 * z, width * height, 1);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
