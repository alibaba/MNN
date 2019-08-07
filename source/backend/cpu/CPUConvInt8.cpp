//
//  CPUConvInt8.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConvInt8.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "Macro.h"
#include "TensorUtils.hpp"
#include <math.h>

#define UNIT 4
#define SRC_UNIT 16

#ifdef __aarch64__
#define DST_XUNIT 4
#else
#define DST_XUNIT 2
#endif

extern "C" {
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step,
                                       size_t dst_depth_quad);
}

namespace MNN {

#ifndef MNN_USE_NEON
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}

static void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                              const float* scale, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (UNIT * SRC_UNIT);
        const auto bias_dz   = bias + dz * UNIT;
        const auto scale_dz  = scale + dz * UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        for (int w = 0; w < DST_XUNIT; ++w) {
            const auto src_x   = src + w * SRC_UNIT;
            auto dst_x         = dst_z + w * UNIT;
            int32_t dstTemp[4] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (UNIT * SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * DST_XUNIT * SRC_UNIT;

                for (int j = 0; j < UNIT; ++j) {
                    const auto weight_j = weight_sz + j * SRC_UNIT;
                    for (int i = 0; i < SRC_UNIT; ++i) {
                        dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < 4; ++j) {
                dst_x[j] = int32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j]);
            }
        }
    }
}

#endif

static void _fastIm2Col(int8_t* colAddr, const int8_t* inputOrigin,
                        const CPUConvolution::Im2ColParameter* im2colParameter, size_t xIndexStart,
                        size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, 0, col_buffer_size);
    const int icDiv8   = im2colParameter->icDiv4 / 2;
    const int srcZStep = im2colParameter->iw * im2colParameter->ih * 4;
    inputOrigin += xIndexStart * UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + SRC_UNIT * i;
        auto inputK   = inputOrigin + UNIT * i;
        for (int sz = 0; sz < icDiv8; ++sz) {
            auto inputZ0           = inputK + srcZStep * (2 * sz + 0);
            auto inputZ1           = inputK + srcZStep * (2 * sz + 1);
            const int indexOutside = sz / 2;
            const int indexInsize  = sz % 2;

            auto dstK0         = colAddrI + (indexOutside * DST_XUNIT * 2 + indexInsize) * (2 * UNIT);
            auto dstK1         = dstK0 + UNIT;
            *((int32_t*)dstK0) = *((int32_t*)inputZ0);
            *((int32_t*)dstK1) = *((int32_t*)inputZ1);
        }
    }
}

static void _im2colCommonZ1(int8_t* colAddr, const int8_t* inputOrigin,
                            const CPUConvolution::Im2ColParameter* im2colParameter, size_t xIndexStart,
                            size_t realDstCount) {
    int col_buffer_size = im2colParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, 0, col_buffer_size);
    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    constexpr int dstXStepInt32 = SRC_UNIT * DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateX)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + SRC_UNIT * i;
        auto inputOffset = inputOrigin + (sx + sfx * dilateX + (sy + sfy * dilateY) * iw) * UNIT;
        auto indexOffset = sfy * kw + sfx;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK       = inputOffset + (fx * dilateX + fy * dilateY * iw) * UNIT;
                auto indexStart   = indexOffset + fy * kw + fx;
                auto indexInside  = indexStart % 4;
                auto indexOutside = indexStart / 4;
                auto dstK0        = (int32_t*)colAddrI + indexOutside * dstXStepInt32 + indexInside;
                dstK0[0]          = *((int32_t*)inputK);
            }
        }
    }
}

static void _im2colCommon(int8_t* colAddr, const int8_t* inputOrigin,
                          const CPUConvolution::Im2ColParameter* im2colParameter, size_t xIndexStart,
                          size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, 0, col_buffer_size);
    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = iw * ih * UNIT;
    constexpr int dstXStepInt32 = SRC_UNIT * DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateX)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + SRC_UNIT * i;
        auto inputOffset = inputOrigin + (sx + sfx * dilateX + (sy + sfy * dilateY) * iw) * UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + (fx * dilateX + fy * dilateY * iw) * UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    const int yIndex      = indexStart + sz;
                    const int ySubOutside = yIndex / UNIT;
                    const int ySubInside  = yIndex % UNIT;
                    auto dstK0            = (int32_t*)colAddrI + ySubOutside * dstXStepInt32 + ySubInside;
                    dstK0[0]              = *((int32_t*)inputK);
                    inputK += srcZStep;
                }
            }
        }
    }
}
CPUConvInt8::~CPUConvInt8() {
    backend()->onReleaseBuffer(mWeightInt8.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mBiasInt32.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mScaleFloat.get(), Backend::STATIC);
}
CPUConvInt8::CPUConvInt8(Backend* backend, const MNN::Convolution2D* convParam, const std::vector<Tensor*>& inptus)
    : CPUConvolution(convParam->common(), backend) {
    const auto convCommon             = convParam->common();
    const auto kx                     = convCommon->kernelX();
    const auto ky                     = convCommon->kernelY();
    const auto kernelCount            = kx * ky;
    const auto srcCount               = inptus[0]->channel();
    const auto outputCount            = convCommon->outputCount();
    const auto outputCountUnit        = UP_DIV(outputCount, UNIT);
    const auto srcCountUnit           = UP_DIV(srcCount, UNIT);
    const auto totalKernelCountD8     = UP_DIV(srcCountUnit * kernelCount, 2);
    const auto totalKernelCountD8Div2 = UP_DIV(totalKernelCountD8, 2);
    mWeightInt8.reset(Tensor::createDevice<int8_t>({outputCountUnit, totalKernelCountD8Div2, UNIT, SRC_UNIT}));
    auto allocRes = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    const int oneTileLen         = mWeightInt8->stride(1);
    const int outputChnnelStride = mWeightInt8->stride(0);
    const auto weightSrc         = convParam->symmetricQuan()->weight()->data();
    auto weightDst               = mWeightInt8->host<int8_t>();
    memset(weightDst, 0, mWeightInt8->size());
    // reorder weight
    for (int k = 0; k < kernelCount; ++k) {
        const auto srcK = weightSrc + k;
        for (int y = 0; y < srcCount; ++y) {
            const int yOutSide    = y / UNIT;
            const int yInSide     = y % UNIT;
            const int yIndex      = yOutSide + k * srcCountUnit;
            const int ySubOutSide = yIndex / UNIT;
            const int ySubInSide  = yIndex % UNIT;

            auto dstY       = weightDst + ySubOutSide * oneTileLen + ySubInSide * UNIT + yInSide;
            const auto srcY = srcK + y * kernelCount;
            for (int x = 0; x < outputCount; ++x) {
                const int xOutSide = x / UNIT;
                const int xInSide  = x % UNIT;
                const int dstIndex = xOutSide * outputChnnelStride + xInSide * SRC_UNIT;
                const int srcIndex = x * kernelCount * srcCount;
                dstY[dstIndex]     = srcY[srcIndex];
            }
        }
    }
    const int outputChannleUp4 = ALIGN_UP4(outputCount);
    mBiasInt32.reset(Tensor::createDevice<int32_t>({outputChannleUp4}));
    allocRes = backend->onAcquireBuffer(mBiasInt32.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }
    auto biasPtr = mBiasInt32->host<int32_t>();
    memset(biasPtr, 0, outputChannleUp4 * sizeof(int32_t));
    memcpy(biasPtr, convParam->symmetricQuan()->bias()->data(), outputCount * sizeof(int32_t));

    mScaleFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    allocRes = backend->onAcquireBuffer(mScaleFloat.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }

    auto scalePtr = mScaleFloat->host<float>();
    memset(scalePtr, 0, outputChannleUp4 * sizeof(float));
    memcpy(scalePtr, convParam->symmetricQuan()->scale()->data(), outputCount * sizeof(float));

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.padX            = convCommon->padX();
    mIm2ColParamter.padY            = convCommon->padY();
    mIm2ColParamter.icDiv4          = srcCountUnit;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.kernelCountUnit = totalKernelCountD8Div2;

    mRelu    = convCommon->relu() || convCommon->relu6();
}

ErrorCode CPUConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];

    mIm2ColParamter.padY = mPadY;

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();

    mTileCount        = UP_DIV(output->height() * output->width(), DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCount);

    // set im2col tensor info
    mTempIm2ColBuffer.setType(DataType_DT_INT8);
    mTempIm2ColBuffer.buffer().dimensions = 3;
    mTempIm2ColBuffer.setLength(0, mThreadNums);
    mTempIm2ColBuffer.setLength(1, DST_XUNIT);
    mTempIm2ColBuffer.setLength(2, mWeightInt8->length(1) * SRC_UNIT);
    TensorUtils::setLinearLayout(&mTempIm2ColBuffer);

    // set reamin tensor info
    mTempRemainBuffer.setType(DataType_DT_INT8);
    mTempRemainBuffer.buffer().dimensions = 3;
    mTempRemainBuffer.setLength(0, mThreadNums);
    mTempRemainBuffer.setLength(1, DST_XUNIT);
    mTempRemainBuffer.setLength(2, ALIGN_UP4(output->channel()));
    TensorUtils::setLinearLayout(&mTempRemainBuffer);

    bool success = backend()->onAcquireBuffer(&mTempIm2ColBuffer, Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(&mTempRemainBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(&mTempIm2ColBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempRemainBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const int outputPlaneLen = output->height() * output->width();
    const int dstZStep       = outputPlaneLen * 4;

    const int batch                  = input->batch();
    const int ocDiv4                 = UP_DIV(output->channel(), 4);
    const auto kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;

    bool fastIm2Col = mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && mIm2ColParamter.icDiv4 % 2 == 0 &&
                      mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && mIm2ColParamter.padX == 0 &&
                      mIm2ColParamter.padY == 0;
    auto im2ColProcess = _im2colCommon;
    if (fastIm2Col) {
        im2ColProcess = _fastIm2Col;
    } else if (input->channel() <= 4) {
        im2ColProcess = _im2colCommonZ1;
    }

    const auto inputDataPtr = input->host<int8_t>();

    const auto weightDataPtr = mWeightInt8->host<int8_t>();
    const auto biasDataPtr   = mBiasInt32->host<int32_t>();
    const auto scaleDataPtr  = mScaleFloat->host<float>();
    auto im2colPtr           = mTempIm2ColBuffer.host<int8_t>();
    auto outputDataPtr       = output->host<int8_t>();
    auto tempRemainPtr       = mTempRemainBuffer.host<int8_t>();
    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcPtr = inputDataPtr + bIndex * input->stride(0);
        auto dstPtr       = outputDataPtr + bIndex * output->stride(0);

        auto threadFunction = [&](int tId) {
            auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer.stride(0);
            auto gemmOutputAddr = tempRemainPtr + tId * mTempRemainBuffer.stride(0);

            for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                const int xIndexStart  = tIndex * DST_XUNIT;
                const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);
                // im2col
                im2ColProcess(colAddr, srcPtr, &mIm2ColParamter, xIndexStart, realDstCount);
                auto outputInTilePtr = dstPtr + xIndexStart * UNIT;
                if (realDstCount == DST_XUNIT) {
                    MNNGemmInt8AddBiasScale_16x4_Unit(outputInTilePtr, colAddr, weightDataPtr, biasDataPtr,
                                                      scaleDataPtr, kernelCountUnitDouble, dstZStep * sizeof(int8_t),
                                                      ocDiv4);
                } else {
                    MNNGemmInt8AddBiasScale_16x4_Unit(gemmOutputAddr, colAddr, weightDataPtr, biasDataPtr, scaleDataPtr,
                                                      kernelCountUnitDouble, UNIT * DST_XUNIT * sizeof(int8_t), ocDiv4);
                    for (int z = 0; z < ocDiv4; ++z) {
                        auto outputZ = outputInTilePtr + z * dstZStep;
                        auto srcZ    = gemmOutputAddr + z * UNIT * DST_XUNIT;
                        memcpy(outputZ, srcZ, realDstCount * UNIT * sizeof(int8_t));
                    }
                }
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
            threadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        if (mRelu) {
            int threadNumber = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
            threadNumber     = std::min(threadNumber, ocDiv4);
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                for (int z = (int)tId; z < ocDiv4; z += threadNumber) {
                    MNNReluInt8(outputDataPtr + z * dstZStep, outputDataPtr + z * dstZStep, dstZStep);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }

    return NO_ERROR;
}

class CPUConvInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUConvInt8(backend, op->main_as_Convolution2D(), inputs);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConvInt8Creator, OpType_ConvInt8);

} // namespace MNN
