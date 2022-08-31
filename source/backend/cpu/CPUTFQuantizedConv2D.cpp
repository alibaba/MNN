//
//  CPUTFQuantizedConv2D.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "backend/cpu/CPUTFQuantizedConv2D.hpp"
#include <math.h>
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
#define SRC_UNIT 16

//SRC_UNIT/UNIT
#define SRC_C4_UNIT 4

// ugly macro compatible with MNNGemmInt8ToFloat32_XX
#ifdef DST_XUNIT
#undef DST_XUNIT
#endif
// One Tile Compute DST_XUNIT * outputChannel 's number
#ifdef __aarch64__
#define DST_XUNIT 4
#else
#define DST_XUNIT 2
#endif

extern "C" {
void MNNQuanToDestUint8(uint8_t* outputInTile, const int32_t* gemmOutputAddr, const int32_t* biasData, size_t ocUnit,
                        size_t realDstCount, size_t dstZStep, size_t srcZstep,
                        const MNN::CPUTFQuantizedConv2D::QuanParameter* parameter);
void MNNLoadU8AndSum(int32_t* inputSum, int8_t* colAddr, const uint8_t* inputOrigin, size_t srcZStep, size_t icDiv8,
                        size_t realDstCount, size_t mFilterOffset);
void MNNGemmint8to32_8x4_Unit(int32_t* dst, const int8_t* src, const int8_t* weight, const int32_t* inputSummer, size_t src_depth_quad,
                                  size_t dst_step, size_t dst_depth_quad);

}

#ifndef MNN_USE_NEON
void MNNGemmint8to32_8x4_Unit(int32_t* dst, const int8_t* src, const int8_t* weight, const int32_t* inputSummer, size_t src_depth_quad,
                              size_t dst_step, size_t dst_depth_quad) {
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto weight_dz = weight + src_depth_quad * dz * SRC_UNIT * UNIT;
        auto dst_z     = dst + dz * dst_step / sizeof(int32_t);
        for (int w = 0; w < DST_XUNIT; ++w) {
            auto dst_x = dst_z + 4 * w;
            ::memset(dst_x, 0, UNIT * sizeof(int32_t));
            auto src_x = src + SRC_UNIT * w;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                auto weight_sz = weight_dz +SRC_UNIT * UNIT * sz;
                auto src_z     = src_x + sz * DST_XUNIT * SRC_UNIT;
                for (int j = 0; j < UNIT; ++j) {
                    auto weight_j = weight_sz + j * SRC_UNIT;
                    for (int i = 0; i < SRC_UNIT; ++i) {
                        auto s0 = (int32_t)(src_z[i+0]);
                        auto s1 = (int32_t)(weight_j[i+0]);
                        dst_x[j] += s0 * s1;
                    }
                }
            }
            for (int j = 0; j < UNIT; ++j) {
                dst_x[j] -= inputSummer[w];
            }
        }
    }
}

void MNNLoadU8AndSum(int32_t* inputSum, int8_t* colAddr, const uint8_t* inputOrigin, size_t srcZStep, size_t icDiv8,
                     size_t realDstCount, size_t mFilterOffset) {
    for (int i = 0; i < realDstCount; ++i) {
        inputSum[i]   = 0;
        auto colAddrI = colAddr + SRC_UNIT * i;
        auto inputK   = inputOrigin + UNIT * i;
        for (int sz = 0; sz < icDiv8; ++sz) {
            auto inputZ0      = inputK + srcZStep * (SRC_C4_UNIT * sz + 0);
            auto inputZ1      = inputK + srcZStep * (SRC_C4_UNIT * sz + 1);
            auto inputZ2      = inputK + srcZStep * (SRC_C4_UNIT * sz + 2);
            auto inputZ3      = inputK + srcZStep * (SRC_C4_UNIT * sz + 3);
            auto indexOutside = sz;

            auto dstK0 = colAddrI + indexOutside * SRC_UNIT * DST_XUNIT;
            auto dstK1 = dstK0 + UNIT;
            auto dstK2 = dstK1 + UNIT;
            auto dstK3 = dstK2 + UNIT;
            for (int u = 0; u < UNIT; ++u) {
                dstK0[u] = (int)inputZ0[u] - 128;
                dstK1[u] = (int)inputZ1[u] - 128;
                dstK2[u] = (int)inputZ2[u] - 128;
                dstK3[u] = (int)inputZ3[u] - 128;
                inputSum[i] += ((int32_t)dstK0[u] + (int32_t)dstK1[u] + (int32_t)dstK2[u] + (int32_t)dstK3[u]) * mFilterOffset;
            }
        }
    }
}

void MNNQuanToDestUint8(uint8_t* outputInTile, const int32_t* gemmOutputAddr, const int32_t* biasData, size_t ocUnit,
                        size_t realDstCount, size_t dstZStep, size_t srcZstep,
                        const MNN::CPUTFQuantizedConv2D::QuanParameter* parameter) {
    dstZStep = dstZStep / sizeof(uint8_t);
    srcZstep = srcZstep / sizeof(int32_t);
    for (int dz = 0; dz < ocUnit; ++dz) {
        auto dstZ  = outputInTile + dz * dstZStep;
        auto srcZ  = gemmOutputAddr + dz * srcZstep;
        auto biasZ = biasData + dz * UNIT;
        for (int x = 0; x < realDstCount; ++x) {
            auto dstX = dstZ + x * UNIT;
            auto srcX = srcZ + x * UNIT;
            for (int i = 0; i < UNIT; i++) {
                int result = srcX[i];
                int acc    = result + biasZ[i];
                acc        = MNN::RoundingDivideByPOT(
                    MNN::SaturatingRoundingDoublingHighMul(acc * (1 << parameter->mOutputShiftBefore),
                                                           parameter->mOutputMultiplier),
                    -parameter->mOutputShiftAfter);
                acc += parameter->mOutputOffset;
                acc     = std::max(acc, parameter->mOutputActivationMin);
                acc     = std::min(acc, parameter->mOutputActivationMax);
                dstX[i] = static_cast<uint8_t>(acc);
            }
        }
    }
}
#endif

namespace MNN {

CPUTFQuantizedConv2D::CPUTFQuantizedConv2D(Backend* backend, const Op* TFQuantizedConv2DOp) : Execution(backend) {
    mTfQuantizedConv2D_param = TFQuantizedConv2DOp->main_as_TfQuantizedConv2D();

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    auto outputChannel               = mTfQuantizedConv2D_param->common()->outputCount();
    auto kx                          = mTfQuantizedConv2D_param->common()->kernelX();
    auto ky                          = mTfQuantizedConv2D_param->common()->kernelY();
    int inputChannel                 = mTfQuantizedConv2D_param->weight()->size() / outputChannel / kx / ky;
    auto outputChannelUnit           = UP_DIV(outputChannel, UNIT);
    auto inputChannelUnit            = UP_DIV(inputChannel, UNIT);
    mIm2ColParamter                  = new ConvolutionCommon::Im2ColParameter;
    mIm2ColParamter->dilateX         = mTfQuantizedConv2D_param->common()->dilateX();
    mIm2ColParamter->dilateY         = mTfQuantizedConv2D_param->common()->dilateY();
    mIm2ColParamter->strideX         = mTfQuantizedConv2D_param->common()->strideX();
    mIm2ColParamter->strideY         = mTfQuantizedConv2D_param->common()->strideY();
    mIm2ColParamter->kernelX         = mTfQuantizedConv2D_param->common()->kernelX();
    mIm2ColParamter->kernelY         = mTfQuantizedConv2D_param->common()->kernelY();
    mIm2ColParamter->padX            = mTfQuantizedConv2D_param->common()->padX();
    mIm2ColParamter->padY            = mTfQuantizedConv2D_param->common()->padY();
    mIm2ColParamter->icDiv4          = inputChannelUnit;
    mIm2ColParamter->kernelCountUnit = UP_DIV(inputChannelUnit * kx * ky, SRC_C4_UNIT);

    mQuanParameter = new QuanParameter;

    float inputScale  = mTfQuantizedConv2D_param->inputQuantizedParam()->scale();
    float filterScale = mTfQuantizedConv2D_param->filterQuantizedParam()->scale();

    {
        double realMultiplier          = 0.0;
        const double inputProductScale = inputScale * filterScale;
        const double outputScale       = mTfQuantizedConv2D_param->outputQuantizedParam()->scale();

        MNN_ASSERT(inputProductScale >= 0);
        realMultiplier = inputProductScale / outputScale;

        MNN_ASSERT(realMultiplier < 1.0);
        int shift = 0;
        QuantizeMultiplierSmallerThanOne(realMultiplier, &mQuanParameter->mOutputMultiplier, &shift);
        shift = -shift;
        if (shift < 0) {
            mQuanParameter->mOutputShiftBefore = 0;
            mQuanParameter->mOutputShiftAfter  = shift;
        } else {
            mQuanParameter->mOutputShiftBefore = shift;
            mQuanParameter->mOutputShiftAfter  = 0;
        }

        mFusedActivationFunction = mTfQuantizedConv2D_param->activationType();
        CalculateActivationRangeUint8(mFusedActivationFunction,
                                      mTfQuantizedConv2D_param->outputQuantizedParam()->zeroPoint(),
                                      mTfQuantizedConv2D_param->outputQuantizedParam()->scale(),
                                      &mQuanParameter->mOutputActivationMin, &mQuanParameter->mOutputActivationMax);
    }
    mQuanParameter->mOutputOffset = mTfQuantizedConv2D_param->outputQuantizedParam()->zeroPoint();

    auto src                = mTfQuantizedConv2D_param->weight()->data();
    int32_t offsetFilter    = mTfQuantizedConv2D_param->filterQuantizedParam()->zeroPoint() - 128;
    auto totalKernelCountD8 = UP_DIV(inputChannelUnit * kx * ky, SRC_C4_UNIT);
    mWeight.reset(Tensor::create<int8_t>(std::vector<int>{outputChannelUnit, totalKernelCountD8, UNIT, SRC_UNIT}));
    ::memset(mWeight->host<int8_t>(), (int8_t)offsetFilter, mWeight->size());

    std::shared_ptr<Tensor> mWeightSum;
    mWeightSum.reset(Tensor::create<int32_t>(std::vector<int>{outputChannelUnit, 4}));
    ::memset(mWeightSum->host<int32_t>(), 0, mWeightSum->size());

    mQuanParameter->mFilterOffset = offsetFilter;
    mQuanParameter->mInputOffset  = mTfQuantizedConv2D_param->inputQuantizedParam()->zeroPoint() - 128;
    mQuanParameter->mOffsetAdd =
        mQuanParameter->mFilterOffset * mQuanParameter->mInputOffset * totalKernelCountD8 * SRC_UNIT;
    auto dst        = mWeight->host<int8_t>();
    int kernelCount = kx * ky;
    auto weightSum  = mWeightSum->host<int32_t>();
    for (int i = 0; i < outputChannel; ++i) {
        weightSum[i] = (int32_t)offsetFilter * totalKernelCountD8 * SRC_UNIT;
    }

    // weight format : hwio -> oc/4, (hw ic/4) / 2, oc4, (hw ic/4) % 2 ic4
    for (int k = 0; k < kernelCount; ++k) {
        auto srcK = src + k * inputChannel * outputChannel;
        for (int y = 0; y < inputChannel; ++y) {
            int yOutSide    = y / UNIT;
            int yInside     = y % UNIT;
            int yIndex      = yOutSide + k * inputChannelUnit;
            int ySubOutside = yIndex / SRC_C4_UNIT;
            int ySubInside  = yIndex % SRC_C4_UNIT;

            auto dstY = dst + ySubOutside * UNIT * SRC_UNIT + ySubInside * UNIT + yInside;
            auto srcY = srcK + y * outputChannel;
            for (int x = 0; x < outputChannel; ++x) {
                int xOutSide = x / UNIT;
                int xInside  = x % UNIT;

                auto dstX = dstY + xOutSide * mWeight->stride(0) + xInside * SRC_UNIT;
                auto srcX = srcY + x;

                dstX[0] = (int)srcX[0] - 128;
                if (dstX[0] == -128) {
                    dstX[0] = -127;
                }

                weightSum[x] += ((int32_t)dstX[0] - (int32_t)offsetFilter);
            }
        }
    }

    auto originBiasData = mTfQuantizedConv2D_param->bias()->data();
    mBias.reset(outputChannelUnit * 4);
    auto biasData = mBias.get();

    // Sum[0, kx*ky*sz](x-x0)*(w-w0) = Sum(xw) - Sum(x)*w0 - Sum(w)*x0 + x0w0*(kx*ky*sz)
    // Let bias[oz] = bias[oz] - Sum[0, kx*ky*sz](w)*x0 + x0w0*(kx*ky*sz)
    for (int i = 0; i < outputChannel; ++i) {
        biasData[i] = originBiasData[i] - weightSum[i] * mQuanParameter->mInputOffset + mQuanParameter->mOffsetAdd;
    }
}

CPUTFQuantizedConv2D::~CPUTFQuantizedConv2D() {
    delete mQuanParameter;
    delete mIm2ColParamter;
}

ErrorCode CPUTFQuantizedConv2D::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input        = inputs[0];
    auto output       = outputs[0];
    auto outputWidth  = output->width();
    auto outputHeight = output->height();
    auto inputWidth   = input->width();
    auto inputHeight  = input->height();

    auto common       = mTfQuantizedConv2D_param->common();
    auto strideX      = common->strideX();
    auto strideY      = common->strideY();
    auto filterWidth  = common->kernelX();
    auto filterHeight = common->kernelY();

    if (common->padMode() == PadMode::PadMode_VALID) {
        mIm2ColParamter->padX = ((outputWidth - 1) * strideX + filterWidth - inputWidth + 1) / 2;
        mIm2ColParamter->padY = ((outputHeight - 1) * strideY + filterHeight - inputHeight + 1) / 2;
    } else {
        mIm2ColParamter->padX = ((outputWidth - 1) * strideX + filterWidth - inputWidth) / 2;
        mIm2ColParamter->padY = ((outputHeight - 1) * strideY + filterHeight - inputHeight) / 2;
    }

    int outputChannel = common->outputCount();

    auto outputChannelUnit = UP_DIV(outputChannel, UNIT);
    auto kernelCountUnit   = mIm2ColParamter->kernelCountUnit;
    mIm2ColParamter->iw    = inputWidth;
    mIm2ColParamter->ih    = inputHeight;
    mIm2ColParamter->ow    = outputWidth;
    mIm2ColParamter->oh    = outputHeight;

    int tileCount = UP_DIV(outputWidth * outputHeight, DST_XUNIT);
    mThreadNumber = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    mThreadNumber = std::min(mThreadNumber, tileCount);

    mTempBuffer.buffer().type          = halide_type_of<int8_t>();
    mTempBuffer.buffer().dimensions    = 3;
    mTempBuffer.buffer().dim[0].extent = mThreadNumber;
    mTempBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempBuffer.buffer().dim[2].extent = kernelCountUnit * SRC_UNIT;
    TensorUtils::setLinearLayout(&mTempBuffer);

    mTempDstBuffer.buffer().type          = halide_type_of<int32_t>();
    mTempDstBuffer.buffer().dimensions    = 3;
    mTempDstBuffer.buffer().dim[0].extent = mThreadNumber;
    mTempDstBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempDstBuffer.buffer().dim[2].extent = outputChannelUnit * UNIT;
    TensorUtils::setLinearLayout(&mTempDstBuffer);

    mTempInputSum.buffer().type          = halide_type_of<int32_t>();
    mTempInputSum.buffer().dimensions    = 2;
    mTempInputSum.buffer().dim[0].extent = mThreadNumber;
    mTempInputSum.buffer().dim[1].extent = DST_XUNIT;
    TensorUtils::setLinearLayout(&mTempInputSum);

    backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    backend()->onAcquireBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    backend()->onAcquireBuffer(&mTempInputSum, Backend::DYNAMIC);

    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempInputSum, Backend::DYNAMIC);

    return NO_ERROR;
}

static void _im2ColCommon(int32_t* inputSum, int8_t* colAddr, const uint8_t* inputOrigin,
                          const CPUTFQuantizedConv2D::QuanParameter* quanParamter,
                          const ConvolutionCommon::Im2ColParameter* im2ColParameter, size_t xIndexStart,
                          size_t realDstCount) {
    int colBufferSize = im2ColParameter->kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(uint8_t);
    ::memset(colAddr, (int8_t)quanParamter->mInputOffset, colBufferSize);
    auto ih        = im2ColParameter->ih;
    auto iw        = im2ColParameter->iw;
    auto kh        = im2ColParameter->kernelY;
    auto kw        = im2ColParameter->kernelX;
    auto dilateX   = im2ColParameter->dilateX;
    auto dilateY   = im2ColParameter->dilateY;
    auto icDiv4    = im2ColParameter->icDiv4;
    auto srcZStep  = iw * ih * UNIT;
    int countSumC8 = im2ColParameter->kernelCountUnit;
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
                auto inputK     = inputOffset + (fx * dilateX) * UNIT + (fy * dilateY) * iw * UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    auto inputZ       = inputK + srcZStep * sz;
                    auto index        = indexStart + sz;
                    auto indexInside  = index % SRC_C4_UNIT;
                    auto indexOutside = index / SRC_C4_UNIT;

                    auto dstK         = colAddrI + indexOutside * SRC_UNIT * DST_XUNIT + UNIT * indexInside;
                    //TODO Optimize it
                    for (int j=0; j<UNIT; ++j) {
                        dstK[j] = (int32_t)inputZ[j] - 128;
                    }
                    //*((int32_t*)dstK) = *((int32_t*)inputZ);
                }
            }
        }
        int32_t inputSumValue = 0;
#ifdef MNN_USE_NEON
        int32x2_t inputSumValueC4 = vmov_n_s32(0);
#endif
        for (int j = 0; j < countSumC8; ++j) {
            auto colAddrIJ = colAddrI + j * SRC_UNIT * DST_XUNIT;
#ifdef MNN_USE_NEON
            auto p0 = vld1_s8(colAddrIJ + 0);
            auto p1 = vld1_s8(colAddrIJ + 8);
            auto q0 = vpaddl_s8(p0);
            auto q1 = vpaddl_s8(p1);
            inputSumValueC4 += vpaddl_s16(q0);
            inputSumValueC4 += vpaddl_s16(q1);
#else
            for (int k = 0; k < SRC_UNIT; ++k) {
                inputSumValue += colAddrIJ[k];
            }
#endif
        }
#ifdef MNN_USE_NEON
        inputSumValue = inputSumValueC4[0] + inputSumValueC4[1];
#endif
        inputSum[i] = inputSumValue * quanParamter->mFilterOffset;
    }
}

ErrorCode CPUTFQuantizedConv2D::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor* input = inputs[0];

    const int strideX = mIm2ColParamter->strideX;
    const int strideY = mIm2ColParamter->strideY;
    auto batchs       = input->batch();
    auto ic           = input->channel();
    auto iw           = input->width();
    auto ih           = input->height();
    auto output       = outputs[0];
    auto oc           = output->channel();
    auto oh           = output->height();
    auto ow           = output->width();

    auto ocUnit = UP_DIV(oc, UNIT);
    int icDiv4  = UP_DIV(ic, UNIT);
    int kh      = mIm2ColParamter->kernelY;
    int kw      = mIm2ColParamter->kernelX;

    auto kernelCountUnit = mIm2ColParamter->kernelCountUnit;
    int outputCount      = ow * oh;
    int outputCountTile  = UP_DIV(outputCount, DST_XUNIT);

    bool fastMode = kw == 1 && kh == 1 && strideX == 1 && strideY == 1 && mIm2ColParamter->padY == 0 &&
                    mIm2ColParamter->padX == 0 && icDiv4 % SRC_C4_UNIT == 0;
    auto gemmFunction = MNNGemmint8to32_8x4_Unit;
    const int* biasData = mBias.get();

    for (int batchIndex = 0; batchIndex < batchs; ++batchIndex) {
        auto inputOrigin  = input->host<uint8_t>() + batchIndex * input->stride(0);
        auto weightOrigin = mWeight->host<int8_t>();
        auto outputOrigin = output->host<uint8_t>() + batchIndex * output->stride(0);

        MNN_CONCURRENCY_BEGIN(tId, mThreadNumber) {
            auto colAddr        = mTempBuffer.host<int8_t>() + tId * mTempBuffer.buffer().dim[0].stride;
            auto gemmOutputAddr = mTempDstBuffer.host<int32_t>() + tId * mTempDstBuffer.buffer().dim[0].stride;
            auto inputSum       = mTempInputSum.host<int32_t>() + mTempInputSum.stride(0) * tId;

            for (int tIndex = (int)tId; tIndex < outputCountTile; tIndex += mThreadNumber) {
                int xIndexStart  = tIndex * DST_XUNIT;
                int realDstCount = ALIMIN(outputCount - xIndexStart, DST_XUNIT);
                /*Im2Col Begin*/
                if (fastMode) {
                    MNNLoadU8AndSum(inputSum, colAddr, inputOrigin + UNIT * xIndexStart, iw * ih * UNIT, icDiv4 / SRC_C4_UNIT,
                                    realDstCount, mQuanParameter->mFilterOffset);
                } else {
                    _im2ColCommon(inputSum, colAddr, inputOrigin, mQuanParameter, mIm2ColParamter, xIndexStart,
                                  realDstCount);
                }

                /*Im2Col End*/

                // GEMM
                gemmFunction(gemmOutputAddr, colAddr, weightOrigin, inputSum, kernelCountUnit, UNIT * DST_XUNIT * sizeof(int32_t),
                                          ocUnit);

                /*Copy Data to Real Output*/
                auto outputInTile = outputOrigin + xIndexStart * UNIT;
                MNNQuanToDestUint8(outputInTile, gemmOutputAddr, biasData, ocUnit, realDstCount,
                                   ow * oh * UNIT * sizeof(uint8_t), DST_XUNIT * UNIT * sizeof(int32_t),
                                   mQuanParameter);
            }
        }

        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUTFQuantizedConv2DCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUTFQuantizedConv2D(backend, op);
    }
};
} // namespace MNN
#endif
namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUTFQuantizedConv2DCreator, OpType_TfQuantizedConv2D);
}
