//
//  Arm82Convolution.cpp
//  MNN
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82Convolution.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82Convolution3x3.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

#ifndef MNN_USE_NEON
static void MNNGemmFP16C8_UNIT(FLOAT16 *dst, const FLOAT16 *src, const FLOAT16 *weight, const FLOAT16 *bias,
                               size_t src_loop, size_t dst_step, size_t dst_loop, size_t relu, size_t relu6,
                               size_t realDstCount) {
    const auto dst_step_tmp = dst_step / sizeof(FLOAT16);

    for (int dz = 0; dz < dst_loop; ++dz) {
        const auto weight_dz = weight + dz * src_loop * (ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT);
        const auto bias_dz   = bias + dz * ARMV82_CHANNEL_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        for (int w = 0; w < DST_XUNIT; ++w) {
            const auto src_x = src + w * ARMV82_CHANNEL_UNIT;
            auto dst_x       = dst_z + w * ARMV82_CHANNEL_UNIT;
            FLOAT16 dstTemp[ARMV82_CHANNEL_UNIT];

            memcpy(dstTemp, bias_dz, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);

            // MAC
            for (int sz = 0; sz < src_loop; ++sz) {
                const auto weight_sz = weight_dz + (ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT) * sz;
                const auto src_z     = src_x + sz * DST_XUNIT * ARMV82_CHANNEL_UNIT;

                for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                    for (int i = 0; i < ARMV82_CHANNEL_UNIT; ++i) {
                        dstTemp[j] += src_z[i] * weight_sz[i * ARMV82_CHANNEL_UNIT + j];
                    }
                }
            } // end MAC

            if (relu) {
                for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                    if (dstTemp[j] < 0) {
                        dstTemp[j] = 0;
                    }
                }
            }
            if (relu6) {
                for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
                    if (dstTemp[j] < 0) {
                        dstTemp[j] = 0;
                    }
                    if (dstTemp[j] > 6) {
                        dstTemp[j] = 6.0;
                    }
                }
            }

            memcpy(dst_x, dstTemp, sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
        }
    }
}
#endif

static void Im2ColTransformer(FLOAT16 *dst, const FLOAT16 *src, CPUConvolution::Im2ColParameter *im2colParam,
                              size_t xIndexStart, size_t realDstCount) {
    {
        const int colBufferSize = im2colParam->kernelCountUnit * DST_XUNIT * ARMV82_CHANNEL_UNIT * sizeof(FLOAT16);
        memset(dst, 0, colBufferSize);
    }
    // src data format is nc8hw8

    const auto ih = im2colParam->ih;
    const auto iw = im2colParam->iw;
    // const auto oh = im2colParameter->oh;
    const auto ow               = im2colParam->ow;
    const auto kh               = im2colParam->kernelY;
    const auto kw               = im2colParam->kernelX;
    const auto dilateX          = im2colParam->dilateX;
    const auto dilateY          = im2colParam->dilateY;
    const auto icDiv4           = im2colParam->icDiv4;
    const auto srcChannleStride = iw * ih * ARMV82_CHANNEL_UNIT;
    const auto stridex          = im2colParam->strideX;
    const auto stridey          = im2colParam->strideY;
    const auto padx             = im2colParam->padX;
    const auto pady             = im2colParam->padY;
    constexpr int dstXStep      = ARMV82_CHANNEL_UNIT * DST_XUNIT;

    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % ow;
        int oy     = xIndex / ow;
        int sx     = ox * stridex - padx;
        int sy     = oy * stridey - pady;
        int sfy    = ALIMAX(0, (UP_DIV(-sy, dilateY)));
        int efy    = ALIMIN(kh, UP_DIV(ih - sy, dilateY));
        int sfx    = ALIMAX(0, (UP_DIV(-sx, dilateX)));
        int efx    = ALIMIN(kw, UP_DIV(iw - sx, dilateX));
        int fyC    = efy - sfy;
        int fxC    = efx - sfx;

        auto colAddrI    = dst + ARMV82_CHANNEL_UNIT * i;
        auto inputOffset = src + (sx + sfx * dilateX + (sy + sfy * dilateY) * iw) * ARMV82_CHANNEL_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;

        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputUnit  = inputOffset + (fx * dilateX + fy * dilateY * iw) * ARMV82_CHANNEL_UNIT;
                auto indexStart = (indexOffset + (fy * kw + fx) * icDiv4) * dstXStep;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    auto dstUnit = colAddrI + indexStart + sz * dstXStep;
                    memcpy(dstUnit, inputUnit, ARMV82_CHANNEL_UNIT * sizeof(FLOAT16));
                    inputUnit += srcChannleStride;
                }
            }
        }
    }

    // shuffle channel
#ifdef MNN_USE_NEON
    if (realDstCount > (DST_XUNIT / 2)) {
        MNNShuffleChannelC8(dst, dst, (size_t)im2colParam->kernelCountUnit, 0);
    } else {
        MNNShuffleChannelC8(dst, dst, (size_t)im2colParam->kernelCountUnit, 1);
    }
#endif
}

static void Im2ColTransformer1x1(FLOAT16 *dst, const FLOAT16 *src, CPUConvolution::Im2ColParameter *im2colParam,
                                 size_t xIndexStart, size_t realDstCount) {
    {
        const int colBufferSize = im2colParam->kernelCountUnit * DST_XUNIT * ARMV82_CHANNEL_UNIT * sizeof(FLOAT16);
        memset(dst, 0, colBufferSize);
    }
    // src data format is nc8hw8
    const auto ih = im2colParam->ih;
    const auto iw = im2colParam->iw;

    const auto icDiv8           = im2colParam->icDiv4;
    const auto srcChannleStride = iw * ih * ARMV82_CHANNEL_UNIT;
    constexpr int dstXStep      = ARMV82_CHANNEL_UNIT * DST_XUNIT;
    const auto srcStartPtr      = src + xIndexStart * ARMV82_CHANNEL_UNIT;

    for (int c = 0; c < icDiv8; ++c) {
        memcpy(dst + c * dstXStep, srcStartPtr + c * srcChannleStride,
               sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT * realDstCount);
    }

// shuffle channel
#ifdef MNN_USE_NEON
    if (realDstCount > (DST_XUNIT / 2)) {
        MNNShuffleChannelC8(dst, dst, (size_t)im2colParam->kernelCountUnit, 0);
    } else {
        MNNShuffleChannelC8(dst, dst, (size_t)im2colParam->kernelCountUnit, 1);
    }
#endif
}

Arm82Convolution::Arm82Convolution(const MNN::Convolution2D *convParam, Backend *bn) : Execution(bn) {
    const auto convCommon   = convParam->common();
    mCommon                 = convCommon;
    const int kx            = convCommon->kernelX();
    const int ky            = convCommon->kernelY();
    const int kernelCount   = kx * ky;
    int inputChannel        = convCommon->inputCount();
    const int outputChannel = convCommon->outputCount();
    if (inputChannel == 0) {
        if (convParam->quanParameter()) {
            inputChannel = convParam->quanParameter()->buffer()->size() / (2 * kernelCount * outputChannel);
        } else {
            inputChannel = convParam->weight()->size() / (kernelCount * outputChannel);
        }
    }
    const int inputChannelUnit  = UP_DIV(inputChannel, ARMV82_CHANNEL_UNIT);
    const int outputChannelUnit = UP_DIV(outputChannel, ARMV82_CHANNEL_UNIT);

    const int totalKernelCountUnit = kernelCount * inputChannelUnit;
    mWeightFp16.reset(Tensor::createDevice<uint16_t>(
        {outputChannelUnit, totalKernelCountUnit, ARMV82_CHANNEL_UNIT, ARMV82_CHANNEL_UNIT}));
    auto allocRes = bn->onAcquireBuffer(mWeightFp16.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }

    auto weightFp16DstPtr = mWeightFp16->host<FLOAT16>();
    memset(weightFp16DstPtr, 0, mWeightFp16->size());

    const FLOAT16 *fp16WeightPtr = nullptr;
    std::vector<FLOAT16> weightFp16;
    if (convParam->quanParameter()) {
        MNN_ASSERT(convParam->quanParameter()->type() == 3);
        // the data type of weight is fp16
        fp16WeightPtr = reinterpret_cast<const FLOAT16 *>(convParam->quanParameter()->buffer()->data());
    } else {
        // the data type of weight is fp32, then quantize weight to be fp16 data type
        int size = convParam->weight()->size();
        weightFp16.resize(size);
        MNNQuantizeFP16(weightFp16.data(), convParam->weight()->data(), size);
        fp16WeightPtr = weightFp16.data();
    }

    auto weightFp16SrcPtr = fp16WeightPtr;

    const int oneChannleKernelSize = kernelCount * inputChannel;

#ifdef MNN_USE_NEON
    int curOcChannel   = 0;
    auto reorderWeight = [&](int ocUnit, int ocUnitNum, const FLOAT16 *weightSrc, FLOAT16 *weightDst) {
        for (int oc = 0; oc < ocUnitNum; ++oc) {
            auto weightDstOcUnit   = weightDst + oc * kernelCount * inputChannelUnit * ARMV82_CHANNEL_UNIT * ocUnit;
            const auto weightSrcOc = weightSrc + oc * ocUnit * oneChannleKernelSize;
            for (int k = 0; k < kernelCount; ++k) {
                auto weightDstK       = weightDstOcUnit + k * inputChannelUnit * ARMV82_CHANNEL_UNIT * ocUnit;
                const auto weightSrcK = weightSrcOc + k;
                for (int y = 0; y < inputChannel; ++y) {
                    const int yOutSide     = y / ARMV82_CHANNEL_UNIT;
                    const int yInSide      = y % ARMV82_CHANNEL_UNIT;
                    auto weightDstIc       = weightDstK + yOutSide * ARMV82_CHANNEL_UNIT * ocUnit + yInSide * ocUnit;
                    const auto weigthSrcIc = weightSrcK + y * kernelCount;

                    for (int x = 0; x < ocUnit; ++x) {
                        if (curOcChannel + x < outputChannel) {
                            weightDstIc[x] = weigthSrcIc[x * oneChannleKernelSize];
                        }
                    }
                }
            }
            curOcChannel += ocUnit;
        }
    };
    const int ocDivDoubleUnit = outputChannelUnit / 2;
    // reorder weight in double ARMV82_CHANNEL_UNIT
    reorderWeight((ARMV82_CHANNEL_UNIT * 2), ocDivDoubleUnit, weightFp16SrcPtr, weightFp16DstPtr);
    auto weightRemainDst = weightFp16DstPtr + kernelCount * inputChannelUnit * ARMV82_CHANNEL_UNIT * ocDivDoubleUnit *
                                                  (ARMV82_CHANNEL_UNIT * 2);
    auto weightRemainSrc = weightFp16SrcPtr + kernelCount * inputChannel * ocDivDoubleUnit * (ARMV82_CHANNEL_UNIT * 2);
    if (outputChannelUnit % 2 == 1) {
        // reorder weight in ARMV82_CHANNEL_UNIT
        reorderWeight(ARMV82_CHANNEL_UNIT, 1, weightRemainSrc, weightRemainDst);
    }
#else
    // reorder weight
    const int ocUnitStride = inputChannelUnit * ARMV82_CHANNEL_UNIT * kernelCount * ARMV82_CHANNEL_UNIT;
    for (int k = 0; k < kernelCount; ++k) {
        const auto weightSrcK = weightFp16SrcPtr + k;
        auto weightDstK       = weightFp16DstPtr + k * inputChannelUnit * ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT;
        for (int y = 0; y < inputChannel; ++y) {
            const int yOutSide = y / ARMV82_CHANNEL_UNIT;
            const int yInSide  = y % ARMV82_CHANNEL_UNIT;

            auto dstY =
                weightDstK + yOutSide * ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT + yInSide * ARMV82_CHANNEL_UNIT;
            const auto srcY = weightSrcK + y * kernelCount;
            for (int x = 0; x < outputChannel; ++x) {
                const int xOutSide = x / ARMV82_CHANNEL_UNIT;
                const int xInSide  = x % ARMV82_CHANNEL_UNIT;
                const int dstIndex = xOutSide * ocUnitStride + xInSide;
                const int srcIndex = x * oneChannleKernelSize;
                dstY[dstIndex]     = srcY[srcIndex];
            }
        }
    }
#endif

    mBiasFp16.reset(Tensor::createDevice<uint16_t>({outputChannelUnit * ARMV82_CHANNEL_UNIT}));
    allocRes = bn->onAcquireBuffer(mBiasFp16.get(), Backend::STATIC);
    if (!allocRes) {
        mValid = false;
        return;
    }

    // TODO, bias is fp32, save bias also in fp16?
    auto biasDstPtr = mBiasFp16->host<FLOAT16>();
    memset(biasDstPtr, 0, mBiasFp16->size());
    MNNQuantizeFP16(biasDstPtr, convParam->bias()->data(), outputChannel);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.padX            = convCommon->padX();
    mIm2ColParamter.padY            = convCommon->padY();
    mIm2ColParamter.icDiv4          = inputChannelUnit;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.kernelCountUnit = totalKernelCountUnit;

    mRelu6 = convCommon->relu6();
    mRelu  = convCommon->relu();
}

Arm82Convolution::~Arm82Convolution() {
    if (mWeightFp16 != nullptr) {
        backend()->onReleaseBuffer(mWeightFp16.get(), Backend::STATIC);
    }
    if (mBiasFp16 != nullptr) {
        backend()->onReleaseBuffer(mBiasFp16.get(), Backend::STATIC);
    }
}

ErrorCode Arm82Convolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    mIm2ColParamter.padX = mCommon->padX();
    mIm2ColParamter.padY = mCommon->padY();
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth   = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight  = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        mIm2ColParamter.padX = padNeededWidth / 2;
        mIm2ColParamter.padY = padNeededHeight / 2;
    }

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();

    mTileCount        = UP_DIV(output->height() * output->width(), DST_XUNIT);
    const int threads = std::max(1, static_cast<Arm82Backend *>(backend())->numberThread());
    mThreadNums       = std::min(threads, mTileCount);

    mIm2ColBuffer.setType(DataType_DT_BFLOAT16);
    mIm2ColBuffer.buffer().dimensions = 3;
    mIm2ColBuffer.setLength(0, mThreadNums);
    mIm2ColBuffer.setLength(1, DST_XUNIT);
    mIm2ColBuffer.setLength(2, mWeightFp16->length(1) * ARMV82_CHANNEL_UNIT);
    TensorUtils::setLinearLayout(&mIm2ColBuffer);

    mRemainBuffer.setType(DataType_DT_BFLOAT16);
    mRemainBuffer.buffer().dimensions = 3;
    mRemainBuffer.setLength(0, mThreadNums);
    mRemainBuffer.setLength(1, DST_XUNIT);
    mRemainBuffer.setLength(2, UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT) * ARMV82_CHANNEL_UNIT);
    TensorUtils::setLinearLayout(&mRemainBuffer);
    bool success = backend()->onAcquireBuffer(&mIm2ColBuffer, Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(&mRemainBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(&mIm2ColBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mRemainBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode Arm82Convolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input               = inputs[0];
    auto output              = outputs[0];
    const int outputPlaneLen = output->height() * output->width();

    const int dstZStep        = outputPlaneLen * ARMV82_CHANNEL_UNIT;
    const int batch           = input->batch();
    const int ocDiv8          = UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT);
    const int kernelCountUnit = mIm2ColParamter.kernelCountUnit;

    const auto inputDataPtr  = input->host<FLOAT16>();
    const auto weightDataPtr = mWeightFp16->host<FLOAT16>();
    const auto biasDataPtr   = mBiasFp16->host<FLOAT16>();
    auto im2ColPtr           = mIm2ColBuffer.host<FLOAT16>();
    auto outputDataPtr       = output->host<FLOAT16>();
    auto remainDataPtr       = mRemainBuffer.host<FLOAT16>();

    auto im2ColProcess = Im2ColTransformer;
    bool useFastIm2Col = mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && mIm2ColParamter.strideX == 1 &&
                         mIm2ColParamter.strideY == 1 && mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0;

    if (useFastIm2Col) {
        im2ColProcess = Im2ColTransformer1x1;
    }

    const int inBatchStride  = ROUND_UP(input->channel(), ARMV82_CHANNEL_UNIT) * input->height() * input->width();
    const int outBatchStride = ocDiv8 * dstZStep;
    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto srcBatchPtr = inputDataPtr + bIndex * inBatchStride;
        auto dstBatchPtr       = outputDataPtr + bIndex * outBatchStride;

        auto threadFunction = [&](int tId) {
            auto im2ColCurPtr  = im2ColPtr + tId * mIm2ColBuffer.stride(0);
            auto gemmOutputPtr = remainDataPtr + tId * mRemainBuffer.stride(0);

            for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                const int xIndexStart  = tIndex * DST_XUNIT;
                const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);

                Im2ColTransformer(im2ColCurPtr, srcBatchPtr, &mIm2ColParamter, xIndexStart, realDstCount);

                auto outputCurTilePtr = dstBatchPtr + xIndexStart * ARMV82_CHANNEL_UNIT;

                if (realDstCount == DST_XUNIT) {
                    // compute one tile
                    MNNGemmFP16C8_UNIT(outputCurTilePtr, im2ColCurPtr, weightDataPtr, biasDataPtr, kernelCountUnit,
                                       dstZStep * sizeof(FLOAT16), ocDiv8, mRelu, mRelu6, realDstCount);
                } else {
                    // compute the remain
                    MNNGemmFP16C8_UNIT(gemmOutputPtr, im2ColCurPtr, weightDataPtr, biasDataPtr, kernelCountUnit,
                                       ARMV82_CHANNEL_UNIT * DST_XUNIT * sizeof(FLOAT16), ocDiv8, mRelu, mRelu6,
                                       realDstCount);
                    for (int z = 0; z < ocDiv8; ++z) {
                        auto outputz = outputCurTilePtr + z * dstZStep;
                        auto srcz    = gemmOutputPtr + z * ARMV82_CHANNEL_UNIT * DST_XUNIT;
                        memcpy(outputz, srcz, realDstCount * ARMV82_CHANNEL_UNIT * sizeof(FLOAT16));
                    }
                }
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, mThreadNums)
        threadFunction((int)tId);
#ifdef MNN_USE_THREAD_POOL
        MNN_CONCURRENCY_ARM82_END();
#else
        MNN_CONCURRENCY_END();
#endif
    }

    return NO_ERROR;
}

class Arm82ConvolutionCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto convParam = op->main_as_Convolution2D();
        // avoid other quantize method entry this creator
        if(convParam->quanParameter() && convParam->quanParameter()->type() != 3){
            return nullptr;
        }
        
#ifdef __aarch64__
        const auto param = convParam->common();
        if (param->kernelX() == 3 && param->kernelY() == 3 && param->strideX() == 1 && param->strideY() == 1 &&
            param->dilateX() == 1 && param->dilateY() == 1) {
            return new Arm82Convolution3x3(convParam, backend);
        }
#endif
        return new Arm82Convolution(convParam, backend);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Convolution, Arm82ConvolutionCreator);

} // namespace MNN
