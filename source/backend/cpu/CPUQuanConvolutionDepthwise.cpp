//
//  CPUQuanConvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2018/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "backend/cpu/CPUQuanConvolutionDepthwise.hpp"
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#define UNIT 4
extern "C" {
void MNNConvRunForUnitDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t fw, size_t fh,
                                     const MNN::ConstConvolutionParameter* parameter, const int32_t* biasData);
void MNNConvRunForLineDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t width,
                                     MNN::ConstConvolutionParameter* parameters, const int32_t* biasData);
}

struct MNN::ConstConvolutionParameter {
    size_t kw;
    size_t kh;
    size_t weightYStep;
    size_t dilateXStep;
    size_t dilateYStep;
    size_t strideXStep;
    int32_t outputMultiplier;
    int32_t outputShiftBefore;
    int32_t outputShiftAfter;
    int32_t outputOffset;
    int32_t outputActivationMin;
    int32_t outputActivationMax;
};

#ifndef MNN_USE_NEON
void MNNConvRunForUnitDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t fw, size_t fh,
                                     const MNN::ConstConvolutionParameter* parameter, const int32_t* biasData) {
    int fx, fy;
    int dstTemp[UNIT];
    for (int i = 0; i < UNIT; ++i) {
        dstTemp[i] = 0;
    }
    auto dilateYStep       = parameter->dilateYStep / sizeof(int16_t);
    auto dilateXStep       = parameter->dilateXStep / sizeof(int16_t);
    auto weightYStep       = parameter->weightYStep / sizeof(int16_t);
    const int16_t* srcZ    = src;
    const int16_t* weightZ = weight;
    for (fy = 0; fy < fh; ++fy) {
        const int16_t* srcY    = srcZ + fy * dilateYStep;
        const int16_t* weightY = weightZ + fy * weightYStep;
        for (fx = 0; fx < fw; ++fx) {
            const int16_t* weightX = weightY + UNIT * fx;
            const int16_t* srcX    = srcY + fx * dilateXStep;
            for (int j = 0; j < UNIT; ++j) {
                dstTemp[j] += ((int32_t)srcX[j]) * ((int32_t)weightX[j]);
            }
        }
    }
    for (int i = 0; i < UNIT; i++) {
        int acc = dstTemp[i] + biasData[i];
        acc     = MNN::SaturatingRoundingDoublingHighMul(acc * (1 << parameter->outputShiftBefore),
                                                     parameter->outputMultiplier);
        acc     = MNN::RoundingDivideByPOT(acc, -parameter->outputShiftAfter);
        acc += parameter->outputOffset;
        acc    = std::max(acc, parameter->outputActivationMin);
        acc    = std::min(acc, parameter->outputActivationMax);
        dst[i] = static_cast<uint8_t>(acc);
    }
}

void MNNConvRunForLineDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t width,
                                     MNN::ConstConvolutionParameter* parameters, const int32_t* biasData) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        uint8_t* dstX = dst + dx * UNIT;
        auto srcX     = src + dx * parameters->strideXStep / sizeof(int16_t);
        MNNConvRunForUnitDepthWiseUint8(dstX, srcX, weight, parameters->kw, parameters->kh, parameters, biasData);
    }
}
#endif

namespace MNN {

CPUQuanConvolutionDepthwise::CPUQuanConvolutionDepthwise(Backend* backend, const Op* CPUDepthwiseOp)
    : Execution(backend) {
    mLayerParam              = CPUDepthwiseOp->main_as_TfQuantizedConv2D();
    auto commonParam         = mLayerParam->common();
    mPadMode                 = commonParam->padMode();
    mStrideH                 = commonParam->strideY();
    mStrideW                 = commonParam->strideX();
    mDepthMultiplier         = mLayerParam->depthMultiplier();
    mFusedActivationFunction = mLayerParam->activationType();
    auto layer               = mLayerParam->common();
    int kw                   = layer->kernelX();
    int kh                   = layer->kernelY();
    int outputCount          = commonParam->outputCount();
    int depthQuad            = UP_DIV(outputCount, UNIT);
    int planeStride          = kw * kh * UNIT;

    const uint8_t* tempWeight = mLayerParam->weight()->data();
    int kernelSize            = depthQuad * UNIT * kw * kh;
    mBias.reset(ALIGN_UP4(mLayerParam->bias()->size()));
    mBias.clear();
    ::memcpy(mBias.get(), mLayerParam->bias()->data(), mLayerParam->bias()->size() * sizeof(int32_t));

    mWeight.reset(kernelSize);
    mWeight.clear();
    auto weight       = mWeight.get();
    auto filterOffset = mLayerParam->filterQuantizedParam()->zeroPoint();
    for (int c = 0; c < outputCount; c++) {
        int plane  = c / UNIT;
        int offset = c % UNIT;
        for (int i = 0; i < kh * kw; i++) {
            int16_t* dst = weight + plane * planeStride + offset + i * UNIT;
            *dst         = (int16_t)((int32_t)tempWeight[i * outputCount + c] - filterOffset);
        }
    }
    mConstParameter = new ConstConvolutionParameter;
}

CPUQuanConvolutionDepthwise::~CPUQuanConvolutionDepthwise() {
    delete mConstParameter;
}

inline int ComputePadding(int stride, int dilationRate, int inSize, int filterSize, int outSize) {
    int effectiveFilterSize = (filterSize - 1) * dilationRate + 1;
    int padding             = ((outSize - 1) * stride + effectiveFilterSize - inSize) / 2;
    return padding > 0 ? padding : 0;
}

ErrorCode CPUQuanConvolutionDepthwise::onResize(const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    auto input       = inputs[0];
    auto inputWidth  = input->width();
    auto inputHeight = input->height();

    auto common              = mLayerParam->common();
    mFusedActivationFunction = mLayerParam->activationType();

    int threadNumber                = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    mTempBuffer.buffer().type       = halide_type_of<int16_t>();
    mTempBuffer.buffer().dimensions = 4;
    mTempBuffer.setLength(0, threadNumber);
    mTempBuffer.setLength(1, inputHeight);
    mTempBuffer.setLength(2, inputWidth);
    mTempBuffer.setLength(3, UNIT);
    TensorUtils::setLinearLayout(&mTempBuffer);

    bool res = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    mConstParameter->dilateXStep = common->dilateX() * UNIT * sizeof(int16_t);
    mConstParameter->dilateYStep = common->dilateY() * inputWidth * UNIT * sizeof(int16_t);
    mConstParameter->strideXStep = common->strideX() * UNIT * sizeof(int16_t);
    mConstParameter->kh          = common->kernelY();
    mConstParameter->kw          = common->kernelX();
    mConstParameter->weightYStep = sizeof(int16_t) * common->kernelX() * UNIT;
    float inputScale             = mLayerParam->inputQuantizedParam()->scale();
    float filterScale            = mLayerParam->filterQuantizedParam()->scale();
    {
        double realMultiplier          = 0.0;
        const double inputProductScale = inputScale * filterScale;
        const double outputScale       = mLayerParam->outputQuantizedParam()->scale();
        realMultiplier                 = inputProductScale / outputScale;

        int exponent;
        QuantizeMultiplier(realMultiplier, &mConstParameter->outputMultiplier, &exponent);
        if (exponent < 0) {
            mConstParameter->outputShiftBefore = 0;
            mConstParameter->outputShiftAfter  = exponent;
        } else {
            mConstParameter->outputShiftBefore = exponent;
            mConstParameter->outputShiftAfter  = 0;
        }
        CalculateActivationRangeUint8(mFusedActivationFunction, mLayerParam->outputQuantizedParam()->zeroPoint(),
                                      mLayerParam->outputQuantizedParam()->scale(),
                                      &mConstParameter->outputActivationMin, &mConstParameter->outputActivationMax);
        mConstParameter->outputOffset = mLayerParam->outputQuantizedParam()->zeroPoint();
    }
    mDilateX   = mLayerParam->common()->dilateX();
    mDilateY   = mLayerParam->common()->dilateY();
    mZeroPoint = mLayerParam->inputQuantizedParam()->zeroPoint();

    const int outputWidth  = outputs[0]->width();
    const int outputHeight = outputs[0]->height();

    int filterHeight = (int)mConstParameter->kh;
    int filterWidth  = (int)mConstParameter->kw;

    mPaddingHeight = ComputePadding(mStrideH, 1, inputHeight, filterHeight, outputHeight);
    mPaddingWidth  = ComputePadding(mStrideW, 1, inputWidth, filterWidth, outputWidth);

    // Compute Mid Rect
    ml = 0; mt = 0; mr = outputWidth; mb = outputHeight;
    for (; ml * mStrideW - mPaddingWidth < 0 && ml < outputWidth; ml++) {
        // do nothing
    }
    for (; mt * mStrideH - mPaddingHeight < 0 && mt < outputHeight; mt++) {
        // do nothing
    }
    for (; (mr - 1) * mStrideW - mPaddingWidth + (filterWidth - 1) * mDilateX >= inputWidth && mr > ml; mr--) {
        // do nothing
    }
    for (; (mb - 1) * mStrideH - mPaddingHeight + (filterHeight - 1) * mDilateY >= inputHeight && mb > mt; mb--) {
        // do nothing
    }

    mDstYStep    = outputWidth * UNIT;
    mSrcYStep    = inputWidth * UNIT;
    mWeightZStep = filterHeight * filterWidth * UNIT;

    return NO_ERROR;
}

ErrorCode CPUQuanConvolutionDepthwise::onExecute(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    const Tensor* input = inputs[0];
    Tensor* output      = outputs[0];

    const int outputBatch  = outputs[0]->batch();
    const int outputWidth  = outputs[0]->width();
    const int outputHeight = outputs[0]->height();

    const int inputHeight  = inputs[0]->height();
    const int inputWidth   = inputs[0]->width();
    const int inputChannel = inputs[0]->channel();

    int filterHeight = (int)mConstParameter->kh;
    int filterWidth  = (int)mConstParameter->kw;

    auto bias = mBias.get();

    auto runBasic = [&](uint8_t* dstZ, const int16_t* srcZ, const int16_t* weightDZ, int L, int T, int R, int B,
                        const int32_t* biasData) {
        for (int dy = T; dy < B; ++dy) {
            uint8_t* dstY = dstZ + dy * mDstYStep;
            int srcStartY = dy * mStrideH - mPaddingHeight;
            int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, mDilateY)));
            int efy       = ALIMIN(filterHeight, UP_DIV(inputHeight - srcStartY, mDilateY));
            auto srcDY    = srcZ + (srcStartY + sfy * mDilateY) * mSrcYStep;
            auto weightDY = weightDZ + sfy * filterWidth * UNIT;
            for (int dx = L; dx < R; ++dx) {
                uint8_t* dstX = dstY + UNIT * dx;
                int srcStartX = dx * mStrideW - mPaddingWidth;
                auto srcDX    = srcDY + srcStartX * UNIT;
                int sfx       = ALIMAX(0, (UP_DIV(-srcStartX, mDilateX)));
                int efx       = ALIMIN(filterWidth, UP_DIV(inputWidth - srcStartX, mDilateX));

                MNNConvRunForUnitDepthWiseUint8(dstX, srcDX + (sfx * mDilateX) * UNIT, weightDY + UNIT * sfx,
                                                efx - sfx, efy - sfy, mConstParameter, biasData);
            }
        }
    };
    int icDiv4       = UP_DIV(inputChannel, 4);
    int threadNumber = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    threadNumber     = std::min(threadNumber, icDiv4);
    for (int batchIndex = 0; batchIndex < outputBatch; ++batchIndex) {
        const uint8_t* srcOrigin = input->host<uint8_t>() + batchIndex * input->stride(0);
        auto dstOrigin           = output->host<uint8_t>() + batchIndex * output->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            auto colBuffer = mTempBuffer.host<int16_t>() + mTempBuffer.stride(0) * tId;
            for (int z = (int)tId; z < icDiv4; z += threadNumber) {
                auto srcZ = srcOrigin + z * inputWidth * inputHeight * UNIT;
                MNNUInt8ToInt16WithOffsetC4Fast(colBuffer, srcZ, mZeroPoint, inputHeight * inputWidth, 1, 0, 0);
                const int32_t* curBiasPtr = bias + z * UNIT;
                uint8_t* dstZ             = dstOrigin + z * outputWidth * outputHeight * UNIT;

                const int16_t* weightDZ = mWeight.get() + z * mWeightZStep;

                runBasic(dstZ, colBuffer, weightDZ, 0, 0, outputWidth, mt, curBiasPtr);
                runBasic(dstZ, colBuffer, weightDZ, 0, mb, outputWidth, outputHeight, curBiasPtr);
                runBasic(dstZ, colBuffer, weightDZ, 0, mt, ml, mb, curBiasPtr);
                runBasic(dstZ, colBuffer, weightDZ, mr, mt, outputWidth, mb, curBiasPtr);

                if (mr > ml) {
                    for (int dy = mt; dy < mb; ++dy) {
                        uint8_t* dstY        = dstZ + dy * mDstYStep;
                        int srcStartY        = dy * mStrideH - mPaddingHeight;
                        const int16_t* srcDY = colBuffer + srcStartY * mSrcYStep;

                        MNNConvRunForLineDepthWiseUint8(dstY + ml * UNIT, srcDY + (ml * mStrideW - mPaddingWidth) * UNIT,
                                                        weightDZ, mr - ml, mConstParameter, curBiasPtr);
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUDepthwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUQuanConvolutionDepthwise(backend, op);
    }
};
} // namespace MNN
#endif
namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUDepthwiseCreator, OpType_QuantizedDepthwiseConv2D);
};
