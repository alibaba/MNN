//
//  CPUQuantizedAdd.cpp
//  MNN
//
//  Created by MNN on 2018/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "backend/cpu/CPUQuantizedAdd.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"

namespace MNN {

CPUQuantizedAdd::CPUQuantizedAdd(Backend *backend, const Op *op) : Execution(backend) {
    mQuantizedAddParam = op->main_as_QuantizedAdd();
}

ErrorCode CPUQuantizedAdd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    mInput1Offset                   = -mQuantizedAddParam->input1QuantizedParam()->zeroPoint();
    mInput2Offset                   = -mQuantizedAddParam->input2QuantizedParam()->zeroPoint();
    mOutputOffset                   = mQuantizedAddParam->outputQuantizedParam()->zeroPoint();
    const int leftShift             = 20;
    const double twiceMaxInputScale = 2 * std::max(mQuantizedAddParam->input1QuantizedParam()->scale(),
                                                   mQuantizedAddParam->input2QuantizedParam()->scale());
    const double realInput1Multiplier = mQuantizedAddParam->input1QuantizedParam()->scale() / twiceMaxInputScale;
    const double realInput2Multiplier = mQuantizedAddParam->input2QuantizedParam()->scale() / twiceMaxInputScale;
    const double realOutputMultiplier =
        twiceMaxInputScale / ((1 << leftShift) * mQuantizedAddParam->outputQuantizedParam()->scale());

    QuantizeMultiplierSmallerThanOne(realInput1Multiplier, &mInput1Multiplier, &mInput1Shift);
    QuantizeMultiplierSmallerThanOne(realInput2Multiplier, &mInput2Multiplier, &mInput2Shift);
    QuantizeMultiplierSmallerThanOne(realOutputMultiplier, &mOutputMultiplier, &mOutputShift);

    CalculateActivationRangeUint8(
        mQuantizedAddParam->activationType(), mQuantizedAddParam->outputQuantizedParam()->zeroPoint(),
        mQuantizedAddParam->outputQuantizedParam()->scale(), &mOutputActivationMin, &mOutputActivationMax);

    int kReverseShiftResult1 = -mInput1Shift;
    int kReverseShiftResult2 = -mInput2Shift;

    int leftShift1  = kReverseShiftResult1 > 0 ? kReverseShiftResult1 : 0;
    mRightShift1 = kReverseShiftResult1 > 0 ? 0 : -kReverseShiftResult1;

    int leftShift2  = kReverseShiftResult2 > 0 ? kReverseShiftResult2 : 0;
    mRightShift2 = kReverseShiftResult2 > 0 ? 0 : -kReverseShiftResult2;

    mLeftShiftOut  = -mOutputShift > 0 ? -mOutputShift : 0;
    mRightShiftOut = -mOutputShift > 0 ? 0 : mOutputShift;

    mLeftShiftResult1 = (1 << leftShift) * ((1 << leftShift1));
    mLeftShiftResult2 = (1 << leftShift) * ((1 << leftShift2));

    MNN_ASSERT(leftShift + leftShift1 == leftShift);
    MNN_ASSERT(leftShift + leftShift2 == leftShift);

    return NO_ERROR;
}

ErrorCode CPUQuantizedAdd::onExecute(const std::vector<MNN::Tensor *> &inputs,
                                     const std::vector<MNN::Tensor *> &outputs) {
#ifdef MNN_USE_NEON
    int16x8_t input1OffsetVec, input2OffsetVec;
    int32x4_t outputOffsetVec, outputActivationMinVec, outputActivationMaxVec, leftShiftResult1Vec, leftShiftResult2Vec, input1MultiplierVec, input2MultiplierVec, outputMultiplierVec, leftShiftOutVec, rightShift1Vec, rightShift2Vec;
    input1OffsetVec        = vdupq_n_s16(mInput1Offset);
    input2OffsetVec        = vdupq_n_s16(mInput2Offset);
    outputOffsetVec        = vdupq_n_s32(mOutputOffset);
    outputActivationMinVec = vdupq_n_s32(mOutputActivationMin);
    outputActivationMaxVec = vdupq_n_s32(mOutputActivationMax);
    leftShiftResult1Vec    = vdupq_n_s32(mLeftShiftResult1);
    leftShiftResult2Vec    = vdupq_n_s32(mLeftShiftResult2);
    input1MultiplierVec    = vdupq_n_s32(mInput1Multiplier);
    input2MultiplierVec    = vdupq_n_s32(mInput2Multiplier);
    outputMultiplierVec    = vdupq_n_s32(mOutputMultiplier);
    leftShiftOutVec        = vdupq_n_s32((1 << mLeftShiftOut));
    rightShift1Vec      = vdupq_n_s32(-mRightShift1);
    rightShift2Vec      = vdupq_n_s32(-mRightShift2);
#endif
    uint8_t *input1Data = inputs[0]->host<uint8_t>();
    uint8_t *input2Data = inputs[1]->host<uint8_t>();
    uint8_t *outputData = outputs[0]->host<uint8_t>();

    int outputChannels = inputs[0]->channel();
    int size = inputs[0]->batch()*inputs[0]->height()*inputs[0]->width()*ROUND_UP(outputChannels, 4);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int countUnit    = UP_DIV(size, threadNumber);

    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        int realDstCount       = (int)ALIMIN(size - tId * countUnit, countUnit);
        uint8_t *curInput1Data = input1Data + tId * countUnit;
        uint8_t *curInput2Data = input2Data + tId * countUnit;
        uint8_t *curOutputData = outputData + tId * countUnit;

        int i = 0;

#ifdef MNN_USE_NEON

        for (; i <= realDstCount - 8; i += 8) {
            uint8x8_t input1Uint8 = vld1_u8(curInput1Data);
            int16x8_t input1S16   = vreinterpretq_s16_u16(vmovl_u8(input1Uint8));
            int16x8_t input1Val   = vaddq_s16(input1S16, input1OffsetVec);

            uint8x8_t input2Uint8 = vld1_u8(curInput2Data);
            int16x8_t input2S16   = vreinterpretq_s16_u16(vmovl_u8(input2Uint8));
            int16x8_t input2Val   = vaddq_s16(input2S16, input2OffsetVec);

            int32x4_t input10 = vmovl_s16(vget_low_s16(input1Val));
            int32x4_t input11 = vmovl_s16(vget_high_s16(input1Val));

            int32x4_t input20 = vmovl_s16(vget_low_s16(input2Val));
            int32x4_t input21 = vmovl_s16(vget_high_s16(input2Val));

            int32x4_t shiftedInput1ValVec0 = vmulq_s32(input10, leftShiftResult1Vec);
            int32x4_t shiftedInput1ValVec1 = vmulq_s32(input11, leftShiftResult1Vec);

            int32x4_t shiftedInput2ValVec0 = vmulq_s32(input20, leftShiftResult2Vec);
            int32x4_t shiftedInput2ValVec1 = vmulq_s32(input21, leftShiftResult2Vec);

            shiftedInput1ValVec0                = vqrdmulhq_s32(shiftedInput1ValVec0, input1MultiplierVec);
            const int32x4_t fixup00             = vshrq_n_s32(vandq_s32(shiftedInput1ValVec0, rightShift1Vec), 31);
            const int32x4_t fixedUpX00          = vqaddq_s32(shiftedInput1ValVec0, fixup00);
            const int32x4_t scaledInput1ValVec0 = vrshlq_s32(fixedUpX00, rightShift1Vec);

            shiftedInput1ValVec1                = vqrdmulhq_s32(shiftedInput1ValVec1, input1MultiplierVec);
            const int32x4_t fixup01             = vshrq_n_s32(vandq_s32(shiftedInput1ValVec1, rightShift1Vec), 31);
            const int32x4_t fixedUpX01          = vqaddq_s32(shiftedInput1ValVec1, fixup01);
            const int32x4_t scaledInput1ValVec1 = vrshlq_s32(fixedUpX01, rightShift1Vec);

            shiftedInput2ValVec0                = vqrdmulhq_s32(shiftedInput2ValVec0, input2MultiplierVec);
            const int32x4_t fixup20             = vshrq_n_s32(vandq_s32(shiftedInput2ValVec0, rightShift2Vec), 31);
            const int32x4_t fixedUpX20          = vqaddq_s32(shiftedInput2ValVec0, fixup20);
            const int32x4_t scaledInput2ValVec0 = vrshlq_s32(fixedUpX20, rightShift2Vec);

            shiftedInput2ValVec1                = vqrdmulhq_s32(shiftedInput2ValVec1, input2MultiplierVec);
            const int32x4_t fixup21             = vshrq_n_s32(vandq_s32(shiftedInput2ValVec1, rightShift2Vec), 31);
            const int32x4_t fixedUpX21          = vqaddq_s32(shiftedInput2ValVec1, fixup21);
            const int32x4_t scaledInput2ValVec1 = vrshlq_s32(fixedUpX21, rightShift2Vec);

            int32x4_t rawSum0 = vaddq_s32(scaledInput1ValVec0, scaledInput2ValVec0);
            int32x4_t rawSum1 = vaddq_s32(scaledInput1ValVec1, scaledInput2ValVec1);

            rawSum0 = RoundingDivideByPOT(
                SaturatingRoundingDoublingHighMul(vmulq_s32(rawSum0, leftShiftOutVec), outputMultiplierVec),
                mRightShiftOut);

            rawSum1 = RoundingDivideByPOT(
                SaturatingRoundingDoublingHighMul(vmulq_s32(rawSum1, leftShiftOutVec), outputMultiplierVec),
                mRightShiftOut);

            rawSum0 = vaddq_s32(rawSum0, outputOffsetVec);
            rawSum1 = vaddq_s32(rawSum1, outputOffsetVec);

            rawSum0 = vmaxq_s32(rawSum0, outputActivationMinVec);
            rawSum1 = vmaxq_s32(rawSum1, outputActivationMinVec);

            rawSum0 = vminq_s32(rawSum0, outputActivationMaxVec);
            rawSum1 = vminq_s32(rawSum1, outputActivationMaxVec);

            int16x4_t rawSumS16n0 = vqmovn_s32(rawSum0);
            int16x4_t rawSumS16n1 = vqmovn_s32(rawSum1);

            int16x8_t resS16 = vcombine_s16(rawSumS16n0, rawSumS16n1);

            uint8x8_t resU8n0 = vqmovun_s16(resS16);

            vst1_u8(curOutputData, resU8n0);

            curInput1Data += 8;
            curInput2Data += 8;
            curOutputData += 8;
        }
        curInput1Data -= i;
        curInput2Data -= i;
        curOutputData -= i;
#endif
        for (; i < realDstCount; i++) {
            const int32_t input1Val        = mInput1Offset + curInput1Data[i];
            const int32_t input2Val        = mInput2Offset + curInput2Data[i];
            const int32_t shiftedInput1Val = input1Val * mLeftShiftResult1;
            const int32_t shiftedInput2Val = input2Val * mLeftShiftResult2;
            const int32_t scaledInput1Val  = RoundingDivideByPOT(
                SaturatingRoundingDoublingHighMul(shiftedInput1Val, mInput1Multiplier), mRightShift1);
            const int32_t scaledInput2Val = RoundingDivideByPOT(
                SaturatingRoundingDoublingHighMul(shiftedInput2Val, mInput2Multiplier), mRightShift2);
            const int32_t rawSum = scaledInput1Val + scaledInput2Val;
            const int32_t rawOutput =
                RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(rawSum * (1 << mLeftShiftOut), mOutputMultiplier),
                                    mRightShiftOut) + mOutputOffset;
            const int32_t clampedOutput = std::min(mOutputActivationMax, std::max(mOutputActivationMin, rawOutput));
            curOutputData[i]            = static_cast<uint8_t>(clampedOutput);
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

class CPUQuantizedAddCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedAdd(backend, op);
    }
};
} // namespace MNN
#endif
namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUQuantizedAddCreator, OpType_QuantizedAdd);
}
