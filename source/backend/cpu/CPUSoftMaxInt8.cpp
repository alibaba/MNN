//
//  CPUSoftMaxInt8.cpp
//  MNNCPU
//
//  Created by jbyang on 2023/4/22.
//

#include "CPUSoftMaxInt8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"
#include "CPUTensorConvert.hpp"

namespace MNN {

CPUSoftmaxInt8::CPUSoftmaxInt8(Backend* backend, int axis) : Execution(backend), mAxis(axis), mStorage(2), mTempOutput(2), mNeedUnpackC4(false) {
    // do nothing.
}

const int kScaledDiffIntegerBits   = 5;
const int kAccumulationIntegerBits = 12;

ErrorCode CPUSoftmaxInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto inputQuant = TensorUtils::getQuantInfo(input);
    float beta  = 1.0;
    float scale = inputQuant[0];
    PreprocessSoftmaxScaling(beta, scale, kScaledDiffIntegerBits, &mInputMultiplier, &mInputLeftShift);
    mDiffMin = -1.0 * CalculateInputRadius(kScaledDiffIntegerBits, mInputLeftShift);

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;
    const int dimensions = input->buffer().dimensions;
    
    int axis = mAxis;
    if (axis < 0) {
        axis += input->dimensions();
    }
    mInside = 1; mOutside = 1;
    for (int i = 0; i < axis; ++i) {
        mOutside *= input->length(i);
    }
    mTargetAxis = input->length(axis);
    for (int i = axis + 1; i < dimensions; ++i) {
        mInside *= input->length(i);
    }

    mStorage.buffer().dim[0].extent = input->length(0);
    mStorage.buffer().dim[1].extent = input->stride(0);
    TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mStorage.buffer().dimensions    = 2;
    mStorage.buffer().type          = input->getType();
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    
    if (mNeedUnpackC4) {
        mTempOutput.buffer().dim[0].extent = output->length(0);
        mTempOutput.buffer().dim[1].extent = output->stride(0);
        TensorUtils::getDescribe(&mTempOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        mTempOutput.buffer().dimensions    = 2;
        mTempOutput.buffer().type          = input->getType();
        backend()->onAcquireBuffer(&mTempOutput, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mTempOutput, Backend::DYNAMIC);
    }
    
    return NO_ERROR;
}

void CPUSoftmaxInt8::QuantizedSoftmax(const uint8_t* inputData, int outerSize, int targetAxis,
                                              int32_t inputBetaMultiplier, int32_t inputBetaLeftShift,
                                               uint8_t* outputData, int threadNum) {
    using FixedPointScaledDiff = FixedPoint<int, kScaledDiffIntegerBits>;
    using FixedPointAccum      = FixedPoint<int, kAccumulationIntegerBits>;
    using FixedPoint0          = FixedPoint<int, 0>;

    const int depth            = targetAxis;
#ifdef MNN_USE_SSE
    int32_t zeroPoint   = 128;
    int32_t minValue    = 0;
    int32_t maxValue    = 255;
    const uint8_t* src_ = inputData;
    uint8_t* dst_       = outputData;
#else
    int32_t zeroPoint = 0;
    int32_t minValue  = -128;
    int32_t maxValue  = 127;
    const int8_t* src_ = (int8_t*)inputData;
    int8_t* dst_       = (int8_t*)outputData;
#endif
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        auto inputDataPtr = src_ + tId * depth;
        uint8_t* outputDataPtr = (uint8_t*)dst_ + tId * depth;
        for (int b = (int)tId; b < outerSize; b += threadNum, inputDataPtr += depth * threadNum, outputDataPtr += depth * threadNum) {
            // Determine the largest entry in the current row
            int8_t maxInRow = -128;
            {
                int c = 0;
#ifdef MNN_USE_NEON
                int8x16_t max16_0 = vdupq_n_s8(0);
                int8x16_t max16_1 = vdupq_n_s8(0);
                for (; c <= depth - 32; c += 32) {
                  max16_0 = vmaxq_s8(max16_0, vld1q_s8(inputDataPtr + c + 0));
                  max16_1 = vmaxq_s8(max16_1, vld1q_s8(inputDataPtr + c + 16));
                }
                int8x16_t max16 = vmaxq_s8(max16_0, max16_1);
                if (c <= depth - 16) {
                  max16 = vmaxq_s8(max16, vld1q_s8(inputDataPtr + c));
                  c += 16;
                }
                int8x8_t max8 = vmax_s8(vget_low_s8(max16), vget_high_s8(max16));
                if (c <= depth - 8) {
                  max8 = vmax_s8(max8, vld1_s8(inputDataPtr + c));
                  c += 8;
                }
                int8x8_t max4 = vmax_s8(max8, vext_s8(max8, max8, 4));
                int8x8_t max2 = vmax_s8(max4, vext_s8(max4, max4, 2));
                int8x8_t max1 = vpmax_s8(max2, max2);
                maxInRow = vget_lane_s8(max1, 0);
#endif
                for (; c < depth; ++c) {
                    maxInRow = std::max(maxInRow, static_cast<int8_t>(inputDataPtr[c] - zeroPoint));
                }
            }

#ifdef MNN_USE_NEON
            using FixedPointAccumInt32x4 = FixedPoint<int32x4_t, kAccumulationIntegerBits>;
            using FixedPointScaledDiffInt32x4 = FixedPoint<int32x4_t, kScaledDiffIntegerBits>;
            using FixedPoint0Int32x4 = FixedPoint<int32x4_t, 0>;
            FixedPoint0Int32x4 input_beta_multiplier_f0 = FixedPoint0Int32x4::FromScalarRaw(inputBetaMultiplier);
            int16x8_t max_in_row_s16 = vdupq_n_s16(maxInRow);
#endif

            FixedPointAccum sumOfExps = FixedPointAccum::Zero();
            {
                int c = 0;
#ifdef MNN_USE_NEON
                int32x4_t diff_min_s32 = vdupq_n_s32(mDiffMin);
                FixedPointAccumInt32x4 sum_of_exps_0 = FixedPointAccumInt32x4::Zero();
                FixedPointAccumInt32x4 sum_of_exps_1 = FixedPointAccumInt32x4::Zero();
                FixedPointAccumInt32x4 zeros = FixedPointAccumInt32x4::Zero();
                for (; c <= depth - 8; c += 8) {
                int16x8_t input_s16 = vmovl_s8(vld1_s8(inputDataPtr + c));
                int16x8_t input_diff_s16 =
                    vsubq_s16(input_s16, max_in_row_s16);
                int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
                int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
                int32x4_t mask_0 =
                    MaskIfGreaterThanOrEqual(input_diff_s32_0, diff_min_s32);
                int32x4_t mask_1 =
                    MaskIfGreaterThanOrEqual(input_diff_s32_1, diff_min_s32);
                FixedPointScaledDiffInt32x4 scaled_diff_0 =
                    input_beta_multiplier_f0 *
                    FixedPointScaledDiffInt32x4::FromRaw(
                        ShiftLeft(input_diff_s32_0, inputBetaLeftShift));
                FixedPointScaledDiffInt32x4 scaled_diff_1 =
                    input_beta_multiplier_f0 *
                    FixedPointScaledDiffInt32x4::FromRaw(
                        ShiftLeft(input_diff_s32_1, inputBetaLeftShift));
                FixedPointAccumInt32x4 exps_0 =
                    Rescale<kAccumulationIntegerBits>(
                        exp_on_negative_values(scaled_diff_0));
                FixedPointAccumInt32x4 exps_1 =
                    Rescale<kAccumulationIntegerBits>(
                        exp_on_negative_values(scaled_diff_1));
                FixedPointAccumInt32x4 masked_exps_0 =
                    SelectUsingMask(mask_0, exps_0, zeros);
                FixedPointAccumInt32x4 masked_exps_1 =
                    SelectUsingMask(mask_1, exps_1, zeros);
                sum_of_exps_0 = sum_of_exps_0 + masked_exps_0;
                sum_of_exps_1 = sum_of_exps_1 + masked_exps_1;
                }
                int32x4_t sum_of_exps_reduced_4 = (sum_of_exps_0 + sum_of_exps_1).raw();
                int32x2_t sum_of_exps_reduced_2 =
                    vadd_s32(vget_low_s32(sum_of_exps_reduced_4),
                            vget_high_s32(sum_of_exps_reduced_4));
                int32x2_t sum_of_exps_reduced_1 =
                    vpadd_s32(sum_of_exps_reduced_2, sum_of_exps_reduced_2);
                sumOfExps =
                    FixedPointAccum::FromRaw(vget_lane_s32(sum_of_exps_reduced_1, 0));
#endif
                for (; c < depth; ++c) {
                    int32_t inputDiff = (inputDataPtr[c] - zeroPoint) - maxInRow;
                    if (inputDiff >= mDiffMin) {
                        const int32_t inputDiffRescaled =
                            MultiplyByQuantizedMultiplierGreaterThanOne(inputDiff, inputBetaMultiplier, inputBetaLeftShift);
                        const FixedPointScaledDiff scaledDiffF8 = FixedPointScaledDiff::FromRaw(inputDiffRescaled);
                        sumOfExps = sumOfExps + Rescale<kAccumulationIntegerBits>(exp_on_negative_values(scaledDiffF8));
                    }
                }
            }

            int fixedSumOfExps  = sumOfExps.raw();
    #if defined(_MSC_VER)
            int headroomPlusOne;
            {
                unsigned long leading_zero = 0;
                if (_BitScanReverse(&leading_zero, static_cast<uint32_t>(fixedSumOfExps))) {
                    headroomPlusOne = 31 - leading_zero;
                } else {
                    headroomPlusOne = 31;
                }
            }
    #else
            int headroomPlusOne = __builtin_clz(static_cast<uint32_t>(fixedSumOfExps));
    #endif

            int numBitsOverUnit        = kAccumulationIntegerBits - headroomPlusOne;

            if (numBitsOverUnit + 31 - 8 > 31) {
                numBitsOverUnit = 8;
            }
            int32_t shiftedSumMinusOne = static_cast<int32_t>((static_cast<uint32_t>(fixedSumOfExps) << headroomPlusOne) -
                                                              (static_cast<uint32_t>(1) << 31));
            FixedPoint0 shiftedScale   = one_over_one_plus_x_for_x_in_0_1(FixedPoint0::FromRaw(shiftedSumMinusOne));

            {
                int c = 0;
#ifdef MNN_USE_NEON
                int16x8_t diff_min_s16 = vdupq_n_s16(mDiffMin);
                for (; c <= depth - 8; c += 8) {
                    int16x8_t input_s16 = vmovl_s8(vld1_s8(inputDataPtr + c));
                    int16x8_t input_diff_s16 =
                        vsubq_s16(input_s16, max_in_row_s16);
                    int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
                    int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
                    uint8x8_t mask = vmovn_u16(vcgeq_s16(input_diff_s16, diff_min_s16));
                    FixedPointScaledDiffInt32x4 scaled_diff_0 =
                        input_beta_multiplier_f0 *
                        FixedPointScaledDiffInt32x4::FromRaw(
                            ShiftLeft(input_diff_s32_0, inputBetaLeftShift));
                    FixedPointScaledDiffInt32x4 scaled_diff_1 =
                        input_beta_multiplier_f0 *
                        FixedPointScaledDiffInt32x4::FromRaw(
                            ShiftLeft(input_diff_s32_1, inputBetaLeftShift));
                    FixedPoint0Int32x4 exp_0 = exp_on_negative_values(scaled_diff_0);
                    FixedPoint0Int32x4 exp_1 = exp_on_negative_values(scaled_diff_1);
                    int32x4_t output_s32_0 = RoundingDivideByPOT(
                        vqrdmulhq_n_s32(exp_0.raw(), shiftedScale.raw()),
                        numBitsOverUnit + 31 - 8);
                    int32x4_t output_s32_1 = RoundingDivideByPOT(
                        vqrdmulhq_n_s32(exp_1.raw(), shiftedScale.raw()),
                        numBitsOverUnit + 31 - 8);
                    int16x8_t output_s16 =
                        vcombine_s16(vqmovn_s32(output_s32_0), vqmovn_s32(output_s32_1));
                    uint8x8_t output_s8 = vqmovun_s16(output_s16);
                    uint8x8_t masked_output = vbsl_u8(mask, output_s8, vdup_n_u8(0));
                    vst1_u8(outputDataPtr + c, masked_output);
                }
#endif
                for (; c < depth; ++c) {
                    int32_t inputDiff = (inputDataPtr[c] - zeroPoint) - maxInRow;
                    if (inputDiff >= mDiffMin) {
                        const int inputDiffRescaled =
                            MultiplyByQuantizedMultiplierGreaterThanOne(inputDiff, inputBetaMultiplier, inputBetaLeftShift);
                        const FixedPointScaledDiff scaledDiffF8 = FixedPointScaledDiff::FromRaw(inputDiffRescaled);
                        FixedPoint0 expIn0                      = exp_on_negative_values(scaledDiffF8);

                        int unsatOutput  = RoundingDivideByPOT((shiftedScale * expIn0).raw(), numBitsOverUnit + 31 - 8) + zeroPoint;
                        outputDataPtr[c] = std::max(std::min(unsatOutput, maxValue), minValue);
                         
                    }
                    else {
                        outputDataPtr[c] = zeroPoint;
                    }
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
}

ErrorCode CPUSoftmaxInt8::onExecute(const std::vector<MNN::Tensor*>& inputs,
                                            const std::vector<MNN::Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    Tensor* input       = inputs[0];
    Tensor* output      = outputs[0];
    uint8_t* inputData  = input->host<uint8_t>();
    uint8_t* outputData = output->host<uint8_t>();
    
    auto batch = input->batch();
    auto dimentions = input->dimensions();
    int areaInput = 1;
    for (int i = 2; i < dimentions; ++i) {
        areaInput *= input->length(i);
    }
    int threadNum = ((CPUBackend *)backend())->threadNumber();

    uint8_t* tempInputData = mStorage.host<uint8_t>();
    auto functions = ((CPUBackend*)backend())->functions();
    if (mNeedUnpackC4) {
        uint8_t* tempOutputData = mTempOutput.host<uint8_t>();
        CPUTensorConverter::convert(inputData, outputData, MNN_DATA_FORMAT_NC4HW4, MNN_DATA_FORMAT_NCHW, batch, areaInput, input->channel(), 1, functions);
        CPUTensorConverter::convert(outputData, tempInputData, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NHWC, mOutside, mInside, mTargetAxis, 1, functions);
        QuantizedSoftmax(tempInputData, mInside * mOutside, mTargetAxis, mInputMultiplier, mInputLeftShift, tempOutputData, threadNum);
        CPUTensorConverter::convert(tempOutputData, tempInputData, MNN_DATA_FORMAT_NHWC, MNN_DATA_FORMAT_NCHW, mOutside, mInside, mTargetAxis, 1, functions);
        CPUTensorConverter::convert(tempInputData, outputData, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NC4HW4, batch, areaInput, input->channel(), 1, functions);
    } else {
        CPUTensorConverter::convert(inputData, outputData, MNN_DATA_FORMAT_NCHW, MNN_DATA_FORMAT_NHWC, mOutside, mInside, mTargetAxis, 1, functions);
        QuantizedSoftmax(outputData, mInside * mOutside, mTargetAxis, mInputMultiplier, mInputLeftShift, tempInputData, threadNum);
        CPUTensorConverter::convert(tempInputData, outputData, MNN_DATA_FORMAT_NHWC, MNN_DATA_FORMAT_NCHW, mOutside, mInside, mTargetAxis, 1, functions);
    }
    
    return NO_ERROR;
}

Execution* CPUSoftmaxInt8::create(const MNN::Op *op, Backend *backend) {
    auto axis = op->main_as_Axis()->axis();
    return new CPUSoftmaxInt8(backend, axis);
}

}
