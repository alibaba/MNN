//
//  CPUQuantizedSoftmax.cpp
//  MNN
//
//  Created by MNN on 2018/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "backend/cpu/CPUQuantizedSoftmax.hpp"
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "core/Macro.h"

namespace MNN {

template <typename T>
CPUQuantizedSoftmax<T>::CPUQuantizedSoftmax(Backend* backend, const Op* op) : Execution(backend) {
    auto quantizedSoftmax_param = op->main_as_QuantizedSoftmax();
    mBeta                       = quantizedSoftmax_param->beta();
    mInputScale                 = quantizedSoftmax_param->inputScale();
}

const int kScaledDiffIntegerBits   = 5;
const int kAccumulationIntegerBits = 12;

template <typename T>
ErrorCode CPUQuantizedSoftmax<T>::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    float beta  = mBeta;
    float scale = mInputScale;
    PreprocessSoftmaxScaling(beta, scale, kScaledDiffIntegerBits, &mInputMultiplier, &mInputLeftShift);
    mDiffMin = -1.0 * CalculateInputRadius(kScaledDiffIntegerBits, mInputLeftShift);

    Tensor* input       = inputs[0];
    Tensor* output      = outputs[0];

    MNN_ASSERT(2 == input->buffer().dimensions || 4 == input->buffer().dimensions);

    mInputDims.clear();
    mOutputDims.clear();
    if (4 == input->buffer().dimensions) {
        for (int i = 0; i < input->buffer().dimensions; i++) {
            mInputDims.push_back(input->buffer().dim[i].extent);
        }
        for (int i = 0; i < output->buffer().dimensions; i++) {
            mOutputDims.push_back(output->buffer().dim[i].extent);
        }
    } else {
        mInputDims.push_back(input->buffer().dim[0].extent);
        mInputDims.push_back(1);
        mInputDims.push_back(1);
        mInputDims.push_back(input->buffer().dim[1].extent);

        mOutputDims.push_back(input->buffer().dim[0].extent);
        mOutputDims.push_back(1);
        mOutputDims.push_back(1);
        mOutputDims.push_back(input->buffer().dim[1].extent);
    }

    return NO_ERROR;
}

template <typename T>
void CPUQuantizedSoftmax<T>::QuantizedSoftmax(const uint8_t* inputData, const std::vector<int>& inputDims,
                                              int32_t inputBetaMultiplier, int32_t inputBetaLeftShift,
                                              uint8_t* outputData, const std::vector<int>& outputDims) {
    using FixedPointScaledDiff = FixedPoint<int, kScaledDiffIntegerBits>;
    using FixedPointAccum      = FixedPoint<int, kAccumulationIntegerBits>;
    using FixedPoint0          = FixedPoint<int, 0>;

    const int outerSize = inputDims.at(0) * inputDims.at(1) * inputDims.at(2);
    const int depth     = inputDims.at(3);

    for (int b = 0; b < outerSize; ++b) {
        const uint8_t* inputDataPtr = inputData + b * depth;
        uint8_t* outputDataPtr      = outputData + b * depth;

        // Determine the largest entry in the current row
        uint8_t maxInRow = 0;
        {
            int c = 0;
            for (; c < depth; ++c) {
                maxInRow = std::max(maxInRow, inputDataPtr[c]);
            }
        }

        FixedPointAccum sumOfExps = FixedPointAccum::Zero();
        {
            int c = 0;
            for (; c < depth; ++c) {
                int32_t inputDiff = static_cast<int32_t>(inputDataPtr[c]) - maxInRow;
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
        int32_t shiftedSumMinusOne = static_cast<int32_t>((static_cast<uint32_t>(fixedSumOfExps) << headroomPlusOne) -
                                                          (static_cast<uint32_t>(1) << 31));
        FixedPoint0 shiftedScale   = one_over_one_plus_x_for_x_in_0_1(FixedPoint0::FromRaw(shiftedSumMinusOne));

        {
            int c = 0;
            for (; c < depth; ++c) {
                int32_t inputDiff = static_cast<int32_t>(inputDataPtr[c]) - maxInRow;
                if (inputDiff >= mDiffMin) {
                    const int inputDiffRescaled =
                        MultiplyByQuantizedMultiplierGreaterThanOne(inputDiff, inputBetaMultiplier, inputBetaLeftShift);
                    const FixedPointScaledDiff scaledDiffF8 = FixedPointScaledDiff::FromRaw(inputDiffRescaled);
                    FixedPoint0 expIn0                      = exp_on_negative_values(scaledDiffF8);
                    int unsatOutput  = RoundingDivideByPOT((shiftedScale * expIn0).raw(), numBitsOverUnit + 31 - 8);
                    outputDataPtr[c] = std::max(std::min(unsatOutput, 255), 0);
                } else {
                    outputDataPtr[c] = 0;
                }
            }
        }
    }
}

template <typename T>
ErrorCode CPUQuantizedSoftmax<T>::onExecute(const std::vector<MNN::Tensor*>& inputs,
                                            const std::vector<MNN::Tensor*>& outputs) {
    Tensor* input       = inputs[0];
    Tensor* output      = outputs[0];
    uint8_t* inputData  = input->host<uint8_t>();
    uint8_t* outputData = output->host<uint8_t>();

    QuantizedSoftmax(inputData, mInputDims, mInputMultiplier, mInputLeftShift, outputData, mOutputDims);

    return NO_ERROR;
}

class CPUQuantizedSoftmaxCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUQuantizedSoftmax<uint8_t>(backend, op);
    }
};
} // namespace MNN
#endif
namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUQuantizedSoftmaxCreator, OpType_QuantizedSoftmax);
}
