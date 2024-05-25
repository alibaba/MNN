//
//  CPUBinaryInt8.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBinaryInt8.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "BinaryUtils.hpp"
#include "math/Vec.hpp"

namespace MNN {

ErrorCode CPUBinaryInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0DataCount = TensorUtils::getRawSize(inputs[0]);
    auto input1DataCount = TensorUtils::getRawSize(inputs[1]);
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
    } else {
        mNeedBroadcastIndex = 1;
    }
    mTotalSize = ((CPUBackend*)backend())->getTensorSize(outputs[0]);

    auto core = static_cast<CPUBackend*>(backend())->functions();

    mQuantScalesInt32.resize(2); // When use int32 scales computing, output scale is needless.
    mQuantScalesFp32.resize(3);
    mQuantScalesInt32[0] = TensorUtils::getDescribe(inputs[0])->quantAttr->scale * (1 << 16);
    mQuantScalesInt32[1] = TensorUtils::getDescribe(inputs[1])->quantAttr->scale * (1 << 16);
    mQuantScalesFp32[0] =  TensorUtils::getDescribe(inputs[0])->quantAttr->scale;
    mQuantScalesFp32[1] =  TensorUtils::getDescribe(inputs[1])->quantAttr->scale;
    if (TensorUtils::getDescribe(outputs[0])->quantAttr->scale != 0) {
        mQuantScalesFp32[2] = 1 / TensorUtils::getDescribe(outputs[0])->quantAttr->scale;
    } else {
        mQuantScalesFp32[2] = 0;
    }

    float inputScale0 = TensorUtils::getDescribe(inputs[0])->quantAttr->scale;
    float inputScale1 = TensorUtils::getDescribe(inputs[1])->quantAttr->scale;
    float outputScale = TensorUtils::getDescribe(outputs[0])->quantAttr->scale;
    ssize_t inputZero0 = (ssize_t)TensorUtils::getDescribe(inputs[0])->quantAttr->zero;
    ssize_t inputZero1 = (ssize_t)TensorUtils::getDescribe(inputs[1])->quantAttr->zero;
    ssize_t outputZero = (ssize_t)TensorUtils::getDescribe(outputs[0])->quantAttr->zero;
    mInputZeros.resize(2);
    mOutputZeros.resize(1);
    mInputScales.resize(2);
    mOutputScales.resize(1);
    mInputZeros = {inputZero0, inputZero1};
    mOutputZeros = {outputZero};
    mInputScales = {inputScale0, inputScale1};
    mOutputScales = {outputScale};

    mMinValue = static_cast<int>(TensorUtils::getDescribe(outputs[0])->quantAttr->min);
    if(mActivationType == 1 && outputs[0]->getType().code == halide_type_float) {
        mMinValue = 0;
    }

    return NO_ERROR;
}

ErrorCode CPUBinaryInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(mTotalSize);

    auto input0Ptr = input->host<int8_t>();
    auto input1Ptr = input1->host<int8_t>();
    auto outputPtr = outputs[0]->host<int8_t>();

    int inpBytes = 1;
    int outBytes = 1;

    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        QuanPrePostParameters params;
        
        params.inputScale = mInputScales.data();
        params.outputScale = mOutputScales.data();
        params.outputZeroPoint = mOutputZeros.data();
        params.inputZeroPoint = mInputZeros.data();
        params.minValue = (ssize_t)mMinValue;
        params.maxValue = (ssize_t)TensorUtils::getDescribe(outputs[0])->quantAttr->max;

        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = mTotalSize - start;
        }
        if (realSize > 0) {
            auto inp0 = input0Ptr + start * inpBytes;
            auto inp1 = input1Ptr + start * inpBytes;
            if (mNeedBroadcastIndex == 0) {
                inp0 = input0Ptr;
            } else if (mNeedBroadcastIndex == 1) {
                inp1 = input1Ptr;
            }
            auto out = outputPtr + start * outBytes;
#ifdef MNN_USE_NEON
            mProc(out, inp0, inp1, mQuantScalesInt32.data(), mQuantScalesFp32.data(), &params, realSize / 4, mNeedBroadcastIndex);         
#else
            mProc(out, inp0, inp1, mQuantScalesInt32.data(), mQuantScalesFp32.data(), &params, realSize, mNeedBroadcastIndex);
#endif
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

MNNBinaryExecInt8 CPUBinaryInt8::selectForInt8(int type) {
    switch (type) {
        case BinaryOpOperation_ADD:
            return MNNBinaryAddInt8;
        case BinaryOpOperation_SUB:
            return MNNBinarySubInt8;
        case BinaryOpOperation_MUL:
            return MNNBinaryMulInt8;
        case BinaryOpOperation_MINIMUM:
            return MNNBinaryMinInt8;
        case BinaryOpOperation_MAXIMUM:
            return MNNBinaryMaxInt8;
        case BinaryOpOperation_SquaredDifference:
            return MNNBinarySqdInt8;
        case BinaryOpOperation_REALDIV:
            return executeInt8<int8_t, int8_t, BinaryRealDiv<float, float, float>>;
        case BinaryOpOperation_FLOORDIV:
            return executeInt8<int8_t, int8_t, BinaryFloorDiv<float, float, float>>;
        case BinaryOpOperation_FLOORMOD:
            return executeInt8<int8_t, int8_t, BinaryFloorMod<float, float, float>>;
        case BinaryOpOperation_POW:
            return executeInt8<int8_t, int8_t, BinaryPow<float, float, float>>;
        case BinaryOpOperation_ATAN2:
            return executeInt8<int8_t, int8_t, BinaryAtan2<float, float, float>>;
        case BinaryOpOperation_MOD:
            return executeInt8<int8_t, int8_t, BinaryMod<float, float, float>>;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

} // namespace MNN
