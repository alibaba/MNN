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
    const int input0DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[0]);
    const int input1DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[1]);
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
        mTotalSize = input1DataCount;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
        mTotalSize = input1DataCount;
    } else {
        mNeedBroadcastIndex = 1;
        mTotalSize = input0DataCount;
    }
    MNN_ASSERT(mTotalSize == ((CPUBackend*)backend())->getTensorSize(outputs[0]));

    auto core = static_cast<CPUBackend*>(backend())->functions();

    mInputOffset0.resize(1);
    mInputOffset1.resize(1);
    mOutputOffset.resize(1);
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
    mInputOffset0[0] = (int8_t)TensorUtils::getDescribe(inputs[0])->quantAttr->zero;
    mInputOffset1[0] = (int8_t)TensorUtils::getDescribe(inputs[1])->quantAttr->zero;
    mOutputOffset[0] = (int8_t)TensorUtils::getDescribe(outputs[0])->quantAttr->zero;

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
    auto precision = static_cast<CPUBackend*>(backend())->precisionMode();
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = mTotalSize - start;
        }
        if (realSize > 0) {
            auto inp0 = input0Ptr + start * inpBytes;
            auto inp1 = input1Ptr + start * inpBytes;
            auto offset0 = mInputOffset0.data();
            auto offset1 = mInputOffset1.data();
            auto offsetDst = mOutputOffset.data();
            if (mNeedBroadcastIndex == 0) {
                inp0 = input0Ptr;
            } else if (mNeedBroadcastIndex == 1) {
                inp1 = input1Ptr;
            }
            auto out = outputPtr + start * outBytes;
#ifdef MNN_USE_NEON
            mProc(out, inp0, inp1, mQuantScalesInt32.data(), mQuantScalesFp32.data(), offset0, offset1, offsetDst, realSize / 4, mNeedBroadcastIndex);
#else
            mProc(out, inp0, inp1, mQuantScalesInt32.data(), mQuantScalesFp32.data(), offset0, offset1, offsetDst, realSize, mNeedBroadcastIndex);
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
