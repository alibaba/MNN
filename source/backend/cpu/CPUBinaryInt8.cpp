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

using Vec16 = MNN::Math::Vec<int8_t, 16>;

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

    mInputQuant0.resize(mTotalSize);
    mInputQuant1.resize(mTotalSize);
    mOutputQuant.resize(mTotalSize);
    std::fill(mInputQuant0.begin(), mInputQuant0.end(), TensorUtils::getDescribe(inputs[0])->quantAttr->scale);
    std::fill(mInputQuant1.begin(), mInputQuant1.end(), TensorUtils::getDescribe(inputs[1])->quantAttr->scale);
    std::fill(mOutputQuant.begin(), mOutputQuant.end(), 1 / TensorUtils::getDescribe(outputs[0])->quantAttr->scale);

    if(mActivationType == 1 && outputs[0]->getType().code == halide_type_float) {
        mActivationExe.reset(new CPURelu(backend(), 0.0));
        mActivationExe->onResize(outputs, outputs);
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
            auto scale0 = mInputQuant0.data() + start;
            auto scale1 = mInputQuant1.data() + start;
            auto scaleDst = mOutputQuant.data() + start;
            if (mNeedBroadcastIndex == 0) {
                inp0 = input0Ptr;
            } else if (mNeedBroadcastIndex == 1) {
                inp1 = input1Ptr;
            }
            auto out = outputPtr + start * outBytes;
#ifdef MNN_USE_NEON
            mProc(out, inp0, inp1, scale0, scale1, scaleDst, realSize / 4, mNeedBroadcastIndex);
#else
             mProc(out, inp0, inp1, scale0, scale1, scaleDst, realSize, mNeedBroadcastIndex);
#endif
        }
    }
    MNN_CONCURRENCY_END();
    
    if(mActivationType == 1 && output->getType().code == halide_type_float) {
        mActivationExe->onExecute(outputs, outputs);;
    }
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
