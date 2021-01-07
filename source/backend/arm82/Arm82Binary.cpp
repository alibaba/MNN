//
//  Arm82Binary.cpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "backend/arm82/Arm82Binary.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Macro.h"
#include <algorithm>

#include <arm_neon.h>

namespace MNN {
template<typename Func>
void Arm82Binary(FLOAT16 *dst, const FLOAT16 *src0, const FLOAT16 *src1, const int elementSize, const int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / ARMV82_CHANNEL_UNIT;
    const int remainCount = elementSize - sizeDivUnit * ARMV82_CHANNEL_UNIT;

    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                float16x8_t a = vld1q_f16(src0Ptr);
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, compute(a, b));
                src0 += 8;
                src1 += 8;
                dst += 8;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc0[8];
            FLOAT16 tempSrc1[8];
            FLOAT16 tempDst[8];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(FLOAT16));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(FLOAT16));
            float16x8_t a = vld1q_f16(tempSrc0);
            float16x8_t b = vld1q_f16(tempSrc1);
            vst1q_f16(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    } else if (0 == needBroadcastIndex) {
        const FLOAT16 srcValue0 = src0[0];
        float16x8_t a = vmovq_n_f16(srcValue0);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                float16x8_t b = vld1q_f16(src1Ptr);
                vst1q_f16(dstPtr, compute(a, b));
                src1 += 8;
                dst += 8;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc1[8];
            FLOAT16 tempDst[8];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(FLOAT16));
            float16x8_t b = vld1q_f16(tempSrc1);
            vst1q_f16(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    } else {
        const FLOAT16 srcValue1 = src1[0];
        float16x8_t b = vmovq_n_f16(srcValue1);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                float16x8_t a = vld1q_f16(src0Ptr);
                vst1q_f16(dstPtr, compute(a, b));
                src0 += 8;
                dst += 8;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc0[8];
            FLOAT16 tempDst[8];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(FLOAT16));
            float16x8_t a = vld1q_f16(tempSrc0);
            vst1q_f16(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    }
}


struct BinaryAdd : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vaddq_f16(x, y);
    }
};

struct BinarySub : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vsubq_f16(x, y);
    }
};

struct BinaryMul : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmulq_f16(x, y);
    }
};

struct BinaryMin : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vminq_f16(x, y);
    }
};

struct BinaryMax : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmaxq_f16(x, y);
    }
};

struct BinarySqd : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmulq_f16(vsubq_f16(x, y), vsubq_f16(x, y));
    }
};

Arm82BinaryFloat::Arm82BinaryFloat(Backend *backend, int32_t type):Execution(backend), mType(type) {
    // Do nothing
}

ErrorCode Arm82BinaryFloat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    const int input0DataCount = ARM82TensorElementSizeHelper(inputs[0]);
    const int input1DataCount = ARM82TensorElementSizeHelper(inputs[1]);
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
    return NO_ERROR;
}

ErrorCode Arm82BinaryFloat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    const auto src0 = input0->host<FLOAT16>();
    const auto src1 = input1->host<FLOAT16>();
    auto dst = output->host<FLOAT16>();
    
    switch (mType) {
        case BinaryOpOperation_ADD:
            Arm82Binary<BinaryAdd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SUB:
            Arm82Binary<BinarySub>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MUL:
            Arm82Binary<BinaryMul>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MINIMUM:
            Arm82Binary<BinaryMin>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MAXIMUM:
            Arm82Binary<BinaryMax>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SquaredDifference:
            Arm82Binary<BinarySqd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        default:
            return NOT_SUPPORT;
            break;
    }
    return NO_ERROR;
}

#define SUPPORT(opcode) if(type == opcode) support = true;

class Arm82BinaryCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = inputs[0]->getType();
        if (dataType.code != halide_type_float) {
            return nullptr;
        }
        bool support = false;
        SUPPORT(BinaryOpOperation_ADD);
        SUPPORT(BinaryOpOperation_SUB);
        SUPPORT(BinaryOpOperation_MUL);
        SUPPORT(BinaryOpOperation_MINIMUM);
        SUPPORT(BinaryOpOperation_MAXIMUM);
        SUPPORT(BinaryOpOperation_SquaredDifference);
        if (support) {
            return new Arm82BinaryFloat(backend, type);
        }
        return nullptr;
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_BinaryOp, Arm82BinaryCreator);


} // namespace MNN

#endif
