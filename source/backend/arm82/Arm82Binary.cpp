//
//  Arm82Binary.cpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include <algorithm>
#include "backend/arm82/Arm82Binary.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "core/Macro.h"

#include <arm_neon.h>

namespace MNN {
template<typename Func>
void Arm82BinaryWrap(FLOAT16 *dst, const FLOAT16 *src0, const FLOAT16 *src1, const int elementSize, const int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / 4;
    const int remainCount = elementSize - sizeDivUnit * 4;

    float A[4];
    float B[4];
    float C[4];
    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                vst1q_f32(A, vcvt_f32_f16(vld1_f16(src0Ptr)));
                vst1q_f32(B, vcvt_f32_f16(vld1_f16(src1Ptr)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                vst1_f16(dstPtr, vcvt_f16_f32(vld1q_f32(C)));
                src0 += 4;
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc0[4];
            FLOAT16 tempSrc1[4];
            FLOAT16 tempDst[4];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(FLOAT16));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(FLOAT16));
            vst1q_f32(A, vcvt_f32_f16(vld1_f16(tempSrc0)));
            vst1q_f32(B, vcvt_f32_f16(vld1_f16(tempSrc1)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(A[v], B[v]);
            }
            vst1_f16(tempDst, vcvt_f16_f32(vld1q_f32(C)));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    } else if (0 == needBroadcastIndex) {
        const FLOAT16 srcValue0 = src0[0];
        float16x4_t a = vmov_n_f16(srcValue0);
        vst1q_f32(A, vcvt_f32_f16(a));
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                vst1q_f32(B, vcvt_f32_f16(vld1_f16(src1Ptr)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                vst1_f16(dstPtr, vcvt_f16_f32(vld1q_f32(C)));
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc1[4];
            FLOAT16 tempDst[4];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(FLOAT16));
            vst1q_f32(B, vcvt_f32_f16(vld1_f16(tempSrc1)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(A[v], B[v]);
            }
            vst1_f16(tempDst, vcvt_f16_f32(vld1q_f32(C)));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    } else {
        const FLOAT16 srcValue1 = src1[0];
        float16x4_t b = vmov_n_f16(srcValue1);
        vst1q_f32(B, vcvt_f32_f16(b));
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                vst1q_f32(A, vcvt_f32_f16(vld1_f16(src0Ptr)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                vst1_f16(dstPtr, vcvt_f16_f32(vld1q_f32(C)));
                src0 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            FLOAT16 tempSrc0[4];
            FLOAT16 tempDst[4];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(FLOAT16));
            vst1q_f32(A, vcvt_f32_f16(vld1_f16(tempSrc0)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(A[v], B[v]);
            }
            vst1_f16(tempDst, vcvt_f16_f32(vld1q_f32(C)));
            ::memcpy(dst, tempDst, remainCount * sizeof(FLOAT16));
        }
    }
}


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


struct VecBinaryAdd : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vaddq_f16(x, y);
    }
};

struct VecBinarySub : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vsubq_f16(x, y);
    }
};

struct VecBinaryMul : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmulq_f16(x, y);
    }
};

struct VecBinaryMin : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vminq_f16(x, y);
    }
};

struct VecBinaryMax : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmaxq_f16(x, y);
    }
};

struct VecBinarySqd : std::binary_function<float16x8_t, float16x8_t, float16x8_t> {
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
            Arm82Binary<VecBinaryAdd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SUB:
            Arm82Binary<VecBinarySub>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MUL:
            Arm82Binary<VecBinaryMul>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MINIMUM:
            Arm82Binary<VecBinaryMin>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MAXIMUM:
            Arm82Binary<VecBinaryMax>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SquaredDifference:
            Arm82Binary<VecBinarySqd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_REALDIV:
            Arm82BinaryWrap<BinaryRealDiv<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_FLOORDIV:
            Arm82BinaryWrap<BinaryFloorDiv<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_FLOORMOD:
            Arm82BinaryWrap<BinaryFloorMod<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_POW:
            Arm82BinaryWrap<BinaryPow<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_ATAN2:
            Arm82BinaryWrap<BinaryAtan2<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MOD:
            Arm82BinaryWrap<BinaryMod<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        default:
            return NOT_SUPPORT;
            break;
    }
    return NO_ERROR;
}

class Arm82BinaryCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = outputs[0]->getType();
        if (dataType.code != halide_type_float) {
            return nullptr;
        }
        return new Arm82BinaryFloat(backend, type);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_BinaryOp, Arm82BinaryCreator);


} // namespace MNN
#endif
