//
//  Arm82Binary.cpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#if defined(__ANDROID__) || defined(__aarch64__)
#include <algorithm>
#include "backend/arm82/Arm82Binary.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "core/Macro.h"

#include <arm_neon.h>

namespace MNN {
template<typename Func>
void Arm82BinaryWrap(void *dstRaw, const void *src0Raw, const void *src1Raw, const int elementSize, const int needBroadcastIndex) {
    auto dst = (FLOAT16*)dstRaw;
    auto src0 = (const FLOAT16*)src0Raw;
    auto src1 = (const FLOAT16*)src1Raw;
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
void Arm82Binary(void *dstRaw, const void *src0Raw, const void *src1Raw, const int elementSize, const int needBroadcastIndex) {
    auto dst = (FLOAT16*)dstRaw;
    auto src0 = (FLOAT16*)src0Raw;
    auto src1 = (FLOAT16*)src1Raw;
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


struct VecBinaryAdd {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vaddq_f16(x, y);
    }
};

struct VecBinarySub {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vsubq_f16(x, y);
    }
};

struct VecBinaryMul {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmulq_f16(x, y);
    }
};

struct VecBinaryMin {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vminq_f16(x, y);
    }
};

struct VecBinaryMax {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmaxq_f16(x, y);
    }
};

struct VecBinarySqd {
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const {
        return vmulq_f16(vsubq_f16(x, y), vsubq_f16(x, y));
    }
};


MNNBinaryExecute Arm82BinaryFloat::select(int32_t type) {
    switch (type) {
        case BinaryOpOperation_ADD:
            return Arm82Binary<VecBinaryAdd>;
            break;
        case BinaryOpOperation_SUB:
            return Arm82Binary<VecBinarySub>;
            break;
        case BinaryOpOperation_MUL:
            return Arm82Binary<VecBinaryMul>;
            break;
        case BinaryOpOperation_MINIMUM:
            return Arm82Binary<VecBinaryMin>;
            break;
        case BinaryOpOperation_MAXIMUM:
            return Arm82Binary<VecBinaryMax>;
            break;
        case BinaryOpOperation_SquaredDifference:
            return Arm82Binary<VecBinarySqd>;
            break;
        case BinaryOpOperation_REALDIV:
            return Arm82BinaryWrap<BinaryRealDiv<float, float, float>>;
            break;
        case BinaryOpOperation_FLOORDIV:
            return Arm82BinaryWrap<BinaryFloorDiv<float, float, float>>;
            break;
        case BinaryOpOperation_FLOORMOD:
            return Arm82BinaryWrap<BinaryFloorMod<float, float, float>>;
            break;
        case BinaryOpOperation_POW:
            return Arm82BinaryWrap<BinaryPow<float, float, float>>;
            break;
        case BinaryOpOperation_ATAN2:
            return Arm82BinaryWrap<BinaryAtan2<float, float, float>>;
            break;
        case BinaryOpOperation_MOD:
            return Arm82BinaryWrap<BinaryMod<float, float, float>>;
            break;
        default:
            return nullptr;
            break;
    }
    return nullptr;
}

} // namespace MNN
#endif
