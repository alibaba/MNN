//
//  Arm82Unary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include <vector>
#include <cmath>
#include <algorithm>
#include "Arm82Unary.hpp"
#include "Arm82Backend.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/UnaryUtils.hpp"
#include "Arm82OptFunc.hpp"
#include "MNN_generated.h"
#include <arm_neon.h>

extern "C" {
    void MNNGeluFP16(FLOAT16* dst, const FLOAT16* src, size_t size, float* parameters);
}
namespace MNN {

struct VecSquare {
    float16x8_t operator()(float16x8_t &x) const {
        return x * x;
    }
};
struct VecRsqrt {
    float16x8_t operator()(float16x8_t &x) const {
        return vrsqrteq_f16(x);
    }
};

struct VecNeg {
    float16x8_t operator()(float16x8_t &x) const {
        return vnegq_f16(x);
    }
};

struct VecAbs {
    float16x8_t operator()(float16x8_t &x) const {
        return vabsq_f16(x);
    }
};
struct VecRecipocal {
    float16x8_t operator()(float16x8_t &x) const {
        return vrecpeq_f16(x);
    }
};

#if defined(__aarch64__)
struct VecSqrt {
    float16x8_t operator()(float16x8_t &x) const {
        return vsqrtq_f16(x);
    }
};
#endif

template<typename Compute>
void FP16VecUnary(void *dstRaw, const void *src0Raw, int elementSize) {
    Compute Func;
    auto dst = (float16_t*)dstRaw;
    auto src0 = (const float16_t*)src0Raw;
    const int sizeDivUnit = elementSize / 8;
    const int remainCount = elementSize - sizeDivUnit * 8;

    if (sizeDivUnit > 0) {
        for (int i = 0; i < sizeDivUnit; ++i) {
            float16x8_t a = vld1q_f16(src0);
            vst1q_f16(dst, Func(a));
            src0 += 8;
            dst += 8;
        }
    }
    if (remainCount > 0) {
        float16_t tempSrc0[8];
        float16_t tempDst[8];
        ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
        float16x8_t a = vld1q_f16(tempSrc0);
        vst1q_f16(tempDst, Func(a));
        ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
    }
}
#define BLOCK_SIZE 16
template<typename Compute>
static void _Wrap(void* outRaw, const void* inpRaw, int realSize) {
    Compute execute;
    float out[BLOCK_SIZE];
    float inp[BLOCK_SIZE];
    int b = realSize / BLOCK_SIZE;
    int remain = realSize % BLOCK_SIZE;
    auto outR = (int16_t*)outRaw;
    auto inpR = (const int16_t*)inpRaw;
    for (int i=0; i<b; ++i) {
        MNNDequantizeFP16(inpR, inp, BLOCK_SIZE);
        execute(out, inp, BLOCK_SIZE);
        MNNQuantizeFP16(out, outR, BLOCK_SIZE);
        outR += BLOCK_SIZE;
        inpR += BLOCK_SIZE;
    }
    if (remain > 0) {
        MNNDequantizeFP16(inpR, inp, remain);
        execute(out, inp, remain);
        MNNQuantizeFP16(out, outR, remain);
    }
}

struct _Exp {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        float offset[4] = {
            1.0f,
            0.0f,
            0.0f,
            0.0f
        };
        MNNExp(out, inp, offset, realSize);
    }
};
struct _ExpM1 {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        float offset[4] = {
            1.0f,
            -1.0f,
            0.0f,
            0.0f
        };
        MNNExp(out, inp, offset, realSize);
    }
};

struct _Tanh {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        MNNTanh(out, inp, realSize);
    }
};
struct _Sigmoid {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        MNNSigmoidLowp(out, inp, realSize);
    }
};

void FP16GELU(void* outRaw, const void* inpRaw, int realSize) {
    int sizeQuad = realSize / 8;
    int start = 0;
    auto out = (FLOAT16*)outRaw;
    auto inp = (const FLOAT16*)inpRaw;

    if (sizeQuad > 0) {
        constexpr float half_scale = 64.f;
        float parameters[9] = {0.044715f, 0.79788458f, 135135.f/half_scale, 17325.f/half_scale, 378.f/half_scale, 62370.f/half_scale, 3150.f/half_scale, 28.f/half_scale, 1.f/half_scale};
        MNNGeluFP16(out, inp, sizeQuad, parameters);
        start = sizeQuad * 8;
    }
    auto tanhf_poly = [](float value) -> float {
        if (value > 5.0f) {
            return 1.0f;
        } else if (value <= -5.0f) {
            return -1.0f;
        } else {
            float x2 = value * value;
            float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            return a / b;
        }
    };
    for (int i = start; i < realSize; i++) {
        float temp = 0.044715f * inp[i] * inp[i] * inp[i];
        temp = 0.79788458f * (temp + inp[i]);
        out[i] = static_cast<FLOAT16>(1.0f + tanhf_poly(temp)) * inp[i] * 0.5f;
    }
}

void FP16HardSwish(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (FLOAT16*)outRaw;
    auto inp = (const FLOAT16*)inpRaw;
    int sizeC8 = realSize / 8;
    int sizeRemain = realSize % 8;
    if (sizeC8 > 0) {
        float16x8_t zero = vdupq_n_f16(0.f);
        float16x8_t three = vdupq_n_f16(3.f);
        float16x8_t six = vdupq_n_f16(6.f);
        float16x8_t divsix = vdupq_n_f16(1.0f/6.f);
        for (int i = 0; i < sizeC8; i++) {
            auto x = vld1q_f16(inp);
            auto y = vmulq_f16(vmulq_f16(x, vminq_f16(vmaxq_f16(vaddq_f16(x, three), zero), six)), divsix);
            vst1q_f16(out, y);
            out += 8;
            inp += 8;
        }
    }
    for (int i=0; i<sizeRemain; ++i) {
        auto x = inp[i];
        float16_t y;
        if (x <= -3) {
            y = 0;
        } else if (x >= 3) {
            y = x;
        } else {
            y = x * (x + 3) / 6;
        }
        out[i] = y;
    }
}

template <typename Func, typename T>
struct _Unary {
    void operator()(void* outputPtr, const void* inputPtr, int elementSize) const {
        Func f;
        const T *inputData = (T*)inputPtr;
        T *outputData      = (T *)outputPtr;
        for (int i=0; i<elementSize; ++i) {
            outputData[i] = f(inputData[i]);
        }
    }
};

MNNUnaryExecute Arm82Unary::select(int type, int precision) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return FP16VecUnary<VecAbs>;
        case UnaryOpOperation_SQUARE:
            return FP16VecUnary<VecSquare>;
        case UnaryOpOperation_NEG:
            return FP16VecUnary<VecNeg>;
        case UnaryOpOperation_RSQRT:
            return FP16VecUnary<VecRsqrt>;
        case UnaryOpOperation_EXP:
            return _Wrap<_Exp>;
        case UnaryOpOperation_COS:
            return _Wrap<_Unary<UnaryCos<float>, float>>;
        case UnaryOpOperation_SIN:
            return _Wrap<_Unary<UnarySin<float>, float>>;
        case UnaryOpOperation_SIGMOID:
            return _Wrap<_Sigmoid>;
        case UnaryOpOperation_TANH:
            return _Wrap<_Tanh>;
        case UnaryOpOperation_TAN:
            return _Wrap<_Unary<UnaryTan<float>, float>>;
        case UnaryOpOperation_ATAN:
            return _Wrap<_Unary<UnaryATan<float>, float>>;
#if defined(__aarch64__)
        case UnaryOpOperation_SQRT:
            return FP16VecUnary<VecSqrt>;
#else
        case UnaryOpOperation_SQRT:
            return _Wrap<_Unary<UnarySqrt<float>, float>>;
#endif
        case UnaryOpOperation_CEIL:
            return _Wrap<_Unary<UnaryCeil<float>, float>>;
        case UnaryOpOperation_RECIPROCAL:
            return FP16VecUnary<VecRecipocal>;
        case UnaryOpOperation_LOG1P:
            return _Wrap<_Unary<UnaryLog1p<float>, float>>;
        case UnaryOpOperation_LOG:
            return _Wrap<_Unary<UnaryLog<float>, float>>;
        case UnaryOpOperation_FLOOR:
            return _Wrap<_Unary<UnaryFloor<float>, float>>;
        case UnaryOpOperation_BNLL:
            return _Wrap<_Unary<UnaryBNLL<float>, float>>;
        case UnaryOpOperation_ACOSH:
            return _Wrap<_Unary<UnaryAcosh<float>, float>>;
        case UnaryOpOperation_SINH:
            return _Wrap<_Unary<UnarySinh<float>, float>>;
        case UnaryOpOperation_ASINH:
            return _Wrap<_Unary<UnaryAsinh<float>, float>>;
        case UnaryOpOperation_ATANH:
            return _Wrap<_Unary<UnaryAtanh<float>, float>>;
        case UnaryOpOperation_SIGN:
            return _Wrap<_Unary<UnarySign<float>, float>>;
        case UnaryOpOperation_ROUND:
            return _Wrap<_Unary<UnaryRound<float>, float>>;
        case UnaryOpOperation_COSH:
            return _Wrap<_Unary<UnaryCosh<float>, float>>;
        case UnaryOpOperation_ERF:
            return _Wrap<_Unary<UnaryErf<float>, float>>;
        case UnaryOpOperation_ERFC:
            return _Wrap<_Unary<UnaryErfc<float>, float>>;
        case UnaryOpOperation_ERFINV:
            return _Wrap<_Unary<UnaryErfinv<float>, float>>;
        case UnaryOpOperation_EXPM1:
            return _Wrap<_ExpM1>;
        case UnaryOpOperation_ASIN:
            return _Wrap<_Unary<UnaryAsin<float>, float>>;
        case UnaryOpOperation_ACOS:
            return _Wrap<_Unary<UnaryAcos<float>, float>>;
        case UnaryOpOperation_HARDSWISH:
            return FP16HardSwish;
        case UnaryOpOperation_GELU:
        case UnaryOpOperation_GELU_STANDARD:
            return FP16GELU;
        default:
            MNN_ERROR("Don't support %d for arm82 unary\n", type);
            break;
    }
    return nullptr;
}
} // namespace MNN

#endif
