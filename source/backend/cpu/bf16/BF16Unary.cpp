#include <vector>
#include <cstddef>
#include "BF16Unary.hpp"
#include "VecHalf.hpp"
#include "math/Vec.hpp"
#include "backend/cpu/UnaryUtils.hpp"
#include "BF16Backend.hpp"

extern "C" {
    void NEON_MNNGelu_BF16(int16_t* dst, const int16_t* src, size_t size, float* parameters);
}
namespace MNN {

using Vec4Half = MNN::Math::VecHalf<4>;
using Vec4 = MNN::Math::Vec<float, 4>;

struct Vec4Square {
    Vec4Half operator()(Vec4Half &x) const {
        return x * x;
    }
};

struct Vec4Neg {
    Vec4Half operator()(Vec4Half &x) const {
        return -x;
    }
};

struct Vec4Abs {
    Vec4Half operator()(Vec4Half &x) const {
        float v[4];
        v[0] = fabs(x[0]);
        v[1] = fabs(x[1]);
        v[2] = fabs(x[2]);
        v[3] = fabs(x[3]);
        auto c = Vec4::load(v);
        Vec4Half value;
        value.value = std::move(c.value);
        return value;
    }
};

template<typename Compute>
void BF16VecUnary(void *dstRaw, const void *src0Raw, int elementSize) {
    Compute Func;
    auto dst = (int16_t*)dstRaw;
    auto src0 = (int16_t*)src0Raw;
    const int sizeDivUnit = elementSize / 4;
    const int remainCount = elementSize - sizeDivUnit * 4;

    if (sizeDivUnit > 0) {
        for (int i = 0; i < sizeDivUnit; ++i) {
            Vec4Half a = Vec4Half::load(src0);
            Vec4Half::save(dst, Func(a));
            src0 += 4;
            dst += 4;
        }
    }
    if (remainCount > 0) {
        int16_t tempSrc0[4];
        int16_t tempDst[4];
        ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
        Vec4Half a = Vec4Half::load(tempSrc0);
        Vec4Half::save(tempDst, Func(a));
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
    auto bf16F = BF16Functions::get();
    auto outR = (int16_t*)outRaw;
    auto inpR = (const int16_t*)inpRaw;
    for (int i=0; i<b; ++i) {
        bf16F->MNNLowpToFp32(inpR, inp, BLOCK_SIZE);
        execute(out, inp, BLOCK_SIZE);
        bf16F->MNNFp32ToLowp(out, outR, BLOCK_SIZE);
        outR += BLOCK_SIZE;
        inpR += BLOCK_SIZE;
    }
    if (remain > 0) {
        bf16F->MNNLowpToFp32(inpR, inp, remain);
        execute(out, inp, remain);
        bf16F->MNNFp32ToLowp(out, outR, remain);
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

struct _HardSwish {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        MNNHardSwishCommon(out, inp, realSize);
    }
};

struct _Gelu {
    void operator()(void* outRaw, const void* inpRaw, int realSize) const {
        auto out = (float*)outRaw;
        auto inp = (const float*)inpRaw;
        MNNGeluCommon(out, inp, realSize);
    }
};
void BF16GELU (void* OutRaw, const void* inpRaw, int realSize) {
    int16_t* out = (int16_t*)OutRaw;
    const int16_t* inp = (const int16_t*)inpRaw;
    int sizeQuad = realSize / 8;
    int start = 0;
    float parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    if (sizeQuad > 0) {
#ifdef MNN_USE_NEON
        NEON_MNNGelu_BF16(out, inp, sizeQuad, parameters);
#endif
        start = sizeQuad * 8;
    }
    int16_t tempInp[8];
    for (int i = start; i < realSize; i++) {
        tempInp[i-start] = inp[i];
    }
#ifdef MNN_USE_NEON
    NEON_MNNGelu_BF16(tempInp, tempInp, 1, parameters);
#endif
    for (int i = start; i < realSize; i++) {
        out[i] = tempInp[i-start];
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

MNNUnaryExecute BF16UnaryFloatSelect(int type, int precision) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return BF16VecUnary<Vec4Abs>;
        case UnaryOpOperation_SQUARE:
            return BF16VecUnary<Vec4Square>;
        case UnaryOpOperation_NEG:
            return BF16VecUnary<Vec4Neg>;
        case UnaryOpOperation_RSQRT:
            return _Wrap<_Unary<UnaryRsqrt<float>, float>>;
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
        case UnaryOpOperation_SQRT:
            return _Wrap<_Unary<UnarySqrt<float>, float>>;
        case UnaryOpOperation_CEIL:
            return _Wrap<_Unary<UnaryCeil<float>, float>>;
        case UnaryOpOperation_RECIPROCAL:
            return _Wrap<_Unary<UnaryRecipocal<float>, float>>;
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
            return _Wrap<_HardSwish>;
        case UnaryOpOperation_GELU:
#ifdef MNN_USE_NEON
            return BF16GELU;
#else
            return _Wrap<_Gelu>;
#endif
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

};
