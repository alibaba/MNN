#include <math.h>
#include <algorithm>
#include "compute/CommonOptFunction.h"
#include "MNN_generated.h"

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMax {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::max(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMin {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::min(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMul {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAdd {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x + y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySub {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryRealDiv {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x / y;
    }
};

/**
 Ref from onnxruntime/onnxruntime/core/providers/cpu/math/element_wise_ops.cc :: Modulus
 */
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryModInt {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        auto res = x % y;
        if ((res < 0 && y > 0) || (res > 0 && y < 0)) {
            res += y;
        }
        return (_ErrorCode)res;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMod {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return fmodf(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreater {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x > y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLess {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x < y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreaterEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x >= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLessEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x <= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x == y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorDiv {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return floor(static_cast<double>(x) / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorMod {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - floor(x / y) * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySquaredDifference {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (x - y) * (x - y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryPow {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return pow(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAtan2 {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return atan2(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLogicalOr {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x || y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLogicalXor {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x ^ y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryNotEqual {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x != y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLeftShift {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)(x << y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryBitwiseAnd {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)(x & y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryRightShift {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)(x >> y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryBitwiseOr {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)(x | y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryBitwiseXor {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)(x ^ y);
    }
};

template<typename Func, typename V, int pack, typename U, typename Tout>
void executeVec(void* outputRaw, const void* inputRaw0, const void* inputRaw1, int elementSize, int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / pack;
    const int remainCount = elementSize - sizeDivUnit * pack;
    auto src0 = (const U*)(inputRaw0);
    auto src1 = (const U*)(inputRaw1);
    auto dst = (Tout*)outputRaw;

    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            int sizeDivC4 = sizeDivUnit / 4;
            int sizeDivUnitRemain = sizeDivUnit % 4;
            for (int i = 0; i < sizeDivC4; ++i) {
                V a0 = V::load(src0);
                V b0 = V::load(src1);
                V a1 = V::load(src0 + 1 * pack);
                V b1 = V::load(src1 + 1 * pack);
                V a2 = V::load(src0 + 2 * pack);
                V b2 = V::load(src1 + 2 * pack);
                V a3 = V::load(src0 + 3 * pack);
                V b3 = V::load(src1 + 3 * pack);
                V::save(dst, compute(a0, b0));
                V::save(dst+1*pack, compute(a1, b1));
                V::save(dst+2*pack, compute(a2, b2));
                V::save(dst+3*pack, compute(a3, b3));
                src0 += 4*pack;
                src1 += 4*pack;
                dst += 4*pack;
            }
            for (int i = 0; i < sizeDivUnitRemain; ++i) {
                V a = V::load(src0);
                V b = V::load(src1);
                V::save(dst, compute(a, b));
                src0 += pack;
                src1 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            U tempSrc0[pack];
            U tempSrc1[pack];
            Tout tempDst[pack];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(U));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(U));
            V a = V::load(tempSrc0);
            V b = V::load(tempSrc1);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(U));
        }
    } else if (0 == needBroadcastIndex) {
        const U srcValue0 = src0[0];
        V a = V(srcValue0);
        if (sizeDivUnit > 0) {
            int sizeDivC4 = sizeDivUnit / 4;
            int sizeUnitRemain = sizeDivUnit % 4;
            for (int i = 0; i < sizeDivC4; ++i) {
                V b0 = V::load(src1);
                V b1 = V::load(src1 + 1*pack);
                V b2 = V::load(src1 + 2*pack);
                V b3 = V::load(src1 + 3*pack);
                V::save(dst, compute(a, b0));
                V::save(dst+1*pack, compute(a, b1));
                V::save(dst+2*pack, compute(a, b2));
                V::save(dst+3*pack, compute(a, b3));
                src1 += 4*pack;
                dst += 4*pack;
            }
            for (int i = 0; i < sizeUnitRemain; ++i) {
                V b = V::load(src1);
                V::save(dst, compute(a, b));
                src1 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            U tempSrc1[pack];
            Tout tempDst[pack];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(U));
            V b = V::load(tempSrc1);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(U));
        }
    } else {
        const auto srcValue1 = static_cast<U>(src1[0]);
        V b = V(srcValue1);
        if (sizeDivUnit > 0) {
            int sizeDivC4 = sizeDivUnit / 4;
            int sizeUnitRemain = sizeDivUnit % 4;
            for (int i = 0; i < sizeDivC4; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                V a0 = V::load(src0Ptr);
                V a1 = V::load(src0Ptr + 1*pack);
                V a2 = V::load(src0Ptr + 2*pack);
                V a3 = V::load(src0Ptr + 3*pack);
                V::save(dstPtr, compute(a0, b));
                V::save(dstPtr+1*pack, compute(a1, b));
                V::save(dstPtr+2*pack, compute(a2, b));
                V::save(dstPtr+3*pack, compute(a3, b));
                src0 += 4*pack;
                dst += 4*pack;
            }
            for (int i = 0; i < sizeUnitRemain; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                V a = V::load(src0Ptr);
                V::save(dstPtr, compute(a, b));
                src0 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            U tempSrc0[pack];
            Tout tempDst[pack];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(U));
            V a = V::load(tempSrc0);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(U));
        }
    }
}

template<typename Vec>
struct VecBinaryAdd  {
    Vec operator()(Vec& x, Vec& y) const {
        return x + y;
    }
};

template<typename Vec>
struct VecBinarySub  {
    Vec operator()(Vec& x, Vec& y) const {
        return x - y;
    }
};

template<typename Vec>
struct VecBinaryMul  {
    Vec operator()(Vec& x, Vec& y) const {
        return x * y;
    }
};

template<typename Vec>
struct VecBinaryMin  {
    Vec operator()(Vec& x, Vec& y) const {
        return Vec::min(x, y);
    }
};

template<typename Vec>
struct VecBinaryMax  {
    Vec operator()(Vec& x, Vec& y) const {
        return Vec::max(x, y);
    }
};

template<typename Vec>
struct VecBinarySqd  {
    Vec operator()(Vec& x, Vec& y) const {
        return (x-y)*(x-y);
    }
};

template<typename Vec>
struct VecBinaryLess  {
    Vec operator()(Vec& x, Vec& y) const {
        return x < y;
    }
};

template<typename Vec>
struct VecBinaryGreater  {
    Vec operator()(Vec& x, Vec& y) const {
        return x > y;
    }
};

template<typename Vec>
struct VecBinaryLessEqual  {
    Vec operator()(Vec& x, Vec& y) const {
        return x <= y;
    }
};

template<typename Vec>
struct VecBinaryGreaterEqual  {
    Vec operator()(Vec& x, Vec& y) const {
        return x >= y;
    }
};

template<typename Vec>
struct VecBinaryEqual  {
    Vec operator()(Vec& x, Vec& y) const {
        return x == y;
    }
};

namespace MNN {
template<typename Tin, typename Tout, typename Func>
void execute(void* outputRaw, const void* inputRaw0, const void* inputRaw1, int elementSize, int broadcastIndex) {
    Func f;
    const int input0DataCount = elementSize;
    const int input1DataCount = elementSize;
    const Tin* input0Data = (const Tin*)inputRaw0;
    const Tin* input1Data = (const Tin*)inputRaw1;
    Tout* outputData = (Tout*)outputRaw;

    if (broadcastIndex == 0) { // data count == 1, not only mean scalar input, maybe of shape (1, 1, 1, ...,1)
        for (int i = 0; i < input1DataCount; i++) {
            outputData[i] = (Tout)(f(input0Data[0], input1Data[i]));
        }
    } else if (broadcastIndex == 1) {
        for (int i = 0; i < input0DataCount; i++) {
            outputData[i] = (Tout)(f(input0Data[i], input1Data[0]));
        }
    } else { // both input contains more than one element，which means no scalar input
        for (int i = 0; i < input0DataCount; i++) {
            outputData[i] = (Tout)(f(input0Data[i], input1Data[i]));
        }
    }
}

template<typename Tin, typename Tout, typename Func>
void executeInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    Func f;
    int size = static_cast<int>(elementSize);
#ifdef MNN_USE_NEON
    size *= 4;
#endif
    float inp0 = 0, inp1 = 0, output = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = (int8_t*)inputRaw0;
    const int8_t* inputData1 = (int8_t*)inputRaw1;
    int8_t* outputData = (int8_t*)outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < size; ++i) {
        if (needBroadcast == 0) {
            inp0 = (inputData0[0]- offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            inp1 = (inputData1[i]- offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            output = f(inp0, inp1);
        } else if (needBroadcast == 1) {
            inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            inp1 = (inputData1[0] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            output = f(inp0, inp1);
        } else {
            inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            inp1 = (inputData1[i] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            output = f(inp0, inp1);
        }
        int value = (int)roundf(output * inputScalesFp32[2]) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

template<typename V, int pack, typename U>
MNNBinaryExecute selectVector(int type) {
    switch (type) {
        case BinaryOpOperation_ADD:
            return executeVec<VecBinaryAdd<V>, V, pack, U, U>;
        case BinaryOpOperation_SUB:
            return executeVec<VecBinarySub<V>, V, pack, U, U>;
        case BinaryOpOperation_MUL:
            return executeVec<VecBinaryMul<V>, V, pack, U, U>;
        case BinaryOpOperation_MINIMUM:
            return executeVec<VecBinaryMin<V>, V, pack, U, U>;
        case BinaryOpOperation_MAXIMUM:
            return executeVec<VecBinaryMax<V>, V, pack, U, U>;
        case BinaryOpOperation_SquaredDifference:
            return executeVec<VecBinarySqd<V>, V, pack, U, U>;
        case BinaryOpOperation_LESS:
            return executeVec<VecBinaryLess<V>, V, pack, U, int32_t>;
        case BinaryOpOperation_LESS_EQUAL:
            return executeVec<VecBinaryLessEqual<V>, V, pack, U, int32_t>;
        case BinaryOpOperation_GREATER:
            return executeVec<VecBinaryGreater<V>, V, pack, U, int32_t>;
        case BinaryOpOperation_GREATER_EQUAL:
            return executeVec<VecBinaryGreaterEqual<V>, V, pack, U, int32_t>;
        case BinaryOpOperation_EQUAL:
            return executeVec<VecBinaryEqual<V>, V, pack, U, int32_t>;
    }
    return nullptr;
}
};
