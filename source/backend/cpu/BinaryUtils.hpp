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
        return atan(x / y);
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

template<typename Func, typename V, int pack>
void executeVec(void* outputRaw, const void* inputRaw0, const void* inputRaw1, int elementSize, int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / pack;
    const int remainCount = elementSize - sizeDivUnit * pack;
    auto src0 = (const float*)(inputRaw0);
    auto src1 = (const float*)(inputRaw1);
    auto dst = (float*)outputRaw;

    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                V a = V::load(src0);
                V b = V::load(src1);
                V::save(dst, compute(a, b));
                src0 += pack;
                src1 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            float tempSrc0[pack];
            float tempSrc1[pack];
            float tempDst[pack];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(float));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(float));
            V a = V::load(tempSrc0);
            V b = V::load(tempSrc1);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(float));
        }
    } else if (0 == needBroadcastIndex) {
        const float srcValue0 = src0[0];
        V a = V(srcValue0);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                V b = V::load(src1Ptr);
                V::save(dstPtr, compute(a, b));
                src1 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            float tempSrc1[pack];
            float tempDst[pack];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(float));
            V b = V::load(tempSrc1);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(float));
        }
    } else {
        const float srcValue1 = src1[0];
        V b = V(srcValue1);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                V a = V::load(src0Ptr);
                V::save(dstPtr, compute(a, b));
                src0 += pack;
                dst += pack;
            }
        }
        if (remainCount > 0) {
            float tempSrc0[pack];
            float tempDst[pack];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(float));
            V a = V::load(tempSrc0);
            V::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(float));
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

template<typename V, int pack>
MNNBinaryExecute selectVector(int type) {
    switch (type) {
        case BinaryOpOperation_ADD:
            return executeVec<VecBinaryAdd<V>, V, pack>;
        case BinaryOpOperation_SUB:
            return executeVec<VecBinarySub<V>, V, pack>;
        case BinaryOpOperation_MUL:
            return executeVec<VecBinaryMul<V>, V, pack>;
        case BinaryOpOperation_MINIMUM:
            return executeVec<VecBinaryMin<V>, V, pack>;
        case BinaryOpOperation_MAXIMUM:
            return executeVec<VecBinaryMax<V>, V, pack>;
        case BinaryOpOperation_SquaredDifference:
            return executeVec<VecBinarySqd<V>, V, pack>;
    }
    return nullptr;
}
};
