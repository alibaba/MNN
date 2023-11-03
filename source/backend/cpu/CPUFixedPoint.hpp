/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef CPUFixedPoint_HPP
#define CPUFixedPoint_HPP

#include <math.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
// Part 1: Low-level integer-arithmetic primitives.
template <typename tIntegerType>
struct FixedPointRawTypeTraits {};

template <>
struct FixedPointRawTypeTraits<std::int32_t> {
    typedef std::int32_t ScalarRawType;
    static constexpr int kLanes = 1;
};

template <>
struct FixedPointRawTypeTraits<std::int16_t> {
    typedef std::int16_t ScalarRawType;
    static constexpr int kLanes = 1;
};

// Returns a SIMD value duplicating a scalar value across all lanes.
template <typename tRawType>
tRawType Dup(typename FixedPointRawTypeTraits<tRawType>::ScalarRawType x) {
    return x;
}

// Plain bit-wise AND
template <typename tIntegerType>
tIntegerType BitAnd(tIntegerType a, tIntegerType b) {
    return a & b;
}

// Plain bit-wise OR
template <typename tIntegerType>
tIntegerType BitOr(tIntegerType a, tIntegerType b) {
    return a | b;
}

// Plain bit-wise XOR
template <typename tIntegerType>
tIntegerType BitXor(tIntegerType a, tIntegerType b) {
    return a ^ b;
}

// Plain bit-wise NOT
template <typename tIntegerType>
tIntegerType BitNot(tIntegerType a) {
    return ~a;
}

// Integer addition. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Add(tIntegerType a, tIntegerType b) {
    return a + b;
}

// Integer subtraction. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Mul(tIntegerType a, tIntegerType b) {
    return a * b;
}

template <typename tIntegerType>
tIntegerType Sub(tIntegerType a, tIntegerType b) {
    return a - b;
}

// Integer unary negative. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Neg(tIntegerType a) {
    return -a;
}

// Integer arithmetic left-shift, equivalent to multiplying with a power of two.
// Not saturating. Negative inputs do not necessarily invoke undefined
// behaviour. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType ShiftLeft(tIntegerType a, int offset) {
    return a * (static_cast<tIntegerType>(1) << offset);
}

// Integer arithmetic right-shift. Not rounding.
// Relying on implementation-defined, but in-practice-consistent,
// C++ compiler behavior.
template <typename tIntegerType>
tIntegerType ShiftRight(tIntegerType a, int offset) {
    return a >> offset;
}

// Each bit of the result is set to the corresponding bit of either then_val or
// else_val depending on whether the corresponding bit of if_mask is set.
// Equivalent to the VBSL instruction in ARM NEON.
template <typename tIntegerType>
tIntegerType SelectUsingMask(tIntegerType if_mask, tIntegerType then_val, tIntegerType else_val) {
    return BitXor(BitAnd(if_mask, then_val), BitAnd(BitNot(if_mask), else_val));
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is non-zero.
template <typename tIntegerType>
tIntegerType MaskIfNonZero(tIntegerType a) {
    static constexpr tIntegerType zero = 0;
    return a ? BitNot(zero) : zero;
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is zero.
template <typename tIntegerType>
tIntegerType MaskIfZero(tIntegerType a) {
    return MaskIfNonZero<tIntegerType>(!a);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are equal.
template <typename tIntegerType>
tIntegerType MaskIfEqual(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a == b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are not equal.
template <typename tIntegerType>
tIntegerType MaskIfNotEqual(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a != b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a > b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThan(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a > b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a >= b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThanOrEqual(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a >= b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a < b.
template <typename tIntegerType>
tIntegerType MaskIfLessThan(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a < b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a <= b.
template <typename tIntegerType>
tIntegerType MaskIfLessThanOrEqual(tIntegerType a, tIntegerType b) {
    return MaskIfNonZero<tIntegerType>(a <= b);
}

// Returns true if all of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool All(tIntegerType a) {
    return a;
}

// Returns true if any of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool Any(tIntegerType a) {
    return a;
}

// Returns (a+b)/2, rounded to the nearest integer.
// Equivalent to VRHADD in the ARM NEON instruction set.
template <typename IntegerType>
IntegerType RoundingHalfSum(IntegerType a, IntegerType b) {
    static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
    return a;
}

template <>
inline std::int32_t RoundingHalfSum(std::int32_t a, std::int32_t b) {
    std::int64_t a64  = a;
    std::int64_t b64  = b;
    std::int64_t sum  = a64 + b64;
    std::int64_t sign = sum >= 0 ? 1 : -1;
    return static_cast<std::int32_t>((sum + sign) / 2);
}

template <>
inline std::int16_t RoundingHalfSum(std::int16_t a, std::int16_t b) {
    std::int32_t a32  = a;
    std::int32_t b32  = b;
    std::int32_t sum  = a32 + b32;
    std::int32_t sign = sum >= 0 ? 1 : -1;
    return static_cast<std::int16_t>((sum + sign) / 2);
}

template <typename IntegerType>
IntegerType SaturatingAdd(IntegerType a, IntegerType b) {
    static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
    return a;
}

// So far this is only needed for int16.
template <>
inline std::int16_t SaturatingAdd(std::int16_t a, std::int16_t b) {
    std::int32_t a32 = a;
    std::int32_t b32 = b;
    std::int32_t sum = a32 + b32;
    return static_cast<std::int16_t>(std::min(32767, std::max(-32768, sum)));
}

// Returns a+b, saturating if the integers are 16bit or narrower,
// otherwise just a plain addition.
template <typename IntegerType, bool Is16Bit>
struct AddSaturatingIf16BitImpl {
    static IntegerType Run(IntegerType a, IntegerType b) {
        return Add(a, b);
    }
};
template <typename IntegerType>
struct AddSaturatingIf16BitImpl<IntegerType, true> {
    static IntegerType Run(IntegerType a, IntegerType b) {
        return SaturatingAdd(a, b);
    }
};
template <typename IntegerType>
IntegerType AddSaturatingIf16Bit(IntegerType a, IntegerType b) {
    using ScalarType = typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
    return AddSaturatingIf16BitImpl<IntegerType, sizeof(ScalarType) == 2>::Run(a, b);
}
template <typename IntegerType>
IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
    static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
    return a;
}

// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
template <>
inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
    std::int64_t a_64(a);
    std::int64_t b_64(b);
    std::int64_t ab_64        = a_64 * b_64;
    std::int32_t nudge        = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    std::int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

template <>
inline std::int16_t SaturatingRoundingDoublingHighMul(std::int16_t a, std::int16_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
    std::int32_t a_32(a);
    std::int32_t b_32(b);
    std::int32_t ab_32        = a_32 * b_32;
    std::int16_t nudge        = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
    std::int16_t ab_x2_high16 = static_cast<std::int16_t>((ab_32 + nudge) / (1 << 15));
    return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
template <typename IntegerType>
inline IntegerType RoundingDivideByPOT(IntegerType x, int exponent) {
    assert(exponent >= 0);
    assert(exponent <= 31);
    const IntegerType mask      = Dup<IntegerType>(static_cast<IntegerType>((1ll << exponent) - 1));
    const IntegerType zero      = Dup<IntegerType>(0);
    const IntegerType one       = Dup<IntegerType>(1);
    const IntegerType remainder = BitAnd(x, mask);
    const IntegerType threshold = Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
    return Add(ShiftRight(x, exponent), BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}

// Returns the product of a run-time integer value by a compile-time power
// of two, with either a positive exponent (equivalent to an arithmetic
// left shift, saturating) or a negative exponent (equivalent to an arithmetic
// right shift, rounding to nearest).
template <int Exponent, typename IntegerType, int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
struct ImplSaturatingRoundingMultiplyByPOT {};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 0> {
    static IntegerType eval(IntegerType x) {
        return x;
    }
};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 1> {
    static IntegerType eval(IntegerType x) {
        using ScalarIntegerType         = typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
        const IntegerType min           = Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::min());
        const IntegerType max           = Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::max());
        const int ScalarIntegerTypeBits = 8 * sizeof(ScalarIntegerType);

        const std::int32_t threshold    = ((1 << (ScalarIntegerTypeBits - 1 - Exponent)) - 1);
        const IntegerType positive_mask = MaskIfGreaterThan(x, Dup<IntegerType>(threshold));
        const IntegerType negative_mask = MaskIfLessThan(x, Dup<IntegerType>(-threshold));

        IntegerType result = ShiftLeft(x, Exponent);
        result             = SelectUsingMask(positive_mask, max, result);
        result             = SelectUsingMask(negative_mask, min, result);
        return result;
    }
};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, -1> {
    static IntegerType eval(IntegerType x) {
        return RoundingDivideByPOT<IntegerType>(x, -Exponent);
    }
};

template <int Exponent, typename IntegerType>
IntegerType SaturatingRoundingMultiplyByPOT(IntegerType x) {
    return ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType>::eval(x);
}

// Part 2: the FixedPoint class.
template <typename tRawType, int tIntegerBits>
class FixedPoint {
public:
    typedef tRawType RawType;

    typedef FixedPointRawTypeTraits<RawType> RawTypeTraits;
    typedef typename RawTypeTraits::ScalarRawType ScalarRawType;

    static constexpr int kTotalBits      = 8 * sizeof(ScalarRawType);
    static constexpr int kIntegerBits    = tIntegerBits;
    static constexpr int kFractionalBits = kTotalBits - 1 - kIntegerBits;
    static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits, "bad IntegerBits");

    typedef FixedPoint<ScalarRawType, kIntegerBits> ScalarFixedPointType;

    static const ScalarRawType ScalarRawMin() {
        return std::numeric_limits<ScalarRawType>::min();
    }

    static const ScalarRawType ScalarRawMax() {
        return std::numeric_limits<ScalarRawType>::max();
    }

    static const ScalarRawType RawMin() {
        return VectorFromScalar(ScalarRawMin());
    }

    static const ScalarRawType RawMax() {
        return VectorFromScalar(ScalarRawMax());
    }

    static FixedPoint FromRaw(RawType x) {
        FixedPoint retval;
        retval.raw() = x;
        return retval;
    }

    static FixedPoint FromScalarRaw(ScalarRawType x) {
        FixedPoint retval;
        retval.raw() = Dup<RawType>(x);
        return retval;
    }

    static FixedPoint FromScalarFixedPoint(ScalarFixedPointType x) {
        return FromScalarRaw(x.raw());
    }

    template <int Exponent>
    static FixedPoint ConstantPOT() {
        static constexpr int kOffset = kFractionalBits + Exponent;
        static_assert(kOffset < 31, "Constant not exactly representable in this fixed-point format");
        return FromScalarRaw(ScalarRawType(1) << kOffset);
    }

    static FixedPoint Zero() {
        return FromScalarRaw(0);
    }

    static FixedPoint One() {
        return FromScalarRaw(kIntegerBits == 0 ? ScalarRawMax()
                                               : (ScalarRawType(1) << (kIntegerBits == 0 ? 0 : kFractionalBits)));
    }

    static FixedPoint FromDouble(double x) {
        const double min_bound = static_cast<double>(ScalarRawMin());
        const double max_bound = static_cast<double>(ScalarRawMax());
        return FromScalarRaw(static_cast<ScalarRawType>(
            std::min(std::max(round(x * static_cast<double>(1ll << kFractionalBits)), min_bound), max_bound)));
    }

    RawType raw() const {
        return i_;
    }
    RawType& raw() {
        return i_;
    }

private:
    RawType i_;
};

// Part 3: implementation of arithmetic operators for the
// FixedPoint class, and a few related functions.

// A FixedPoint multiplication is just a
// SaturatingRoundingDoublingHighMul operation on the underlying
// raw integer values. The IntegerBits simply add up, as is obvious
// from the fact that the range is [-2^IntegerBits, 2^IntegerBits).
template <typename tRawType, int tIntegerBits_a, int tIntegerBits_b>
FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> operator*(FixedPoint<tRawType, tIntegerBits_a> a,
                                                                FixedPoint<tRawType, tIntegerBits_b> b) {
    FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> c;
    c.raw() = SaturatingRoundingDoublingHighMul(a.raw(), b.raw());
    return c;
}

// Tweaking IntegerBits gives exact multiplication by a power of two.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tExponent + tIntegerBits> ExactMulByPot(FixedPoint<tRawType, tIntegerBits> a) {
    FixedPoint<tRawType, tExponent + tIntegerBits> c;
    c.raw() = a.raw();
    return c;
}

// If we want to leave IntegerBits fixed, then multiplication
// by a power of two has to be saturating/rounding, not exact anymore.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingRoundingMultiplyByPOT(FixedPoint<tRawType, tIntegerBits> a) {
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(SaturatingRoundingMultiplyByPOT<tExponent>(a.raw()));
}

// Generic arithmetic operators.

#define MAKE_FIXEDPOINT_UNARY_FUNC(FuncName, ImplFuncName)                              \
    template <typename tRawType, int tIntegerBits>                                      \
    FixedPoint<tRawType, tIntegerBits> FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
        return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw()));      \
    }

#define MAKE_FIXEDPOINT_BINARY_FUNC(FuncName, ImplFuncName)                                 \
    template <typename tRawType, int tIntegerBits>                                          \
    FixedPoint<tRawType, tIntegerBits> FuncName(FixedPoint<tRawType, tIntegerBits> a,       \
                                                FixedPoint<tRawType, tIntegerBits> b) {     \
        return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw(), b.raw())); \
    }

MAKE_FIXEDPOINT_UNARY_FUNC(operator-, Neg)
MAKE_FIXEDPOINT_UNARY_FUNC(operator~, BitNot)
MAKE_FIXEDPOINT_BINARY_FUNC(operator+, Add)
MAKE_FIXEDPOINT_BINARY_FUNC(operator-, Sub)
MAKE_FIXEDPOINT_BINARY_FUNC(operator&, BitAnd)
MAKE_FIXEDPOINT_BINARY_FUNC(operator^, BitXor)
MAKE_FIXEDPOINT_BINARY_FUNC(operator|, BitOr)
MAKE_FIXEDPOINT_BINARY_FUNC(RoundingHalfSum, RoundingHalfSum)

#undef MAKE_FIXEDPOINT_UNARY_FUNC
#undef MAKE_FIXEDPOINT_BINARY_FUNC

#define MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(FuncName)    \
    template <typename tRawType, int tIntegerBits>            \
    tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
        return FuncName(a.raw());                             \
    }

#define MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(FuncName)                                         \
    template <typename tRawType, int tIntegerBits>                                                  \
    tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) { \
        return FuncName(a.raw(), b.raw());                                                          \
    }

MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfZero)
MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfNonZero)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfNotEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThanOrEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThanOrEqual)

#undef MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW
#undef MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SelectUsingMask(tRawType if_mask, FixedPoint<tRawType, tIntegerBits> then_val,
                                                   FixedPoint<tRawType, tIntegerBits> else_val) {
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(SelectUsingMask(if_mask, then_val.raw(), else_val.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator==(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) {
    return All(MaskIfEqual(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator!=(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) {
    return !(a == b);
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingAdd(FixedPoint<tRawType, tIntegerBits> a,
                                                 FixedPoint<tRawType, tIntegerBits> b) {
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(SaturatingAdd(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> AddSaturatingIf16Bit(FixedPoint<tRawType, tIntegerBits> a,
                                                        FixedPoint<tRawType, tIntegerBits> b) {
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(AddSaturatingIf16Bit(a.raw(), b.raw()));
}

// Conversion to floating-point.
template <typename tRawType, int tIntegerBits>
double ToDouble(FixedPoint<tRawType, tIntegerBits> x) {
    static_assert(FixedPointRawTypeTraits<tRawType>::kLanes == 1, "not applicable to SIMD types");
    typedef FixedPoint<tRawType, tIntegerBits> F;
    return x.raw() / static_cast<double>(1ll << F::kFractionalBits);
}

// Rescale changes the number of IntegerBits and updates the underlying
// raw integer value accordingly.
template <int tIntegerBitsDst, typename tRawType, int tIntegerBitsSrc>
FixedPoint<tRawType, tIntegerBitsDst> Rescale(FixedPoint<tRawType, tIntegerBitsSrc> x) {
    static constexpr int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
    FixedPoint<tRawType, tIntegerBitsDst> result;
    result.raw() = SaturatingRoundingMultiplyByPOT<kExponent>(x.raw());
    return result;
}

// CheckedFixedPointConstant allows to specify fixed-point constants
// initialized as real numbers, in a way that does not compile floating-point
// arithmetic in production code, yet still checks agreement with the
// floating-point expressions when asserts are enabled.
//
// The raw integer value provided is always a int32, encoding a 32-bit
// fixed-point value, regardless of the actual Scalar type. This allows
// writing generic code that applies just as well to the 32-bit and 16-bit
// cases. In the 16-bit case, the raw integer value is internally
// rounding-shifted by 16 bits to the right.
template <typename FixedPointType>
inline typename FixedPointType::ScalarRawType RescaleConstantInitializer(std::int32_t int32_value) {
    typedef typename FixedPointType::ScalarRawType ScalarRawType;
    static constexpr int ScalarTypeBits = 8 * sizeof(ScalarRawType);
    return static_cast<ScalarRawType>(RoundingDivideByPOT<std::int32_t>(int32_value, 32 - ScalarTypeBits));
}
#ifdef GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS
template <typename FixedPointType>
FixedPointType CheckedFixedPointConstant(std::int32_t raw_value, double double_value) {
    const FixedPointType result = FixedPointType::FromScalarRaw(raw_value);
    assert(result == FixedPointType::FromDouble(double_value));
    return result;
}
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawInt32Value, DoubleValue)                  \
    (CheckedFixedPointConstant<FixedPointType>(RescaleConstantInitializer<FixedPointType>(ScalarRawInt32Value), \
                                               DoubleValue))

#else
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawInt32Value, DoubleValue) \
    (FixedPointType::FromScalarRaw(RescaleConstantInitializer<FixedPointType>(ScalarRawInt32Value)))
#endif

// Implementation of exponential function.

// Returns exp(x) for x in [-1/4, 0).
template <typename tRawType>
FixedPoint<tRawType, 0> exp_on_interval_between_negative_one_quarter_and_0_excl(FixedPoint<tRawType, 0> a) {
    typedef FixedPoint<tRawType, 0> F;
    const F constant_term     = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 1895147668, std::exp(-1.0 / 8.0));
    const F constant_1_over_3 = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 715827883, 1.0 / 3.0);
    // We're evaluating a Taylor expansion around -1/8, so we do the change of
    // variable: x = a + 1/8.
    // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
    F x         = a + F::template ConstantPOT<-3>();
    F x2        = x * x;
    F x3        = x2 * x;
    F x4        = x2 * x2;
    F x4_over_4 = SaturatingRoundingMultiplyByPOT<-2>(x4);
    F x4_over_24_plus_x3_over_6_plus_x2_over_2 =
        SaturatingRoundingMultiplyByPOT<-1>(((x4_over_4 + x3) * constant_1_over_3) + x2);
    return AddSaturatingIf16Bit(constant_term, constant_term * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2));
}

// Returns exp(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> exp_on_negative_values(FixedPoint<tRawType, tIntegerBits> a) {
    typedef FixedPoint<tRawType, tIntegerBits> InputF;
    typedef FixedPoint<tRawType, 0> ResultF;
    static constexpr int kFractionalBits   = InputF::kFractionalBits;
    static constexpr int kIntegerBits      = InputF::kIntegerBits;
    const InputF kOneQuarter               = InputF::template ConstantPOT<-2>();
    InputF mask                            = kOneQuarter - InputF::FromScalarRaw(1);
    InputF a_mod_quarter_minus_one_quarter = (a & mask) - kOneQuarter;
    ResultF result =
        exp_on_interval_between_negative_one_quarter_and_0_excl(Rescale<0>(a_mod_quarter_minus_one_quarter));
    tRawType remainder = (a_mod_quarter_minus_one_quarter - a).raw();

#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)                                                  \
    if (kIntegerBits > Exponent) {                                                                                   \
        const ResultF kMultiplier =                                                                                  \
            GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(ResultF, FixedPointMultiplier, std::exp(-std::pow(2.0, Exponent))); \
        static constexpr int kShiftAmount = kIntegerBits > Exponent ? kFractionalBits + Exponent : 0;                \
        result = SelectUsingMask(MaskIfNonZero(BitAnd(remainder, Dup<tRawType>(1 << kShiftAmount))),                 \
                                 result * kMultiplier, result);                                                      \
    }

    GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);
    GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);
    GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);
    GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);
    GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);
    GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);
    GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);

#undef GEMMLOWP_EXP_BARREL_SHIFTER

    static constexpr int clampB = kIntegerBits > 5 ? 36 - kIntegerBits : 0;
    if (kIntegerBits > 5) {
        const InputF clamp = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(InputF, -(1 << clampB), -32.0);
        result             = SelectUsingMask(MaskIfLessThan(a, clamp), ResultF::Zero(), result);
    }

    result = SelectUsingMask(MaskIfZero(a), ResultF::One(), result);
    return result;
}

// Implementation of tanh: (1 - exp(-2x)) / (1 + exp(-2x)).

// Returns (1 - x) / (1 + x) for x in (0, 1).
template <typename tRawType>
FixedPoint<tRawType, 0> one_minus_x_over_one_plus_x_for_x_in_0_1(FixedPoint<tRawType, 0> a) {
    typedef FixedPoint<tRawType, 0> F0;
    typedef FixedPoint<tRawType, 2> F2;
    F0 half_denominator = RoundingHalfSum(a, F0::One());
    // Newton-Raphson division
    // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
    // Refer to that page for the logic behind the 48/17 and 32/17 constants.
    const F2 constant_48_over_17     = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
    const F2 constant_neg_32_over_17 = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
    F2 x                             = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
    for (int i = 0; i < 3; i++) {
        F2 half_denominator_times_x           = half_denominator * x;
        F2 one_minus_half_denominator_times_x = F2::One() - half_denominator_times_x;
        x                                     = x + Rescale<2>(x * one_minus_half_denominator_times_x);
    }
    return Rescale<0>(x - F2::One());
}

// Returns -tanh(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> neg_tanh_on_negative_values(FixedPoint<tRawType, tIntegerBits> a) {
    return one_minus_x_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(ExactMulByPot<1>(a)));
}

// Returns tanh(x) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> tanh(FixedPoint<tRawType, tIntegerBits> a) {
    typedef FixedPoint<tRawType, tIntegerBits> InputF;
    typedef FixedPoint<tRawType, 0> ResultF;
    tRawType mask_if_negative = MaskIfLessThan(a, InputF::Zero());
    tRawType mask_if_zero     = MaskIfZero(a);
    InputF n                  = SelectUsingMask(mask_if_negative, a, -a);
    ResultF t                 = neg_tanh_on_negative_values(n);
    return SelectUsingMask(mask_if_zero, ResultF::Zero(), SelectUsingMask(mask_if_negative, -t, t));
}

// Implementation of logistic function.

// Returns 1 / (1 + x) for x in (0, 1).
template <typename tRawType>
FixedPoint<tRawType, 0> one_over_one_plus_x_for_x_in_0_1(FixedPoint<tRawType, 0> a) {
    typedef FixedPoint<tRawType, 0> F0;
    typedef FixedPoint<tRawType, 2> F2;
    F0 half_denominator = RoundingHalfSum(a, F0::One());
    // Newton-Raphson division
    // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
    // Refer to that page for the logic behind the 48/17 and 32/17 constants.
    const F2 constant_48_over_17     = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
    const F2 constant_neg_32_over_17 = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
    F2 x                             = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
    for (int i = 0; i < 3; i++) {
        F2 half_denominator_times_x           = half_denominator * x;
        F2 one_minus_half_denominator_times_x = F2::One() - half_denominator_times_x;
        x                                     = x + Rescale<2>(x * one_minus_half_denominator_times_x);
    }
    return Rescale<0>(ExactMulByPot<-1>(x));
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for x > 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic_on_positive_values(FixedPoint<tRawType, tIntegerBits> a) {
    return one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(-a));
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic(FixedPoint<tRawType, tIntegerBits> a) {
    typedef FixedPoint<tRawType, tIntegerBits> InputF;
    typedef FixedPoint<tRawType, 0> ResultF;
    tRawType mask_if_positive  = MaskIfGreaterThan(a, InputF::Zero());
    tRawType mask_if_zero      = MaskIfZero(a);
    InputF abs_input           = SelectUsingMask(mask_if_positive, a, -a);
    ResultF result_if_positive = logistic_on_positive_values(abs_input);
    ResultF result_if_negative = ResultF::One() - result_if_positive;
    const ResultF one_half     = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(ResultF, 1 << 30, 0.5);
    return SelectUsingMask(mask_if_zero, one_half,
                           SelectUsingMask(mask_if_positive, result_if_positive, result_if_negative));
}

inline int MultiplyByQuantizedMultiplierSmallerThanOneExp(int x, int quantized_multiplier, int left_shift) {
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

inline int MultiplyByQuantizedMultiplier(int x, int quantized_multiplier, int shift) {
    int left_shift  = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
                               right_shift);
}

inline int MultiplyByQuantizedMultiplierGreaterThanOne(int x, int quantized_multiplier, int left_shift) {
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier);
}

inline int Offset(const std::vector<int> dims, int i0, int i1, int i2, int i3) {
    int b = dims.at(1);
    int c = dims.at(2);
    int d = dims.at(3);
    return i3 * b * c * d + i2 * c * d + i1 * d + i0;
}

#ifdef MNN_USE_NEON

template <>
struct FixedPointRawTypeTraits<int32x4_t> {
    typedef std::int32_t ScalarRawType;
    static constexpr int kLanes = 4;
};

template <>
struct FixedPointRawTypeTraits<int16x8_t> {
    typedef std::int16_t ScalarRawType;
    static constexpr int kLanes = 8;
};

template <>
inline int32x4_t BitAnd(int32x4_t a, int32x4_t b) {
    return vandq_s32(a, b);
}

template <>
inline int16x8_t BitAnd(int16x8_t a, int16x8_t b) {
    return vandq_s16(a, b);
}

template <>
inline int32x4_t BitOr(int32x4_t a, int32x4_t b) {
    return vorrq_s32(a, b);
}

template <>
inline int16x8_t BitOr(int16x8_t a, int16x8_t b) {
    return vorrq_s16(a, b);
}

template <>
inline int32x4_t BitXor(int32x4_t a, int32x4_t b) {
    return veorq_s32(a, b);
}

template <>
inline int16x8_t BitXor(int16x8_t a, int16x8_t b) {
    return veorq_s16(a, b);
}

template <>
inline int32x4_t BitNot(int32x4_t a) {
    return veorq_s32(a, vdupq_n_s32(-1));
}

template <>
inline int16x8_t BitNot(int16x8_t a) {
    return veorq_s16(a, vdupq_n_s16(-1));
}

template <>
inline int32x4_t Add(int32x4_t a, int32x4_t b) {
    return vaddq_s32(a, b);
}

template <>
inline int16x8_t Add(int16x8_t a, int16x8_t b) {
    return vaddq_s16(a, b);
}

template <>
inline int32x4_t Sub(int32x4_t a, int32x4_t b) {
    return vsubq_s32(a, b);
}

template <>
inline int16x8_t Sub(int16x8_t a, int16x8_t b) {
    return vsubq_s16(a, b);
}

template <>
inline int32x4_t Neg(int32x4_t a) {
    return vnegq_s32(a);
}

template <>
inline int16x8_t Neg(int16x8_t a) {
    return vnegq_s16(a);
}

template <>
inline int32x4_t ShiftLeft(int32x4_t a, int offset) {
    return vshlq_s32(a, vdupq_n_s32(offset));
}

template <>
inline int16x8_t ShiftLeft(int16x8_t a, int offset) {
    return vshlq_s16(a, vdupq_n_s16(offset));
}

template <>
inline int32x4_t ShiftRight(int32x4_t a, int offset) {
    return vshlq_s32(a, vdupq_n_s32(-offset));
}

template <>
inline int16x8_t ShiftRight(int16x8_t a, int offset) {
    return vshlq_s16(a, vdupq_n_s16(-offset));
}

template <>
inline int32x4_t SelectUsingMask(int32x4_t if_mask, int32x4_t then_val, int32x4_t else_val) {
    return vbslq_s32(vreinterpretq_u32_s32(if_mask), then_val, else_val);
}

template <>
inline int16x8_t SelectUsingMask(int16x8_t if_mask, int16x8_t then_val, int16x8_t else_val) {
    return vbslq_s16(vreinterpretq_u16_s16(if_mask), then_val, else_val);
}

template <>
inline int32x4_t MaskIfEqual(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vceqq_s32(a, b));
}

template <>
inline int16x8_t MaskIfEqual(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vceqq_s16(a, b));
}

template <>
inline int32x4_t MaskIfNotEqual(int32x4_t a, int32x4_t b) {
    return BitNot(MaskIfEqual(a, b));
}

template <>
inline int16x8_t MaskIfNotEqual(int16x8_t a, int16x8_t b) {
    return BitNot(MaskIfEqual(a, b));
}

template <>
inline int32x4_t MaskIfZero(int32x4_t a) {
    return MaskIfEqual(a, vdupq_n_s32(0));
}

template <>
inline int16x8_t MaskIfZero(int16x8_t a) {
    return MaskIfEqual(a, vdupq_n_s16(0));
}

template <>
inline int32x4_t MaskIfNonZero(int32x4_t a) {
    return vreinterpretq_s32_u32(vtstq_s32(a, a));
}

template <>
inline int16x8_t MaskIfNonZero(int16x8_t a) {
    return vreinterpretq_s16_u16(vtstq_s16(a, a));
}

template <>
inline int32x4_t MaskIfGreaterThan(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcgtq_s32(a, b));
}

template <>
inline int16x8_t MaskIfGreaterThan(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vcgtq_s16(a, b));
}

template <>
inline int32x4_t MaskIfGreaterThanOrEqual(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcgeq_s32(a, b));
}

template <>
inline int16x8_t MaskIfGreaterThanOrEqual(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vcgeq_s16(a, b));
}

template <>
inline int32x4_t MaskIfLessThan(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

template <>
inline int16x8_t MaskIfLessThan(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vcltq_s16(a, b));
}

template <>
inline int32x4_t MaskIfLessThanOrEqual(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcleq_s32(a, b));
}

template <>
inline int16x8_t MaskIfLessThanOrEqual(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vcleq_s16(a, b));
}

template <>
inline bool All(int32x4_t a) {
    a = vandq_s32(a, vextq_s32(a, a, 1));
    a = vandq_s32(a, vextq_s32(a, a, 2));
    return vgetq_lane_s32(a, 0);
}

template <>
inline bool All(int16x8_t a) {
    a = vandq_s16(a, vextq_s16(a, a, 1));
    a = vandq_s16(a, vextq_s16(a, a, 2));
    a = vandq_s16(a, vextq_s16(a, a, 4));
    return vgetq_lane_s16(a, 0);
}

template <>
inline bool Any(int32x4_t a) {
    a = vorrq_s32(a, vextq_s32(a, a, 1));
    a = vorrq_s32(a, vextq_s32(a, a, 2));
    return vgetq_lane_s32(a, 0);
}

template <>
inline bool Any(int16x8_t a) {
    a = vorrq_s16(a, vextq_s16(a, a, 1));
    a = vorrq_s16(a, vextq_s16(a, a, 2));
    a = vorrq_s16(a, vextq_s16(a, a, 4));
    return vgetq_lane_s16(a, 0);
}

template <>
inline int32x4_t RoundingHalfSum(int32x4_t a, int32x4_t b) {
    return vrhaddq_s32(a, b);
}

template <>
inline int16x8_t RoundingHalfSum(int16x8_t a, int16x8_t b) {
    return vrhaddq_s16(a, b);
}

template <>
inline int32x4_t SaturatingRoundingDoublingHighMul(int32x4_t a, int32x4_t b) {
    return vqrdmulhq_s32(a, b);
}

template <>
inline int16x8_t SaturatingRoundingDoublingHighMul(int16x8_t a, int16x8_t b) {
    return vqrdmulhq_s16(a, b);
}

template <>
inline int32x4_t RoundingDivideByPOT(int32x4_t x, int exponent) {
    const int32x4_t shift_vec  = vdupq_n_s32(-exponent);
    const int32x4_t fixup      = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
    const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
    return vrshlq_s32(fixed_up_x, shift_vec);
}

template <>
inline int16x8_t RoundingDivideByPOT(int16x8_t x, int exponent) {
    const int16x8_t shift_vec  = vdupq_n_s16(-exponent);
    const int16x8_t fixup      = vshrq_n_s16(vandq_s16(x, shift_vec), 15);
    const int16x8_t fixed_up_x = vqaddq_s16(x, fixup);
    return vrshlq_s16(fixed_up_x, shift_vec);
}

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, 1> {
    static int32x4_t eval(int32x4_t x) {
        return vqshlq_n_s32(x, Exponent);
    }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, -1> {
    static int32x4_t eval(int32x4_t x) {
        const int32x4_t fixup      = vshrq_n_s32(x, 31);
        const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
        return vrshrq_n_s32(fixed_up_x, -Exponent);
    }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int16x8_t, 1> {
    static int16x8_t eval(int16x8_t x) {
        return vqshlq_n_s16(x, Exponent);
    }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int16x8_t, -1> {
    static int16x8_t eval(int16x8_t x) {
        const int16x8_t fixup      = vshrq_n_s16(x, 15);
        const int16x8_t fixed_up_x = vqaddq_s16(x, fixup);
        return vrshrq_n_s16(fixed_up_x, -Exponent);
    }
};

template <>
inline int32x4_t Dup<int32x4_t>(std::int32_t x) {
    return vdupq_n_s32(x);
}

template <>
inline int16x8_t Dup<int16x8_t>(std::int16_t x) {
    return vdupq_n_s16(x);
}

// So far this is only needed for int16.
template <>
inline int16x8_t SaturatingAdd(int16x8_t a, int16x8_t b) {
    return vqaddq_s16(a, b);
}
#endif

} // namespace MNN

#endif /* CPUFixedPoint_HPP */
