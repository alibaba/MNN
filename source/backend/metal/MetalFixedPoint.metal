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

/*
 Convert to metal by MNN.
 Copyright © 2018, Alibaba Group Holding Limited
 */

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

namespace MNN {
    // Part 1: Low-level integer-arithmetic primitives.
    template <typename tIntegerType>
    struct FixedPointRawTypeTraits {};
    
    template <>
    struct FixedPointRawTypeTraits<int32_t> {
        typedef int32_t ScalarRawType;
        static constant int kLanes = 1;
    };

    template <>
    struct FixedPointRawTypeTraits<int16_t> {
        typedef int16_t ScalarRawType;
        static constant int kLanes = 1;
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
    tIntegerType SelectUsingMask(tIntegerType if_mask, tIntegerType then_val,
                                 tIntegerType else_val) {
        return BitXor(BitAnd(if_mask, then_val), BitAnd(BitNot(if_mask), else_val));
    }
    
    // For each input scalar, the corresponding bits of the result are set if the
    // input scalar is non-zero.
    template <typename tIntegerType>
    tIntegerType MaskIfNonZero(tIntegerType a) {
        constexpr tIntegerType zero = 0;
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
//        static_assert(is_same<IntegerType, void>::value, "unimplemented");
        return a;
    }
    
    template <>
    inline int32_t RoundingHalfSum(int32_t a, int32_t b) {
        return hadd(a, b);    }
    
    template <>
    inline int16_t RoundingHalfSum(int16_t a, int16_t b) {
        return hadd(a, b);
    }
    
    template <typename IntegerType>
    IntegerType SaturatingAdd(IntegerType a, IntegerType b) {
//        static_assert(is_same<IntegerType, void>::value, "unimplemented");
        return a;
    }
    
    // So far this is only needed for int16.
    template <>
    inline int16_t SaturatingAdd(int16_t a, int16_t b) {
        int32_t a32 = a;
        int32_t b32 = b;
        int32_t sum = a32 + b32;
        return static_cast<int16_t>(min(32767, max(-32768, sum)));
    }
    
    // Returns a+b, saturating if the integers are 16bit or narrower,
    // otherwise just a plain addition.
    template <typename IntegerType, bool Is16Bit>
    struct AddSaturatingIf16BitImpl {
        static IntegerType Run(IntegerType a, IntegerType b) { return Add(a, b); }
    };
    template <typename IntegerType>
    struct AddSaturatingIf16BitImpl<IntegerType, true> {
        static IntegerType Run(IntegerType a, IntegerType b) {
            return SaturatingAdd(a, b);
        }
    };
    template <typename IntegerType>
    IntegerType AddSaturatingIf16Bit(IntegerType a, IntegerType b) {
        using ScalarType =
        typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
        return AddSaturatingIf16BitImpl<IntegerType, sizeof(ScalarType) == 2>::Run(a,
                                                                                   b);
    }
    
    // Returns the product of a run-time integer value by a compile-time power
    // of two, with either a positive exponent (equivalent to an arithmetic
    // left shift, saturating) or a negative exponent (equivalent to an arithmetic
    // right shift, rounding to nearest).
    template <int Exponent, typename IntegerType,
    int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
    struct ImplSaturatingRoundingMultiplyByPOT {};
    
    template <int Exponent, typename IntegerType>
    struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 0> {
        static IntegerType eval(IntegerType x) { return x; }
    };
    
    template <int Exponent, typename IntegerType>
    struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 1> {
        static IntegerType eval(IntegerType x) {
            using ScalarIntegerType =
            typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
            const IntegerType min = Dup<IntegerType>(num_limits<ScalarIntegerType>::min());
            const IntegerType max = Dup<IntegerType>(num_limits<ScalarIntegerType>::max());
            const int ScalarIntegerTypeBits = 8 * sizeof(ScalarIntegerType);
            
            const int32_t threshold = ((1 << (ScalarIntegerTypeBits - 1 - Exponent)) - 1);
            const IntegerType positive_mask = MaskIfGreaterThan(x, Dup<IntegerType>(threshold));
            const IntegerType negative_mask = MaskIfLessThan(x, Dup<IntegerType>(-threshold));
            
            IntegerType result = ShiftLeft(x, Exponent);
            result = SelectUsingMask(positive_mask, max, result);
            result = SelectUsingMask(negative_mask, min, result);
            return result;
        }
    };
    
    template <int Exponent, typename IntegerType>
    struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, -1> {
        static IntegerType eval(IntegerType x) {
            return round_divide_by_pot<IntegerType>(x, -Exponent);
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
        
        static constant int kTotalBits = 8 * sizeof(ScalarRawType);
        static constant int kIntegerBits = tIntegerBits;
        static constant int kFractionalBits = kTotalBits - 1 - kIntegerBits;
//        static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits, "bad IntegerBits");
        
        typedef FixedPoint<ScalarRawType, kIntegerBits> ScalarFixedPointType;
        
        static const ScalarRawType ScalarRawMin() {
            return num_limits<ScalarRawType>::min();
        }
        
        static const ScalarRawType ScalarRawMax() {
            return num_limits<ScalarRawType>::max();
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
            constexpr int kOffset = kFractionalBits + Exponent;
//            static_assert(kOffset < 31, "Constant not exactly representable in this fixed-point format");
            return FromScalarRaw(ScalarRawType(1) << kOffset);
        }
        
        static FixedPoint Zero() { return FromScalarRaw(0); }
        
        static FixedPoint One() {
            return FromScalarRaw(kIntegerBits == 0
                                 ? ScalarRawMax()
                                 : (ScalarRawType(1) << (kIntegerBits == 0 ? 0 : kFractionalBits)));
        }
    
        
        RawType raw() const { return i_; }
        thread RawType& raw() { return i_; }
        
    private:
        RawType i_;
    };
    
    // Part 3: implementation of arithmetic operators for the
    // FixedPoint class, and a few related functions.
    
    // A FixedPoint multiplication is just a
    // saturate_round_x2_high_mul operation on the underlying
    // raw integer values. The IntegerBits simply add up, as is obvious
    // from the fact that the range is [-2^IntegerBits, 2^IntegerBits).
    template <typename tRawType, int tIntegerBits_a, int tIntegerBits_b>
    FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> operator*(
                                                                    FixedPoint<tRawType, tIntegerBits_a> a,
                                                                    FixedPoint<tRawType, tIntegerBits_b> b) {
        FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> c;
        c.raw() = saturate_round_x2_high_mul(a.raw(), b.raw());
        return c;
    }
    
    // Tweaking IntegerBits gives exact multiplication by a power of two.
    template <int tExponent, typename tRawType, int tIntegerBits>
    FixedPoint<tRawType, tExponent + tIntegerBits> ExactMulByPot(
                                                                 FixedPoint<tRawType, tIntegerBits> a) {
        FixedPoint<tRawType, tExponent + tIntegerBits> c;
        c.raw() = a.raw();
        return c;
    }
    
    // If we want to leave IntegerBits fixed, then multiplication
    // by a power of two has to be saturating/rounding, not exact anymore.
    template <int tExponent, typename tRawType, int tIntegerBits>
    FixedPoint<tRawType, tIntegerBits> SaturatingRoundingMultiplyByPOT(
                                                                       FixedPoint<tRawType, tIntegerBits> a) {
        return FixedPoint<tRawType, tIntegerBits>::FromRaw(
                                                           SaturatingRoundingMultiplyByPOT<tExponent>(a.raw()));
    }
    
    // Generic arithmetic operators.
    
#define MAKE_FIXEDPOINT_UNARY_FUNC(FuncName, ImplFuncName)                     \
template <typename tRawType, int tIntegerBits>                               \
FixedPoint<tRawType, tIntegerBits> FuncName(                                 \
FixedPoint<tRawType, tIntegerBits> a) {                                  \
return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw())); \
}
    
#define MAKE_FIXEDPOINT_BINARY_FUNC(FuncName, ImplFuncName) \
template <typename tRawType, int tIntegerBits>            \
FixedPoint<tRawType, tIntegerBits> FuncName(              \
FixedPoint<tRawType, tIntegerBits> a,                 \
FixedPoint<tRawType, tIntegerBits> b) {               \
return FixedPoint<tRawType, tIntegerBits>::FromRaw(     \
ImplFuncName(a.raw(), b.raw()));                    \
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
    
#define MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(FuncName)  \
template <typename tRawType, int tIntegerBits>            \
tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
return FuncName(a.raw());                               \
}
    
#define MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(FuncName) \
template <typename tRawType, int tIntegerBits>            \
tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a,   \
FixedPoint<tRawType, tIntegerBits> b) { \
return FuncName(a.raw(), b.raw());                      \
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
    FixedPoint<tRawType, tIntegerBits> SelectUsingMask(
                                                       tRawType if_mask, FixedPoint<tRawType, tIntegerBits> then_val,
                                                       FixedPoint<tRawType, tIntegerBits> else_val) {
        return FixedPoint<tRawType, tIntegerBits>::FromRaw(
                                                           SelectUsingMask(if_mask, then_val.raw(), else_val.raw()));
    }
    
    template <typename tRawType, int tIntegerBits>
    bool operator==(FixedPoint<tRawType, tIntegerBits> a,
                    FixedPoint<tRawType, tIntegerBits> b) {
        return All(MaskIfEqual(a.raw(), b.raw()));
    }
    
    template <typename tRawType, int tIntegerBits>
    bool operator!=(FixedPoint<tRawType, tIntegerBits> a,
                    FixedPoint<tRawType, tIntegerBits> b) {
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
    
    // Rescale changes the number of IntegerBits and updates the underlying
    // raw integer value accordingly.
    template <int tIntegerBitsDst, typename tRawType, int tIntegerBitsSrc>
    FixedPoint<tRawType, tIntegerBitsDst> Rescale(
                                                  FixedPoint<tRawType, tIntegerBitsSrc> x) {
        constexpr int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
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
    inline typename FixedPointType::ScalarRawType RescaleConstantInitializer(int32_t int32_value) {
        typedef typename FixedPointType::ScalarRawType ScalarRawType;
        constexpr int ScalarTypeBits = 8 * sizeof(ScalarRawType);
        return static_cast<ScalarRawType>(round_divide_by_pot<int32_t>(int32_value, 32 - ScalarTypeBits));
    }

#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawInt32Value, DoubleValue) \
(FixedPointType::FromScalarRaw(RescaleConstantInitializer<FixedPointType>(ScalarRawInt32Value)))
    
    // Implementation of exponential function.
    
    // Returns exp(x) for x in [-1/4, 0).
    template <typename tRawType>
    FixedPoint<tRawType, 0> exp_on_interval_between_negative_one_quarter_and_0_excl(FixedPoint<tRawType, 0> a) {
        typedef FixedPoint<tRawType, 0> F;
        const F constant_term =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 1895147668, exp(-1.0 / 8.0));
        const F constant_1_over_3 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 715827883, 1.0 / 3.0);
        // We're evaluating a Taylor expansion around -1/8, so we do the change of
        // variable: x = a + 1/8.
        // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
        F x = a + F::template ConstantPOT<-3>();
        F x2 = x * x;
        F x3 = x2 * x;
        F x4 = x2 * x2;
        F x4_over_4 = SaturatingRoundingMultiplyByPOT<-2>(x4);
        F x4_over_24_plus_x3_over_6_plus_x2_over_2 =
            SaturatingRoundingMultiplyByPOT<-1>(((x4_over_4 + x3) * constant_1_over_3) + x2);
        return AddSaturatingIf16Bit(constant_term,
                                    constant_term * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2));
    }
    
    // Returns exp(x) for x < 0.
    template <typename tRawType, int tIntegerBits>
    FixedPoint<tRawType, 0> exp_on_negative_values(FixedPoint<tRawType, tIntegerBits> a) {
        typedef FixedPoint<tRawType, tIntegerBits> InputF;
        typedef FixedPoint<tRawType, 0> ResultF;
        constexpr int kFractionalBits = InputF::kFractionalBits;
        constexpr int kIntegerBits = InputF::kIntegerBits;
        const InputF kOneQuarter = InputF::template ConstantPOT<-2>();
        InputF mask = kOneQuarter - InputF::FromScalarRaw(1);
        InputF a_mod_quarter_minus_one_quarter = (a & mask) - kOneQuarter;
        ResultF result = exp_on_interval_between_negative_one_quarter_and_0_excl(Rescale<0>(a_mod_quarter_minus_one_quarter));
        tRawType remainder = (a_mod_quarter_minus_one_quarter - a).raw();
        
#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)         \
if (kIntegerBits > Exponent) {                                            \
    const ResultF kMultiplier = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(ResultF, FixedPointMultiplier, exp(-pow(2.0, Exponent))); \
    constexpr int kShiftAmount = kIntegerBits > Exponent ? kFractionalBits + Exponent : 0; \
    result = SelectUsingMask(                                               \
        MaskIfNonZero(BitAnd(remainder, Dup<tRawType>(1 << kShiftAmount))), \
        result * kMultiplier, result);                                      \
}
        
        GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);
        GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);
        GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);
        GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);
        GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);
        GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);
        GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);
        
#undef GEMMLOWP_EXP_BARREL_SHIFTER
        
        constexpr int clampB = kIntegerBits > 5 ? 36 - kIntegerBits : 0;
        if (kIntegerBits > 5) {
            const InputF clamp =
            GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(InputF, -(1 << clampB), -32.0);
            result = SelectUsingMask(MaskIfLessThan(a, clamp), ResultF::Zero(), result);
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
        F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
        for (int i = 0; i < 3; i++) {
            F2 half_denominator_times_x = half_denominator * x;
            F2 one_minus_half_denominator_times_x =
            F2::One() - half_denominator_times_x;
            x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
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
        tRawType mask_if_zero = MaskIfZero(a);
        InputF n = SelectUsingMask(mask_if_negative, a, -a);
        ResultF t = neg_tanh_on_negative_values(n);
        return SelectUsingMask(mask_if_zero, ResultF::Zero(),
                               SelectUsingMask(mask_if_negative, -t, t));
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
        F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
        for (int i = 0; i < 3; i++) {
            F2 half_denominator_times_x = half_denominator * x;
            F2 one_minus_half_denominator_times_x =
            F2::One() - half_denominator_times_x;
            x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
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
        tRawType mask_if_positive = MaskIfGreaterThan(a, InputF::Zero());
        tRawType mask_if_zero = MaskIfZero(a);
        InputF abs_input = SelectUsingMask(mask_if_positive, a, -a);
        ResultF result_if_positive = logistic_on_positive_values(abs_input);
        ResultF result_if_negative = ResultF::One() - result_if_positive;
        const ResultF one_half =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(ResultF, 1 << 30, 0.5);
        return SelectUsingMask(mask_if_zero, one_half, SelectUsingMask(mask_if_positive, result_if_positive, result_if_negative));
    }
    
    inline int MultiplyByQuantizedMultiplierSmallerThanOneExp(int x, int quantized_multiplier, int left_shift) {
        return round_divide_by_pot(saturate_round_x2_high_mul(x, quantized_multiplier), -left_shift);
    }
    
    inline int MultiplyByQuantizedMultiplier(int x, int quantized_multiplier, int shift) {
        int left_shift = shift > 0 ? shift : 0;
        int right_shift = shift > 0 ? 0 : -shift;
        return round_divide_by_pot(saturate_round_x2_high_mul(x * (1 << left_shift), quantized_multiplier), right_shift);
    }
    
    inline int MultiplyByQuantizedMultiplierGreaterThanOne(int x, int quantized_multiplier, int left_shift) {
        return saturate_round_x2_high_mul(x * (1 << left_shift), quantized_multiplier);
    }
}
