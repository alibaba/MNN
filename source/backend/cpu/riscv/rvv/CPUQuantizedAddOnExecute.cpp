#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#ifndef __RISCV_VXRM_RNU
#define __RISCV_VXRM_RNU 0
#endif

namespace {

// RVV vsmul with RNU has the same rounding and saturation semantics as MNN's scalar fixed-point helper.
static inline vint32m8_t saturatingRoundingDoublingHighMul(vint32m8_t value, int32_t multiplier, size_t vl) {
    return __riscv_vsmul_vx_i32m8(value, multiplier, __RISCV_VXRM_RNU, vl);
}

// Match RoundingDivideByPOT exactly, including ties for negative values.
static inline vint32m8_t roundingDivideByPOT(vint32m8_t value, int exponent, size_t vl) {
    if (exponent <= 0) {
        return value;
    }

    const int32_t maskValue      = static_cast<int32_t>((static_cast<uint64_t>(1) << exponent) - 1);
    const int32_t thresholdValue = maskValue >> 1;
    vint32m8_t remainder         = __riscv_vand_vx_i32m8(value, maskValue, vl);
    vint32m8_t threshold         = __riscv_vmv_v_x_i32m8(thresholdValue, vl);
    vbool4_t negative            = __riscv_vmslt_vx_i32m8_b4(value, 0, vl);
    threshold                    = __riscv_vadd_vx_i32m8_mu(negative, threshold, threshold, 1, vl);
    vbool4_t shouldRound         = __riscv_vmslt_vv_i32m8_b4(threshold, remainder, vl);
    vint32m8_t result            = __riscv_vsra_vx_i32m8(value, exponent, vl);
    return __riscv_vadd_vx_i32m8_mu(shouldRound, result, result, 1, vl);
}

} // namespace

void CPUQuantizedAdd_RVV(const uint8_t* input1Data, const uint8_t* input2Data, uint8_t* outputData, size_t size,
                         int32_t input1Offset, int32_t input2Offset, int32_t outputOffset,
                         int32_t leftShiftResult1, int32_t leftShiftResult2, int32_t input1Multiplier,
                         int32_t input2Multiplier, int32_t rightShift1, int32_t rightShift2, int32_t leftShiftOut,
                         int32_t outputMultiplier, int32_t rightShiftOut, int32_t outputActivationMin,
                         int32_t outputActivationMax) {
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e32m8(size);

        vuint8m2_t input1U8   = __riscv_vle8_v_u8m2(input1Data, vl);
        vuint16m4_t input1U16 = __riscv_vwaddu_vx_u16m4(input1U8, 0, vl);
        vuint32m8_t input1U32 = __riscv_vwaddu_vx_u32m8(input1U16, 0, vl);
        vint32m8_t input1     = __riscv_vreinterpret_v_u32m8_i32m8(input1U32);

        vuint8m2_t input2U8   = __riscv_vle8_v_u8m2(input2Data, vl);
        vuint16m4_t input2U16 = __riscv_vwaddu_vx_u16m4(input2U8, 0, vl);
        vuint32m8_t input2U32 = __riscv_vwaddu_vx_u32m8(input2U16, 0, vl);
        vint32m8_t input2     = __riscv_vreinterpret_v_u32m8_i32m8(input2U32);

        input1 = __riscv_vadd_vx_i32m8(input1, input1Offset, vl);
        input1 = __riscv_vmul_vx_i32m8(input1, leftShiftResult1, vl);
        input1 = saturatingRoundingDoublingHighMul(input1, input1Multiplier, vl);
        input1 = roundingDivideByPOT(input1, rightShift1, vl);

        input2 = __riscv_vadd_vx_i32m8(input2, input2Offset, vl);
        input2 = __riscv_vmul_vx_i32m8(input2, leftShiftResult2, vl);
        input2 = saturatingRoundingDoublingHighMul(input2, input2Multiplier, vl);
        input2 = roundingDivideByPOT(input2, rightShift2, vl);

        vint32m8_t sum = __riscv_vadd_vv_i32m8(input1, input2, vl);
        sum            = __riscv_vmul_vx_i32m8(sum, 1 << leftShiftOut, vl);
        sum            = saturatingRoundingDoublingHighMul(sum, outputMultiplier, vl);
        sum            = roundingDivideByPOT(sum, rightShiftOut, vl);
        sum = __riscv_vadd_vx_i32m8(sum, outputOffset, vl);
        sum = __riscv_vmax_vx_i32m8(sum, outputActivationMin, vl);
        sum = __riscv_vmin_vx_i32m8(sum, outputActivationMax, vl);

        vuint32m8_t outputU32 = __riscv_vreinterpret_v_i32m8_u32m8(sum);
        vuint16m4_t outputU16 = __riscv_vncvt_x_x_w_u16m4(outputU32, vl);
        vuint8m2_t outputU8   = __riscv_vncvt_x_x_w_u8m2(outputU16, vl);
        __riscv_vse8_v_u8m2(outputData, outputU8, vl);

        input1Data += vl;
        input2Data += vl;
        outputData += vl;
        size -= vl;
    }
}
