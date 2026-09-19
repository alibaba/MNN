#include <riscv_vector.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

void MNNConvInt8ComputeBiasFloat_RVV(float* dst, const int32_t* bias, const float* weightScale, float inputScale,
                                     float outputScale, size_t size) {
    // Operation order must match the scalar path in CPUConvolution.cpp:
    //     bias * weightScale * inputScale / outputScale
    //
    // outputScale is loop-invariant, so the division is replaced by a reciprocal
    // computed once outside the loop, plus a Markstein correction:
    //
    //     r  = 1.0f / outputScale   // hoisted
    //     q0 = t * r                // vfmul
    //     e  = fma(-outputScale, q0, t)   // exact residual, vfnmsac
    //     q  = fma(r, e, q0)        // corrected quotient, vfmacc
    //
    // The correction makes the result identical to the correctly rounded t /
    // outputScale on every finite input, so this form does not change any output
    // bit. It costs one multiply and two FMAs per element instead of one divide.
    const bool needScale = (inputScale != 0.0f && outputScale != 0.0f);
    float r = needScale ? (1.0f / outputScale) : 0.0f;
    // 1.0f / outputScale overflows to inf when outputScale is subnormal, which the
    // per-element division does not do, and a zero or subnormal reciprocal breaks
    // the correction. The test runs once outside the loop, and when it fails the
    // loop keeps the original divide.
    const bool rcpOk = std::isfinite(r) && r != 0.0f && (fabsf(r) >= 1.17549435e-38f);
    if (!rcpOk) {
        r = 0.0f;
    }
    size_t offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m8(size - offset);
        const vint32m8_t biasValue = __riscv_vle32_v_i32m8(bias + offset, vl);
        vfloat32m8_t value = __riscv_vfcvt_f_x_v_f32m8(biasValue, vl);
        const vfloat32m8_t scale = __riscv_vle32_v_f32m8(weightScale + offset, vl);
        value = __riscv_vfmul_vv_f32m8(value, scale, vl);
        if (needScale) {
            value = __riscv_vfmul_vf_f32m8(value, inputScale, vl);
            if (rcpOk) {
                const vfloat32m8_t q0 = __riscv_vfmul_vf_f32m8(value, r, vl);
                const vfloat32m8_t e = __riscv_vfnmsac_vf_f32m8(value, outputScale, q0, vl); // value - outputScale * q0
                value = __riscv_vfmacc_vf_f32m8(q0, r, e, vl);                              // q0 + r * e
            } else {
                value = __riscv_vfdiv_vf_f32m8(value, outputScale, vl);
            }
        }
        __riscv_vse32_v_f32m8(dst + offset, value, vl);
        offset += vl;
    }
}
