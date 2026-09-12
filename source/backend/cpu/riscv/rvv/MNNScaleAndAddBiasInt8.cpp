#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// Truncating division by a power of two, bit-exact with C++ integer division
// (which truncates toward zero):
//     x >= 0 -> x >> shift
//     x <  0 -> (x + (1 << shift) - 1) >> shift
// A hardware vector integer division is emulated on RVV and costs far more than
// the add + select + shift sequence below, while producing identical results.
static inline vint32m4_t truncDivPow2(vint32m4_t x, int shift, size_t vl) {
    vbool8_t neg = __riscv_vmslt_vx_i32m4_b8(x, 0, vl);
    vint32m4_t fix = __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(0, vl), (int32_t)((1 << shift) - 1), neg, vl);
    return __riscv_vsra_vx_i32m4(__riscv_vadd_vv_i32m4(x, fix, vl), shift, vl);
}

void MNNScaleAndAddBiasInt8_RVV(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha,
                                int32_t mShiftBits, ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint,
                                int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack) {
    const int32_t inZP = *inputZeroPoint;
    const int32_t outZP = *outputZeroPoint;
    // mShiftBits == 0 is degenerate (the scalar reference evaluates 1 << -1 and is
    // undefined behaviour), so keep the division an identity instead of copying it.
    const int shift = mShiftBits > 0 ? mShiftBits : 0;
    const int32_t roundOffset = shift > 0 ? (1 << (shift - 1)) : 0;

    for (int z = 0; z < biasNumber; ++z) {
        auto dstZ = dst + planeNumber * pack * z;
        const auto srcZ = src + planeNumber * pack * z;
        const auto biasZ = bias + pack * z;
        const auto alphaZ = alpha + pack * z;

        for (int p = 0; p < planeNumber; ++p) {
            auto dstX = dstZ + pack * p;
            const auto srcX = srcZ + pack * p;

            size_t vl;
            for (size_t i = 0; i < pack; i += vl) {
                vl = __riscv_vsetvl_e8m1(pack - i);

                vint8m1_t v_src = __riscv_vle8_v_i8m1(srcX + i, vl);
                vint32m4_t v_src32 = __riscv_vwadd_vx_i32m4(__riscv_vwsub_vx_i16m2(v_src, inZP, vl), 0, vl);

                vint32m4_t v_alpha = __riscv_vle32_v_i32m4(alphaZ + i, vl);
                vint32m4_t v_bias = __riscv_vle32_v_i32m4(biasZ + i, vl);

                // val = (src - inZP) * alpha + bias
                vint32m4_t val = __riscv_vmacc_vv_i32m4(v_bias, v_src32, v_alpha, vl);

                // Scalar reference: val < 0 ? (val - 2^(shift-1)) / 2^shift : (val + 2^(shift-1)) / 2^shift
                vbool8_t is_neg = __riscv_vmslt_vx_i32m4_b8(val, 0, vl);
                vint32m4_t v_add =
                    __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(roundOffset, vl), -roundOffset, is_neg, vl);
                val = __riscv_vadd_vv_i32m4(val, v_add, vl);
                if (shift > 0) {
                    val = truncDivPow2(val, shift, vl);
                }

                // Add the output zero point and saturate.
                val = __riscv_vadd_vx_i32m4(val, outZP, vl);
                val = __riscv_vmax_vx_i32m4(val, (int32_t)minValue, vl);
                val = __riscv_vmin_vx_i32m4(val, (int32_t)maxValue, vl);

                vint16m2_t vout16 = __riscv_vncvt_x_x_w_i16m2(val, vl);
                vint8m1_t vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
                __riscv_vse8_v_i8m1(dstX + i, vout8, vl);
            }
        }
    }
}
