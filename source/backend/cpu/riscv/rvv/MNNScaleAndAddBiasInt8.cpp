#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void MNNScaleAndAddBiasInt8_RVV(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha,
                                int32_t mShiftBits, ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint,
                                int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack) {
    int32_t inZP = *inputZeroPoint;
    int32_t outZP = *outputZeroPoint;
    int32_t d = mShiftBits - 1;
    int32_t shift_div = 1 << mShiftBits; // 相当于 C++ 中的 (1 << mShiftBits)
    int32_t offset_pos = 1 << d;
    int32_t offset_neg = -(1 << d);

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

                // 处理位移与舍入
                vbool8_t is_neg = __riscv_vmslt_vx_i32m4_b8(val, 0, vl);
                vint32m4_t v_add =
                    __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(offset_pos, vl), offset_neg, is_neg, vl);
                val = __riscv_vadd_vv_i32m4(val, v_add, vl);
                val = __riscv_vdiv_vx_i32m4(val, shift_div, vl);

                // 加上 outZP 并限幅
                val = __riscv_vadd_vx_i32m4(val, outZP, vl);
                val = __riscv_vmax_vx_i32m4(val, minValue, vl);
                val = __riscv_vmin_vx_i32m4(val, maxValue, vl);

                vint16m2_t vout16 = __riscv_vncvt_x_x_w_i16m2(val, vl);
                vint8m1_t vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
                __riscv_vse8_v_i8m1(dstX + i, vout8, vl);
            }
        }
    }
}