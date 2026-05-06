#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void MNNBinaryMinInt8_RVV(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1,
                          ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params,
                          size_t elementSize, size_t needBroadcast) {
    int32_t zp0 = params->inputZeroPoint[0];
    int32_t zp1 = params->inputZeroPoint[1];
    int32_t zpOut = params->outputZeroPoint[0];
    int32_t scale0 = inputScalesInt32[0];
    int32_t scale1 = inputScalesInt32[1];
    int32_t minV = params->minValue;
    int32_t maxV = params->maxValue;

    int32_t scalar_val0 = (needBroadcast == 0) ? (inputRaw0[0] - zp0) * scale0 : 0;
    int32_t scalar_val1 = (needBroadcast == 1) ? (inputRaw1[0] - zp1) * scale1 : 0;

    size_t vl;
    for (size_t i = 0; i < elementSize; i += vl) {
        vl = __riscv_vsetvl_e8m1(elementSize - i);
        vint32m4_t vi0, vi1;

        if (needBroadcast == 0) {
            vi0 = __riscv_vmv_v_x_i32m4(scalar_val0, vl);
            vint8m1_t v1_8 = __riscv_vle8_v_i8m1(inputRaw1 + i, vl);
            vint32m4_t v1_32 = __riscv_vwadd_vx_i32m4(__riscv_vwsub_vx_i16m2(v1_8, zp1, vl), 0, vl);
            vi1 = __riscv_vmul_vx_i32m4(v1_32, scale1, vl);
        } else if (needBroadcast == 1) {
            vi1 = __riscv_vmv_v_x_i32m4(scalar_val1, vl);
            vint8m1_t v0_8 = __riscv_vle8_v_i8m1(inputRaw0 + i, vl);
            vint32m4_t v0_32 = __riscv_vwadd_vx_i32m4(__riscv_vwsub_vx_i16m2(v0_8, zp0, vl), 0, vl);
            vi0 = __riscv_vmul_vx_i32m4(v0_32, scale0, vl);
        } else {
            vint8m1_t v0_8 = __riscv_vle8_v_i8m1(inputRaw0 + i, vl);
            vint8m1_t v1_8 = __riscv_vle8_v_i8m1(inputRaw1 + i, vl);
            vint32m4_t v0_32 = __riscv_vwadd_vx_i32m4(__riscv_vwsub_vx_i16m2(v0_8, zp0, vl), 0, vl);
            vint32m4_t v1_32 = __riscv_vwadd_vx_i32m4(__riscv_vwsub_vx_i16m2(v1_8, zp1, vl), 0, vl);
            vi0 = __riscv_vmul_vx_i32m4(v0_32, scale0, vl);
            vi1 = __riscv_vmul_vx_i32m4(v1_32, scale1, vl);
        }

        // 取最小值
        vint32m4_t vres = __riscv_vmin_vv_i32m4(vi0, vi1, vl);

        // 模拟 C++ 的带舍入的整数除法:
        // 伪代码: res < 0 ? (res - 32768)/65536 : (res + 32768)/65536
        vbool8_t is_neg = __riscv_vmslt_vx_i32m4_b8(vres, 0, vl);
        vint32m4_t offset_val = __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(32768, vl), -32768, is_neg, vl);
        vres = __riscv_vadd_vv_i32m4(vres, offset_val, vl);
        // 使用 RVV 整型除法对齐 C++ 向零截断行为
        vres = __riscv_vdiv_vx_i32m4(vres, 65536, vl);

        // 加上输出零点并限幅
        vres = __riscv_vadd_vx_i32m4(vres, zpOut, vl);
        vres = __riscv_vmax_vx_i32m4(vres, minV, vl);
        vres = __riscv_vmin_vx_i32m4(vres, maxV, vl);

        // 压回 int8
        vint16m2_t vout16 = __riscv_vncvt_x_x_w_i16m2(vres, vl);
        vint8m1_t vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(outputRaw + i, vout8, vl);
    }
}