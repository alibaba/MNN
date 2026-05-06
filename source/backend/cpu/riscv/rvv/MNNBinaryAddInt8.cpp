#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void MNNBinaryAddInt8_RVV(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1,
                          ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params,
                          size_t elementSize, size_t needBroadcast) {
    int32_t zp0 = params->inputZeroPoint[0];
    int32_t zp1 = params->inputZeroPoint[1];
    int32_t zpOut = params->outputZeroPoint[0];
    float scale0 = inputScalesFp32[0];
    float scale1 = inputScalesFp32[1];
    float scaleOut = inputScalesFp32[2];
    int32_t minV = params->minValue;
    int32_t maxV = params->maxValue;

    // 预先处理标量值，避免在循环中重复计算
    float scalar_val0 = (needBroadcast == 0) ? (inputRaw0[0] - zp0) * scale0 : 0.0f;
    float scalar_val1 = (needBroadcast == 1) ? (inputRaw1[0] - zp1) * scale1 : 0.0f;

    size_t vl;
    for (size_t i = 0; i < elementSize; i += vl) {
        vl = __riscv_vsetvl_e8m1(elementSize - i);
        vfloat32m4_t vf0, vf1;

        // 1. 广播分支与反量化加载
        if (needBroadcast == 0) {
            vf0 = __riscv_vfmv_v_f_f32m4(scalar_val0, vl);
            vint8m1_t vi1 = __riscv_vle8_v_i8m1(inputRaw1 + i, vl);
            vint16m2_t vw1 = __riscv_vwsub_vx_i16m2(vi1, zp1, vl);
            vint32m4_t vw1_32 = __riscv_vwadd_vx_i32m4(vw1, 0, vl);
            vf1 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(vw1_32, vl), scale1, vl);
        } else if (needBroadcast == 1) {
            vf1 = __riscv_vfmv_v_f_f32m4(scalar_val1, vl);
            vint8m1_t vi0 = __riscv_vle8_v_i8m1(inputRaw0 + i, vl);
            vint16m2_t vw0 = __riscv_vwsub_vx_i16m2(vi0, zp0, vl);
            vint32m4_t vw0_32 = __riscv_vwadd_vx_i32m4(vw0, 0, vl);
            vf0 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(vw0_32, vl), scale0, vl);
        } else {
            vint8m1_t vi0 = __riscv_vle8_v_i8m1(inputRaw0 + i, vl);
            vint8m1_t vi1 = __riscv_vle8_v_i8m1(inputRaw1 + i, vl);
            vint16m2_t vw0 = __riscv_vwsub_vx_i16m2(vi0, zp0, vl);
            vint16m2_t vw1 = __riscv_vwsub_vx_i16m2(vi1, zp1, vl);
            vint32m4_t vw0_32 = __riscv_vwadd_vx_i32m4(vw0, 0, vl);
            vint32m4_t vw1_32 = __riscv_vwadd_vx_i32m4(vw1, 0, vl);
            vf0 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(vw0_32, vl), scale0, vl);
            vf1 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(vw1_32, vl), scale1, vl);
        }

        // 2. 真实计算 (加法)
        vfloat32m4_t vsum = __riscv_vfadd_vv_f32m4(vf0, vf1, vl);

        // 3. 重量化 (乘以 scaleOut)
        vsum = __riscv_vfmul_vf_f32m4(vsum, scaleOut, vl);

        // 4. 浮点转整型 (RVV 的 vfcvt_x_f 默认向最近偶数舍入，能很好地拟合 roundf)
        vint32m4_t vout32 = __riscv_vfcvt_x_f_v_i32m4(vsum, vl);

        // 5. 加上输出零点并限幅
        vout32 = __riscv_vadd_vx_i32m4(vout32, zpOut, vl);
        vout32 = __riscv_vmax_vx_i32m4(vout32, minV, vl);
        vout32 = __riscv_vmin_vx_i32m4(vout32, maxV, vl);

        // 6. 窄化回 int8 并存储
        vint16m2_t vout16 = __riscv_vncvt_x_x_w_i16m2(vout32, vl);
        vint8m1_t vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(outputRaw + i, vout8, vl);
    }
}