//
//  MNNReluWithSlopeChannelInt8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../../compute/Int8FunctionsOpt.h"
void MNNReluWithSlopeChannelInt8_RVV(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber,
                                     size_t depthQuad, const QuanPrePostParameters* params, size_t pack) {
    const float offset = 0.f;
    const int8_t* srcPtr = src;
    int8_t* dstPtr = dst;

    const float inputZero = (float)params->inputZeroPoint[0] + offset;
    const float outputZero = (float)params->outputZeroPoint[0] + offset;
    const float inputScale = params->inputScale[0];
    const float outputScale = params->outputScale[0];
    const int32_t minval = params->minValue + (int32_t)offset;
    const int32_t maxval = params->maxValue + (int32_t)offset;

    const size_t vl = __riscv_vsetvl_e8m1(pack);

    vint32m4_t v_min = __riscv_vmv_v_x_i32m4(minval, vl);
    vint32m4_t v_max = __riscv_vmv_v_x_i32m4(maxval, vl);
    vfloat32m4_t v_inputZero = __riscv_vfmv_v_f_f32m4(inputZero, vl);
    vfloat32m4_t v_inputScale = __riscv_vfmv_v_f_f32m4(inputScale, vl);
    vfloat32m4_t v_outputScale = __riscv_vfmv_v_f_f32m4(outputScale, vl);
    vfloat32m4_t v_outputZero = __riscv_vfmv_v_f_f32m4(outputZero, vl);

    for (size_t j = 0; j < depthQuad; ++j) {
        const float* slopeZ = slope + pack * j;
        const int8_t* srcZ = srcPtr + pack * j * planeNumber;
        int8_t* dstZ = dstPtr + pack * j * planeNumber;

        vfloat32m4_t v_slope = __riscv_vle32_v_f32m4(slopeZ, vl);

        for (size_t i = 0; i < planeNumber; ++i) {
            const int8_t* srcBase = srcZ + pack * i;
            int8_t* dstBase = dstZ + pack * i;

            vint8m1_t v_src = __riscv_vle8_v_i8m1(srcBase, vl);
            vint32m4_t v_src_s32 = __riscv_vsext_vf4_i32m4(v_src, vl);
            vfloat32m4_t v_src_f32 = __riscv_vfcvt_f_x_v_f32m4(v_src_s32, vl);

            vfloat32m4_t v_val = __riscv_vfsub_vv_f32m4(v_src_f32, v_inputZero, vl);
            v_val = __riscv_vfmul_vv_f32m4(v_val, v_inputScale, vl);

            vbool8_t v_mask = __riscv_vmflt_vf_f32m4_b8(v_val, 0.0f, vl);
            vfloat32m4_t v_val_slope = __riscv_vfmul_vv_f32m4(v_val, v_slope, vl);
            v_val = __riscv_vmerge_vvm_f32m4(v_val, v_val_slope, v_mask, vl);

            v_val = __riscv_vfmul_vv_f32m4(v_val, v_outputScale, vl);
            v_val = __riscv_vfadd_vv_f32m4(v_val, v_outputZero, vl);

            vint32m4_t v_out = __riscv_vfcvt_x_f_v_i32m4(v_val, vl);
            v_out = __riscv_vmax_vv_i32m4(v_out, v_min, vl);
            v_out = __riscv_vmin_vv_i32m4(v_out, v_max, vl);

            vint16m2_t v_out16 = __riscv_vncvt_x_x_w_i16m2(v_out, vl);
            vint8m1_t v_out8 = __riscv_vncvt_x_x_w_i8m1(v_out16, vl);

            __riscv_vse8_v_i8m1(dstBase, v_out8, vl);
        }
    }
}