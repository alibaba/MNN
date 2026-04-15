//
//  MNNFloat2Int8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <stdint.h>

void MNNFloat2Int8_RVV(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                       ssize_t maxValue, const float* zeroPoint, ssize_t quanParamVec) {
    float scale[4] = {scalep[0], scalep[0], scalep[0], scalep[0]};
    float zero[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};

    if (quanParamVec & 1) {
        scale[0] = scalep[0];
        scale[1] = scalep[1];
        scale[2] = scalep[2];
        scale[3] = scalep[3];
    }
    if (quanParamVec & 2) {
        zero[0] = zeroPoint[0];
        zero[1] = zeroPoint[1];
        zero[2] = zeroPoint[2];
        zero[3] = zeroPoint[3];
    }

    const float minf = (float)minValue;
    const float maxf = (float)maxValue;
    const size_t total = sizeQuad * 4;

    // get vl，create scale/zero cyclic template
    // template by e32m1
    size_t vl_template = __riscv_vsetvlmax_e32m2();

    // iota + modulo operation,create channel index，then gather scale/zero
    //  index: [0,1,2,3,0,1,2,3,...]
    vuint32m2_t v_idx = __riscv_vid_v_u32m2(vl_template);             // [0,1,2,...,vl-1]
    vuint32m2_t v_ch = __riscv_vremu_vx_u32m2(v_idx, 4, vl_template); // [0,1,2,3,0,1,2,3,...]

    // gather：from scale[4] to zero[4] by channel index
    vfloat32m2_t v_scale_tpl =
        __riscv_vloxei32_v_f32m2(scale, __riscv_vsll_vx_u32m2(v_ch, 2, vl_template), vl_template);
    vfloat32m2_t v_zero_tpl = __riscv_vloxei32_v_f32m2(zero, __riscv_vsll_vx_u32m2(v_ch, 2, vl_template), vl_template);

    vfloat32m2_t v_min = __riscv_vfmv_v_f_f32m2(minf, vl_template);
    vfloat32m2_t v_max = __riscv_vfmv_v_f_f32m2(maxf, vl_template);

    // main loop
    size_t i = 0;
    while (i < total) {
        size_t vl = __riscv_vsetvl_e32m2(total - i);

        vfloat32m2_t v_src = __riscv_vle32_v_f32m2(src + i, vl);

        // scale&zero using template（vl <= vl_template，the previous v1 elements is aligned）
        vfloat32m2_t v_mul = __riscv_vfmul_vv_f32m2(v_src, v_scale_tpl, vl);
        vfloat32m2_t v_add = __riscv_vfadd_vv_f32m2(v_mul, v_zero_tpl, vl);
        vfloat32m2_t v_clamp = __riscv_vfmin_vv_f32m2(__riscv_vfmax_vv_f32m2(v_add, v_min, vl), v_max, vl);

        // float→int8
        vint32m2_t v_i32 = __riscv_vfcvt_x_f_v_i32m2(v_clamp, vl);
        vint16m1_t v_i16 = __riscv_vncvt_x_x_w_i16m1(v_i32, vl);
        vint8mf2_t v_i8 = __riscv_vncvt_x_x_w_i8mf2(v_i16, vl);

        __riscv_vse8_v_i8mf2(dst + i, v_i8, vl);

        i += vl;
    }
}