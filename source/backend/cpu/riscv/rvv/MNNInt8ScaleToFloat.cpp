//
//  MNNInt8ScaleToFloat.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

void MNNInt8ScaleToFloat_RVV(float* dst, const int8_t* src, const float* scale, size_t size, const float* zeroPoint, ssize_t quantParamVec) {
    size_t total_elems = size * 4;
    size_t i = 0;

    size_t vl = __riscv_vsetvl_e32m4(4);
    vfloat32m4_t v_scale = __riscv_vfmv_v_f_f32m4(scale[0], vl);
    vfloat32m4_t v_zero = __riscv_vfmv_v_f_f32m4(zeroPoint[0], vl);

    if (quantParamVec & 1) {
        v_scale = __riscv_vle32_v_f32m4(scale, vl);
    }
    if (quantParamVec >> 1) {
        v_zero = __riscv_vle32_v_f32m4(zeroPoint, vl);
    }

    vl = __riscv_vsetvl_e32m4(total_elems);
    for (i = 0; i < total_elems; i += vl) {
        vl = __riscv_vsetvl_e32m4(total_elems - i);

        vint8m1_t v_src_i8 = __riscv_vle8_v_i8m1(src + i, vl);
        vint32m4_t v_src_i32 = __riscv_vsext_vf4_i32m4(v_src_i8, vl);
        vfloat32m4_t v_src_f32 = __riscv_vfcvt_f_x_v_f32m4(v_src_i32, vl);
        vfloat32m4_t v_sub = __riscv_vfsub_vv_f32m4(v_src_f32, v_zero, vl);
        vfloat32m4_t v_dst = __riscv_vfmul_vv_f32m4(v_sub, v_scale, vl);

        __riscv_vse32_v_f32m4(dst + i, v_dst, vl);
    }
}
