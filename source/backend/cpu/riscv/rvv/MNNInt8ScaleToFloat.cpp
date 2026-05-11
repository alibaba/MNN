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

void MNNInt8ScaleToFloat_RVV(float* dst, const int8_t* src, const float* scale, size_t size, const float* zeroPoint,
                             ssize_t quantParamVec) {
    size_t total_elems = size * 4;
    size_t i = 0;

    // Prepare 4-element scale/zero arrays (broadcast scalar or copy vector per quantParamVec)
    float scale4[4] = {scale[0], scale[0], scale[0], scale[0]};
    float zero4[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};
    if (quantParamVec & 1) {
        scale4[0] = scale[0];
        scale4[1] = scale[1];
        scale4[2] = scale[2];
        scale4[3] = scale[3];
    }
    if (quantParamVec >> 1) {
        zero4[0] = zeroPoint[0];
        zero4[1] = zeroPoint[1];
        zero4[2] = zeroPoint[2];
        zero4[3] = zeroPoint[3];
    }

    // Pre-build periodic scale/zero templates outside the loop.
    // vsetvlmax ensures the template covers the full hardware vector length,
    // so the main loop can reuse the first `vl` elements directly without rebuilding each iteration.
    size_t vl_max = __riscv_vsetvlmax_e32m4();

    // iota -> [0,1,2,...,vl_max-1], modulo 4 -> channel index [0,1,2,3,0,1,2,3,...]
    vuint32m4_t v_idx = __riscv_vid_v_u32m4(vl_max);
    vuint32m4_t v_ch = __riscv_vremu_vx_u32m4(v_idx, 4, vl_max);

    // Byte offset = channel_index * 4 (float is 4 bytes)
    vuint32m4_t v_byte_off = __riscv_vsll_vx_u32m4(v_ch, 2, vl_max);

    // Gather scale/zero values by channel index to build full-length periodic templates
    vfloat32m4_t v_scale_tpl = __riscv_vloxei32_v_f32m4(scale4, v_byte_off, vl_max);
    vfloat32m4_t v_zero_tpl = __riscv_vloxei32_v_f32m4(zero4, v_byte_off, vl_max);

    // Main loop: vl <= vl_max, so the first `vl` elements of the template are always aligned
    while (i < total_elems) {
        size_t vl = __riscv_vsetvl_e32m4(total_elems - i);

        vint8m1_t v_src_i8 = __riscv_vle8_v_i8m1(src + i, vl);
        vint32m4_t v_src_i32 = __riscv_vsext_vf4_i32m4(v_src_i8, vl);
        vfloat32m4_t v_src_f32 = __riscv_vfcvt_f_x_v_f32m4(v_src_i32, vl);

        // Compute (src - zeroPoint) * scale using the pre-built periodic templates
        vfloat32m4_t v_sub = __riscv_vfsub_vv_f32m4(v_src_f32, v_zero_tpl, vl);
        vfloat32m4_t v_dst = __riscv_vfmul_vv_f32m4(v_sub, v_scale_tpl, vl);

        __riscv_vse32_v_f32m4(dst + i, v_dst, vl);
        i += vl;
    }
}
