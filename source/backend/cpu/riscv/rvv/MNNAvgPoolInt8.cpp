//
//  MNNAvgPoolInt8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <stdint.h>
#include <limits.h>

void MNNAvgPoolInt8_RVV(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely,
                        size_t stridesx, ssize_t paddingx, ssize_t factor) {
    const size_t vl = __riscv_vsetvl_e8m1(16);
    const size_t pack = 16;

    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;

    vint32m4_t v_factor = __riscv_vmv_v_x_i32m4(factor, vl);

    for (size_t ox = 0; ox < outputWidth; ++ox) {
        vint32m4_t vec_sum = __riscv_vmv_v_x_i32m4(0, vl);

        for (size_t y = 0; y < kernely; ++y) {
            for (size_t x = 0; x < kernelx; ++x) {
                const int8_t* inputPtr = srcPtr + pack * (x + inputWidth * y);
                vint8m1_t vec_input = __riscv_vle8_v_i8m1(inputPtr, vl);
                vint32m4_t vec_input_ext = __riscv_vsext_vf4_i32m4(vec_input, vl);
                vec_sum = __riscv_vadd_vv_i32m4(vec_sum, vec_input_ext, vl);
            }
        }

        vint32m4_t v_mul = __riscv_vmul_vv_i32m4(vec_sum, v_factor, vl);
        vint32m4_t v_shr = __riscv_vsra_vx_i32m4(v_mul, 24, vl);
        vint16m2_t v_temp = __riscv_vncvt_x_x_w_i16m2(v_shr, vl);
        vint8m1_t v_result = __riscv_vncvt_x_x_w_i8m1(v_temp, vl);

        __riscv_vse8_v_i8m1(dstPtr, v_result, vl);

        dstPtr += pack;
        srcPtr += pack * stridesx;
    }
}
