//
//  MNNMaxPoolInt8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <stdint.h>
#include <limits.h>

void MNNMaxPoolInt8_RVV(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx) {
    const size_t vl = __riscv_vsetvl_e8m1(16);
    const size_t pack = 16;

    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;

    for (size_t ox = 0; ox < outputWidth; ++ox) {
        vint8m1_t vec_max = __riscv_vmv_v_x_i8m1(INT8_MIN, vl);

        for (size_t y = 0; y < kernely; ++y) {
            for (size_t x = 0; x < kernelx; ++x) {
                const int8_t* inputPtr = srcPtr + pack * (x + inputWidth * y);
                vint8m1_t vec_input = __riscv_vle8_v_i8m1(inputPtr, vl);
                vec_max = __riscv_vmax_vv_i8m1(vec_max, vec_input, vl);
            }
        }

        __riscv_vse8_v_i8m1(dstPtr, vec_max, vl);

        dstPtr += pack;
        srcPtr += pack * stridesx;
    }
}
