//
//  MNNPackC4RVV.cpp
//  MNN
//
//  Created by ISCAS on 2025/11/24.
//  Copyright (c) 2025, ISCAS.
//
#include <riscv_vector.h>

void MNNPackC4RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const ptrdiff_t dstStrideBytes = 4 * (ptrdiff_t)sizeof(float);
    const size_t depthC4 = (depth + 3) / 4;

    for (size_t z = 0; z < depthC4; ++z) {
        const size_t cBase = z * 4;
        const float* srcZ = src + cBase * srcAreaStride;
        float* dstZ = dst + z * dstAreaStride * 4;
        size_t valid = depth - cBase;
        if (valid > 4) {
            valid = 4;
        }

        for (size_t y = 0; y < valid; ++y) {
            const float* srcChannel = srcZ + y * srcAreaStride;

            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(srcChannel + x, vl);
                __riscv_vsse32_v_f32m8(dstZ + 4 * x + y, dstStrideBytes, v, vl);

                x += vl;
            }
        }

        for (size_t y = valid; y < 4; ++y) {
            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                __riscv_vsse32_v_f32m8(dstZ + 4 * x + y, dstStrideBytes, zero, vl);

                x += vl;
            }
        }
    }
}
