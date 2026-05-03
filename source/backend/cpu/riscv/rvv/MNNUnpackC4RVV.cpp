//
//  MNNUnpackC4RVV.cpp
//  MNN
//
//  Created by ISCAS on 2025/11/24.
//  Copyright (c) 2025, ISCAS.
//
#include <riscv_vector.h>

void MNNUnpackC4RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const ptrdiff_t srcStrideBytes = 4 * (ptrdiff_t)sizeof(float);
    const size_t depthC4 = (depth + 3) / 4;

    for (size_t z = 0; z < depthC4; ++z) {
        const size_t cBase = z * 4;
        const float* srcZ = src + z * srcAreaStride * 4;
        size_t valid = depth - cBase;
        if (valid > 4) {
            valid = 4;
        }

        for (size_t y = 0; y < valid; ++y) {
            const float* srcChannel = srcZ + y;
            float* dstChannel = dst + (cBase + y) * dstAreaStride;

            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t v = __riscv_vlse32_v_f32m8(srcChannel + 4 * x, srcStrideBytes, vl);
                __riscv_vse32_v_f32m8(dstChannel + x, v, vl);

                x += vl;
            }
        }
    }
}
