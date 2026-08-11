#include <riscv_vector.h>

void MNNPackC2(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const ptrdiff_t dstStrideBytes = 2 * (ptrdiff_t)sizeof(float);
    const size_t depthC2 = (depth + 1) / 2;

    for (size_t z = 0; z < depthC2; ++z) {
        const size_t cBase = z * 2;
        float* dstZ = dst + z * dstAreaStride * 2;
        size_t valid = depth - cBase;
        if (valid > 2)
            valid = 2;

        for (size_t y = 0; y < valid; ++y) {
            const float* srcChannel = src + (cBase + y) * srcAreaStride;

            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(srcChannel + x, vl);
                __riscv_vsse32_v_f32m8(dstZ + 2 * x + y, dstStrideBytes, v, vl);
                x += vl;
            }
        }

        for (size_t y = valid; y < 2; ++y) {
            size_t x = 0;
            while (x < area) {
                const size_t vl = __riscv_vsetvl_e32m8(area - x);
                vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                __riscv_vsse32_v_f32m8(dstZ + 2 * x + y, dstStrideBytes, zero, vl);
                x += vl;
            }
        }
    }
}
