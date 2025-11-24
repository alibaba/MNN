#include <riscv_vector.h>

void MNNPackC2(float *dst, const float *src, size_t area, size_t depth, int *areaOffset) {
    int depthC2 = depth / 2;
    int depthRemain = depthC2 * 2;
    int remain = depth - depthRemain;
    const float *srcOffset = src;
    const float *srcChannel[2];

    for (int z = 0; z < depthC2; ++z) {
        float *dstZ = dst + z * areaOffset[1] * 2;

        for (int y = 0; y < 2; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }

        size_t x = 0;
        size_t vl = __riscv_vsetvl_e32m8(area);

        for (; x + vl <= area; x += vl) {
            float *dstPtr = dstZ + x * 2;
            vfloat32m8_t vec = __riscv_vle32_v_f32m8(srcChannel[0] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 0, 2 * sizeof(float), vec, vl);
            vec = __riscv_vle32_v_f32m8(srcChannel[1] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 1, 2 * sizeof(float), vec, vl);
        }

        for (; x < area; ++x) {
            float *dstPtr = dstZ + x * 2;
            dstPtr[0] = srcChannel[0][x];
            dstPtr[1] = srcChannel[1][x];
        }

        srcOffset += areaOffset[0] * 2;
    }

    if (remain > 0) {
        float *dstZ = dst + depthC2 * areaOffset[1] * 2;

        for (int y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }

        size_t x = 0;
        size_t vl = __riscv_vsetvl_e32m8(area);

        for (; x + vl <= area; x += vl) {
            float *dstPtr = dstZ + x * 2;

            for (int y = 0; y < remain; ++y) {
                vfloat32m8_t vec = __riscv_vle32_v_f32m8(srcChannel[y] + x, vl);
                __riscv_vsse32_v_f32m8(dstPtr + y, 2 * sizeof(float), vec, vl);
            }

            vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int y = remain; y < 2; ++y) {
                __riscv_vsse32_v_f32m8(dstPtr + y, 2 * sizeof(float), zero, vl);
            }
        }

        for (; x < area; ++x) {
            float *dstPtr = dstZ + x * 2;

            for (int y = 0; y < remain; ++y) {
                dstPtr[y] = srcChannel[y][x];
            }

            for (int y = remain; y < 2; ++y) {
                dstPtr[y] = 0.0f;
            }
        }
    }
}

