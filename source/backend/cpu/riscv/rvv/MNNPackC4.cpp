#include <riscv_vector.h>

void MNNPackC4(float *dst, const float *src, size_t area, size_t depth, int *areaOffset) {
    int depthC4 = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain = depth - depthRemain;
    const float *srcOffset = src;
    const float *srcChannel[4];

    for (int z = 0; z < depthC4; ++z) {
        float *dstZ = dst + z * areaOffset[1] * 4;

        for (int y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }

        size_t x = 0;
        size_t vl = __riscv_vsetvl_e32m8(area);

        for (; x + vl <= area; x += vl) {
            float *dstPtr = dstZ + x * 4;
            vfloat32m8_t vec = __riscv_vle32_v_f32m8(srcChannel[0] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 0, 4 * sizeof(float), vec, vl);
            vec = __riscv_vle32_v_f32m8(srcChannel[1] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 1, 4 * sizeof(float), vec, vl);
            vec = __riscv_vle32_v_f32m8(srcChannel[2] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 2, 4 * sizeof(float), vec, vl);
            vec = __riscv_vle32_v_f32m8(srcChannel[3] + x, vl);
            __riscv_vsse32_v_f32m8(dstPtr + 3, 4 * sizeof(float), vec, vl);
        }

        for (; x < area; ++x) {
            float *dstPtr = dstZ + x * 4;
            dstPtr[0] = srcChannel[0][x];
            dstPtr[1] = srcChannel[1][x];
            dstPtr[2] = srcChannel[2][x];
            dstPtr[3] = srcChannel[3][x];
        }

        srcOffset += areaOffset[0] * 4;
    }

    if (remain > 0) {
        float *dstZ = dst + depthC4 * areaOffset[1] * 4;

        for (int y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }

        size_t x = 0;
        size_t vl = __riscv_vsetvl_e32m8(area);

        for (; x + vl <= area; x += vl) {
            float *dstPtr = dstZ + x * 4;

            for (int y = 0; y < remain; ++y) {
                vfloat32m8_t vec = __riscv_vle32_v_f32m8(srcChannel[y] + x, vl);
                __riscv_vsse32_v_f32m8(dstPtr + y, 4 * sizeof(float), vec, vl);
            }

            vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int y = remain; y < 4; ++y) {
                __riscv_vsse32_v_f32m8(dstPtr + y, 4 * sizeof(float), zero, vl);
            }
        }

        for (; x < area; ++x) {
            float *dstPtr = dstZ + x * 4;

            for (int y = 0; y < remain; ++y) {
                dstPtr[y] = srcChannel[y][x];
            }

            for (int y = remain; y < 4; ++y) {
                dstPtr[y] = 0.0f;
            }
        }
    }
}

