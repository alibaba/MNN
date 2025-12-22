#include <riscv_vector.h>

void MNNUnpackC4(float *dst, const float *src, size_t area, size_t depth, int *areaOffset) {
    int depthC4 = depth / 4;        
    int depthRemain = depthC4 * 4;  
    int remain = depth - depthRemain;
    const float *srcOffset = src;

    for (int z = 0; z < depthC4; ++z) {
        float *dstZ[4];

        for (int y = 0; y < 4; ++y) {
            dstZ[y] = dst + (z * 4 + y) * areaOffset[1];
        }

        size_t x = 0;
        size_t vl = __riscv_vsetvl_e32m8(area);

        for (; x + vl <= area; x += vl) {
            vfloat32m8_t vec = __riscv_vlse32_v_f32m8(srcOffset + 0, 4 * sizeof(float), vl);
            __riscv_vse32_v_f32m8(dstZ[0] + x, vec, vl);
            vec = __riscv_vlse32_v_f32m8(srcOffset + 1, 4 * sizeof(float), vl);
            __riscv_vse32_v_f32m8(dstZ[1] + x, vec, vl);
            vec = __riscv_vlse32_v_f32m8(srcOffset + 2, 4 * sizeof(float), vl);
            __riscv_vse32_v_f32m8(dstZ[2] + x, vec, vl);
            vec = __riscv_vlse32_v_f32m8(srcOffset + 3, 4 * sizeof(float), vl);
            __riscv_vse32_v_f32m8(dstZ[3] + x, vec, vl);
            srcOffset += 4 * vl;
        }

        for (; x < area; ++x) {
            dstZ[0][x] = srcOffset[0];
            dstZ[1][x] = srcOffset[1];
            dstZ[2][x] = srcOffset[2];
            dstZ[3][x] = srcOffset[3];
            srcOffset += (areaOffset[0] - area) * 4;
        }
    }

    if (remain > 0) {
        float *dstZ = dst + depthC4 * areaOffset[1] * 4;
        const float *srcBase = srcOffset;

        for (int y = 0; y < remain; ++y) {
            float *dstChannel = dstZ + y * areaOffset[1];
            const float *srcChannel = srcBase + y;

            for (size_t x = 0; x < area; ++x) {
                dstChannel[x] = srcChannel[0];
                srcChannel += 4;
            }
        }
    }
}

