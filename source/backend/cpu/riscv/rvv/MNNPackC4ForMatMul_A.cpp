#include <riscv_vector.h>
#include <algorithm>

#define TILE_E_SIZE 1024

void MNNPackC4ForMatMul_A(float *destOrigin, float const **sourceGroup, const int32_t *info, const int32_t *el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];

    for (int n = 0; n < number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto destBase = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];        
        int limit = l / 4 * 4; 
        int x = 0;

        for (; x < limit; x += 4) {
            auto xC = x / 4;
            const float *sourcePtrBase = source + xC * eReal * 4;
            float *destPtrCol0 = destBase + (x + 0) * eDest;
            float *destPtrCol1 = destBase + (x + 1) * eDest;
            float *destPtrCol2 = destBase + (x + 2) * eDest;
            float *destPtrCol3 = destBase + (x + 3) * eDest;

            for (int yBase = 0; yBase < e; yBase += TILE_E_SIZE) {
                int eBlock = std::min(e - yBase, TILE_E_SIZE);

                for (int yOffset = 0; yOffset < eBlock; ) {
                    size_t vl = __riscv_vsetvl_e32m8(eBlock - yOffset);                    
                    const float *sourceYPtr = sourcePtrBase + (yBase + yOffset) * 4 * offset;
                    const size_t sourceStride = 4 * offset * sizeof(float);

                    vfloat32m8_t col0 = __riscv_vlse32_v_f32m8(sourceYPtr + 0, sourceStride, vl);
                    vfloat32m8_t col1 = __riscv_vlse32_v_f32m8(sourceYPtr + 1, sourceStride, vl);
                    vfloat32m8_t col2 = __riscv_vlse32_v_f32m8(sourceYPtr + 2, sourceStride, vl);
                    vfloat32m8_t col3 = __riscv_vlse32_v_f32m8(sourceYPtr + 3, sourceStride, vl);
                    
                    __riscv_vse32_v_f32m8(destPtrCol0 + yBase + yOffset, col0, vl);
                    __riscv_vse32_v_f32m8(destPtrCol1 + yBase + yOffset, col1, vl);
                    __riscv_vse32_v_f32m8(destPtrCol2 + yBase + yOffset, col2, vl);
                    __riscv_vse32_v_f32m8(destPtrCol3 + yBase + yOffset, col3, vl);

                    yOffset += vl;
                }
            }
        }
        
        for (; x < l; ++x) {
            auto xC = x / 4;
            auto xR = x % 4;
            const float* sourcePtrBase = source + xC * eReal * 4 + xR;
            float* destPtrCol = destBase + x * eDest;

            for (int yBase = 0; yBase < e; yBase += TILE_E_SIZE) {
                int eBlock = std::min(e - yBase, TILE_E_SIZE);
                for (int yOffset = 0; yOffset < eBlock; ++yOffset) {
                    int y = yBase + yOffset;
                    destPtrCol[y] = sourcePtrBase[y * 4 * offset];
                }
            }
        }
    }
}
