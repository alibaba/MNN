#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>

// Sparse im2col blit: dest[x * eDest + yR] = source[(x / 4) * eReal * 4 + y * 4 * offset + x % 4],
// where yR = y % eDest.
//
// The naive access pattern is scattered on both sides, so performance is bound by the
// number of non-coalesced memory transactions, which vector gather/scatter cannot reduce.
// Instead, the vector lanes run along y: for a fixed x the destination is contiguous in
// yR, so every store is unit-stride, while the source is a fixed stride of 4 * offset in
// y. This replaces two scattered transactions per element with one strided load plus one
// contiguous store per vector, which is what actually wins on RVV hardware.
void _MNNPackC4Int8ForMatMul_ASparse_RVV(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info,
                                         const int32_t* el) {
    const int number = info[0];
    const int eReal = info[1];
    const int eDest = info[2];
    const int offset = info[3];
    const ptrdiff_t yStride = static_cast<ptrdiff_t>(offset) * 4;
    const int yChunk = eDest < 16 ? eDest : 16;
    for (int n = 0; n < number; ++n) {
        const int e = el[4 * n + 0];
        const int l = el[4 * n + 1];
        const int eOffset = el[4 * n + 2];
        const int lOffset = el[4 * n + 3];
        int8_t* dest = destOrigin + lOffset * eDest + eOffset;
        const int8_t* source = sourceGroup[n];
        const int groups = (l + 3) / 4;
        for (int y0 = 0; y0 < e; y0 += yChunk) {
            // Keep the whole chunk inside one dest row segment so the stores stay contiguous.
            const int eC = ALIMIN(yChunk, ALIMIN(e - y0, eDest - y0 % eDest));
            const int yR0 = y0 % eDest;
            const size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(eC));
            for (int g = 0; g < groups; ++g) {
                const int8_t* base =
                    source + static_cast<ptrdiff_t>(g) * eReal * 4 + static_cast<ptrdiff_t>(y0) * yStride;
                // The source buffer is allocated for (l + 3) / 4 groups, so the partial
                // tail group reads at most one group of padding, never out of bounds.
                const int valid = ALIMIN(4, l - 4 * g);
                int8_t* dstBase = dest + static_cast<ptrdiff_t>(g) * 4 * eDest + yR0;
                vint8m1_t vr0 = __riscv_vlse8_v_i8m1(base + 0, yStride, vl);
                if (valid > 0) {
                    __riscv_vse8_v_i8m1(dstBase + 0 * eDest, vr0, vl);
                }
                if (valid > 1) {
                    vint8m1_t vr1 = __riscv_vlse8_v_i8m1(base + 1, yStride, vl);
                    __riscv_vse8_v_i8m1(dstBase + 1 * eDest, vr1, vl);
                }
                if (valid > 2) {
                    vint8m1_t vr2 = __riscv_vlse8_v_i8m1(base + 2, yStride, vl);
                    __riscv_vse8_v_i8m1(dstBase + 2 * eDest, vr2, vl);
                }
                if (valid > 3) {
                    vint8m1_t vr3 = __riscv_vlse8_v_i8m1(base + 3, yStride, vl);
                    __riscv_vse8_v_i8m1(dstBase + 3 * eDest, vr3, vl);
                }
            }
        }
    }
}
