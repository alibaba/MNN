#include <riscv_vector.h>
#include <stdint.h>

void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];

    for (int n = 0; n < number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int y = 0; y < e; ++y) {
            auto yR = y % eDest;
            auto destY = dest + yR;
            auto sourceY = source + y * 4 * offset;

            int x = 0;
            size_t vl = __riscv_vsetvl_e32m1(4);

            for (; x <= l - 4; x += 4) {
                int xC = x / 4;
                vfloat32m1_t v_src = __riscv_vle32_v_f32m1(sourceY + xC * eReal * 4, vl);
                __riscv_vsse32_v_f32m1(destY + x * eDest, eDest * sizeof(float), v_src, vl);
            }

            for (; x < l; ++x) {
                int xR = x % 4;
                int xC = x / 4;
                destY[x * eDest] = sourceY[xC * eReal * 4 + xR];
            }
        }
    }
}