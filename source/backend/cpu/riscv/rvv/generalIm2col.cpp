#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <algorithm>

void generalIm2col_RVV(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int LP,
                       int pack) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];

    int eReal_pack = eReal * pack;
    int eDest_LP = eDest * LP;

    for (int n = 0; n < number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];

        auto dest = destOrigin + eOffset * LP;
        auto source = sourceGroup[n];

        for (int y = 0; y < e; ++y) {
            auto yR = y % eDest;
            float* dst_y = dest + yR * LP;
            const float* src_y = source + y * pack * offset;

            int x_total = lOffset;
            int xC = 0;

            for (int x = 0; x < l; x += pack) {
                int current_pack = std::min(pack, l - x);

                int xOut = x_total / LP;
                int xIn = x_total % LP;

                const float* s_ptr = src_y + xC * eReal_pack;
                float* d_ptr = dst_y + xOut * eDest_LP + xIn;

                // Note: This RVV path is currently a placeholder.
                // Vectorization along this dimension is limited when LP=1 and pack=4,
                // causing it to fall through to the scalar loop below.
                // For now, we use a scalar loop to handle this case.
                // In the future, we can explore more efficient vectorization strategies.
                if (xIn + current_pack <= LP) {
                    size_t vl = __riscv_vsetvl_e32m1(current_pack);
                    vfloat32m1_t v_val = __riscv_vle32_v_f32m1(s_ptr, vl);
                    __riscv_vse32_v_f32m1(d_ptr, v_val, vl);
                } else {
                    for (int i = 0; i < current_pack; ++i) {
                        int temp_x = x_total + i;
                        dst_y[(temp_x / LP) * eDest_LP + (temp_x % LP)] = s_ptr[i];
                    }
                }
                xC++;
                x_total += current_pack;
            }
        }
    }
}
