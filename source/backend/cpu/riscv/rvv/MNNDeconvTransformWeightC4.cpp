#include <riscv_vector.h>

#include "core/Macro.h"

void MNNDeconvTransformWeightC4_RVV(const float* src, float* dst, int outputCount, int srcCount, int area) {
    constexpr int pack = 4;
    int outputC4      = UP_DIV(outputCount, pack);
    int fullOutputC4  = outputCount / pack;
    int tail          = outputCount % pack;
    size_t vl4        = __riscv_vsetvl_e32m1(pack);

    for (int c = 0; c < srcCount; ++c) {
        for (int oc = 0; oc < fullOutputC4; ++oc) {
            for (int a = 0; a < area; ++a) {
                int dstIndex = c * outputC4 * area * pack + oc * area * pack + a * pack;
                const float* srcPtr = src + c * outputCount * area + oc * pack * area + a;
                vfloat32m1_t value  = __riscv_vlse32_v_f32m1(srcPtr, area * sizeof(float), vl4);
                __riscv_vse32_v_f32m1(dst + dstIndex, value, vl4);
            }
        }

        if (tail > 0) {
            int oc    = fullOutputC4;
            size_t vl = __riscv_vsetvl_e32m1(tail);
            for (int a = 0; a < area; ++a) {
                int dstIndex = c * outputC4 * area * pack + oc * area * pack + a * pack;
                const float* srcPtr = src + c * outputCount * area + oc * pack * area + a;
                vfloat32m1_t value  = __riscv_vlse32_v_f32m1(srcPtr, area * sizeof(float), vl);
                __riscv_vse32_v_f32m1(dst + dstIndex, value, vl);
                for (int p = tail; p < pack; ++p) {
                    dst[dstIndex + p] = 0.0f;
                }
            }
        }
    }
}
