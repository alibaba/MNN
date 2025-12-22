#include <riscv_vector.h>
#include <cfloat>

#define UNIT 4

void MNNMinFloat(float *input, float *minBuffer, int32_t inputCountUnit) {
    const float init = FLT_MAX;
    for (int j = 0; j < UNIT; ++j) {
        float local = init;
        size_t i = 0;

        while (i < (size_t)inputCountUnit) {
            size_t vl = __riscv_vsetvl_e32m8(inputCountUnit - i);
            float *p0 = input + (i * UNIT * 2) + j * 2;
            float *p1 = p0 + 1;
            vfloat32m8_t v0 = __riscv_vlse32_v_f32m8(p0, UNIT * 2 * sizeof(float), vl);
            vfloat32m8_t v1 = __riscv_vlse32_v_f32m8(p1, UNIT * 2 * sizeof(float), vl);
            vfloat32m8_t vmin = __riscv_vfmin_vv_f32m8(v0, v1, vl);
            vfloat32m1_t vred = __riscv_vfredmin_vs_f32m8_f32m1(vmin, __riscv_vfmv_s_f_f32m1(local, 1), vl);
            local = __riscv_vfmv_f_s_f32m1_f32(vred);
            i += vl;
        }
        minBuffer[j] = local;
    }
}
