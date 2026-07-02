#include <riscv_vector.h>

void MNNAccumulateSequenceNumber_RVV(float* dst, const float* src, int size) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e32m8(size - i);
        vfloat32m8_t vs = __riscv_vle32_v_f32m8(src + i, vl);
        acc = __riscv_vfadd_vv_f32m8_tu(acc, vs, vl);
        i += vl;
    }
    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, sun, vlmax);
    *dst = __riscv_vfmv_f_s_f32m1_f32(sum);
}
