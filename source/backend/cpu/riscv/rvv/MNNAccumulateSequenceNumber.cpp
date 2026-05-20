#include <riscv_vector.h>
void MNNAccumulateSequenceNumber_RVV(float* dst, const float* src, int size) {
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    int n = size;
    for (; n > 0;) {
        vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t v_src = __riscv_vle32_v_f32m1(src, vl);
        v_sum = __riscv_vfadd_vv_f32m1(v_sum, v_src, vl);
        n -= vl;
        src += vl;
    }
    vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v_total = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
    float sum = __riscv_vfmv_f_s_f32m1_f32(v_total);
    *dst = sum;
}
