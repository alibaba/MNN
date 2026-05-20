#include <riscv_vector.h>
#include <cstddef>

void MNNAccumulateSequenceNumber_RVV(float* dst, const float* src, int size) {
    if (size <= 0) {
        *dst = 0.0f;
        return;
    }

    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t sumVector = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    size_t remaining = static_cast<size_t>(size);
    while (remaining >= vlmax) {
        const vfloat32m8_t value = __riscv_vle32_v_f32m8(src, vlmax);
        sumVector = __riscv_vfadd_vv_f32m8(sumVector, value, vlmax);
        src += vlmax;
        remaining -= vlmax;
    }

    vfloat32m1_t sumScalar = __riscv_vfredusum_vs_f32m8_f32m1(sumVector, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax);
    float sum = __riscv_vfmv_f_s_f32m1_f32(sumScalar);

    if (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m8(remaining);
        const vfloat32m8_t value = __riscv_vle32_v_f32m8(src, vl);
        sumScalar = __riscv_vfredusum_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(sum, 1), vl);
        sum = __riscv_vfmv_f_s_f32m1_f32(sumScalar);
    }
    *dst = sum;
}
