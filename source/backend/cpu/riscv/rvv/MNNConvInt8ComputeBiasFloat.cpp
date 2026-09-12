#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

void MNNConvInt8ComputeBiasFloat_RVV(float* dst, const int32_t* bias, const float* weightScale, float scaleRatio,
                                     size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m8(size - offset);
        const vint32m8_t biasValue = __riscv_vle32_v_i32m8(bias + offset, vl);
        vfloat32m8_t value = __riscv_vfcvt_f_x_v_f32m8(biasValue, vl);
        const vfloat32m8_t scale = __riscv_vle32_v_f32m8(weightScale + offset, vl);
        value = __riscv_vfmul_vv_f32m8(value, scale, vl);
        value = __riscv_vfmul_vf_f32m8(value, scaleRatio, vl);
        __riscv_vse32_v_f32m8(dst + offset, value, vl);
        offset += vl;
    }
}
