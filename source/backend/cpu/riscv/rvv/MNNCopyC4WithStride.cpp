#include <riscv_vector.h>

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    ptrdiff_t srcStrideByte = srcStride * sizeof(float);
    ptrdiff_t dstStrideByte = dstStride * sizeof(float);
size_t vl;

    for (size_t i = count; i > 0; i -= vl) {
        vl = __riscv_vsetvl_e32m8(i);
        vfloat32m8_t data = __riscv_vlse32_v_f32m8(source + 0, srcStrideByte, vl);
        __riscv_vsse32_v_f32m8(dest + 0, dstStrideByte, data, vl);
        data = __riscv_vlse32_v_f32m8(source + 1, srcStrideByte, vl);
        __riscv_vsse32_v_f32m8(dest + 1, dstStrideByte, data, vl);
        data = __riscv_vlse32_v_f32m8(source + 2, srcStrideByte, vl);
        __riscv_vsse32_v_f32m8(dest + 2, dstStrideByte, data, vl);
        data = __riscv_vlse32_v_f32m8(source + 3, srcStrideByte, vl);
        __riscv_vsse32_v_f32m8(dest + 3, dstStrideByte, data, vl);
        source += vl * srcStride;
        dest   += vl * dstStride;
    }
}

