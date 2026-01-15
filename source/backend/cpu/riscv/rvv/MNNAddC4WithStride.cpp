#include <riscv_vector.h>

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    ptrdiff_t srcStrideByte = srcStride * sizeof(float);
    ptrdiff_t dstStrideByte = dstStride * sizeof(float);
    size_t vl;

    for (size_t i = count; i > 0; i -= vl) {
        vl = __riscv_vsetvl_e32m8(i);
        vfloat32m8_t vs = __riscv_vlse32_v_f32m8(source + 0, srcStrideByte, vl);
        vfloat32m8_t vd = __riscv_vlse32_v_f32m8(dest + 0, dstStrideByte, vl);
        vd = __riscv_vfadd_vv_f32m8(vd, vs, vl);
        __riscv_vsse32_v_f32m8(dest + 0, dstStrideByte, vd, vl);
        vs = __riscv_vlse32_v_f32m8(source + 1, srcStrideByte, vl);
        vd = __riscv_vlse32_v_f32m8(dest + 1, dstStrideByte, vl);
        vd = __riscv_vfadd_vv_f32m8(vd, vs, vl);
        __riscv_vsse32_v_f32m8(dest + 1, dstStrideByte, vd, vl);
        vs = __riscv_vlse32_v_f32m8(source + 2, srcStrideByte, vl);
        vd = __riscv_vlse32_v_f32m8(dest + 2, dstStrideByte, vl);
        vd = __riscv_vfadd_vv_f32m8(vd, vs, vl);
        __riscv_vsse32_v_f32m8(dest + 2, dstStrideByte, vd, vl);
        vs = __riscv_vlse32_v_f32m8(source + 3, srcStrideByte, vl);
        vd = __riscv_vlse32_v_f32m8(dest + 3, dstStrideByte, vl);
        vd = __riscv_vfadd_vv_f32m8(vd, vs, vl);
        __riscv_vsse32_v_f32m8(dest + 3, dstStrideByte, vd, vl);
        source += vl * srcStride;
        dest   += vl * dstStride;
    }
}
