#include <riscv_vector.h>
void MNNDeconvRunForUnitDepthWise(const float* dst, float* src,
                                     const float* weight, size_t fw, size_t fh,
                                     size_t step) {
    const size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t vdst = __riscv_vle32_v_f32m1(dst, vl);

    for (size_t fy = 0; fy < fh; ++fy) {
        for (size_t fx = 0; fx < fw; ++fx) {
            vfloat32m1_t vweight = __riscv_vle32_v_f32m1(weight + fy * fw * 4 + fx * 4, vl);
            size_t offset = fy * step + fx * 4;
            vfloat32m1_t vsrc = __riscv_vle32_v_f32m1(src + offset, vl);
            vsrc = __riscv_vfmacc_vv_f32m1(vsrc, vdst, vweight, vl);
            __riscv_vse32_v_f32m1(src + offset, vsrc, vl);
        }
    }
}
