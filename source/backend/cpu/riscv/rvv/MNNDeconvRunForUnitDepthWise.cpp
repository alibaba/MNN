#include <riscv_vector.h>
#include <cstddef>

void MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    const ptrdiff_t weightStrideBytes = 4 * sizeof(float);
    const ptrdiff_t srcStrideBytes = static_cast<ptrdiff_t>(dilateX_step) * sizeof(float);

    for (size_t fy = 0; fy < fh; ++fy) {
        float* srcY = src + fy * dilateY_step;
        const float* weightY = weight + fy * weight_y_step;
        for (size_t fx = 0; fx < fw;) {
            const size_t vl = __riscv_vsetvl_e32m8(fw - fx);

            for (int c = 0; c < 4; ++c) {
                vfloat32m8_t srcValue = __riscv_vlse32_v_f32m8(srcY + fx * dilateX_step + c, srcStrideBytes, vl);
                const vfloat32m8_t weightValue = __riscv_vlse32_v_f32m8(weightY + fx * 4 + c, weightStrideBytes, vl);
                srcValue = __riscv_vfmacc_vf_f32m8(srcValue, dst[c], weightValue, vl);
                __riscv_vsse32_v_f32m8(srcY + fx * dilateX_step + c, srcStrideBytes, srcValue, vl);
            }
            fx += vl;
        }
    }
}
