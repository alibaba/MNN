#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

void MNNConvRunForUnitint8_t(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                             size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                             size_t dilateX_step, size_t dilateY_step, float* alpha) {
    const size_t vl = __riscv_vsetvl_e32m4(4);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (size_t sz = 0; sz < src_depth_quad; ++sz) {
        const int8_t* srcZ = src + sz * src_depth_step;
        const int8_t* weightZ = weight + sz * weight_z_step;
        for (size_t fy = 0; fy < fh; ++fy) {
            const int8_t* srcY = srcZ + fy * dilateY_step;
            const int8_t* weightY = weightZ + fy * weight_y_step;
            for (size_t fx = 0; fx < fw; ++fx) {
                const int8_t* srcX = srcY + fx * dilateX_step;
                const int8_t* weightX = weightY + 16 * fx;
                for (int i = 0; i < 4; ++i) {
                    const vint8m1_t weight8 = __riscv_vle8_v_i8m1(weightX + 4 * i, vl);
                    const vint32m4_t weight32 = __riscv_vsext_vf4_i32m4(weight8, vl);
                    const vfloat32m4_t weightF = __riscv_vfcvt_f_x_v_f32m4(weight32, vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, static_cast<float>(srcX[i]), weightF, vl);
                }
            }
        }
    }

    const vfloat32m4_t alphaV = __riscv_vle32_v_f32m4(alpha, vl);
    __riscv_vse32_v_f32m4(dst, __riscv_vfmul_vv_f32m4(acc, alphaV, vl), vl);
}
