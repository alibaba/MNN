#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

void MNNConvRunForLineint8_t(float* dst, const int8_t* src, const int8_t* weight, size_t width, size_t src_w_setup,
                             size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                             size_t dilateY_step, float* alpha) {
    const ptrdiff_t srcWidthStrideBytes = static_cast<ptrdiff_t>(src_w_setup);
    const ptrdiff_t dstWidthStrideBytes = 4 * sizeof(float);

    for (size_t dx = 0; dx < width;) {
        const size_t vl = __riscv_vsetvl_e32m4(width - dx);
        vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const int8_t* srcDx = src + src_w_setup * dx;

        for (size_t sz = 0; sz < src_depth_quad; ++sz) {
            const int8_t* srcZ = srcDx + sz * src_depth_step;
            const int8_t* weightZ = weight + sz * fh * fw * 16;
            for (size_t fy = 0; fy < fh; ++fy) {
                const int8_t* srcY = srcZ + fy * dilateY_step;
                const int8_t* weightY = weightZ + fy * fw * 16;
                for (size_t fx = 0; fx < fw; ++fx) {
                    const int8_t* srcX = srcY + fx * dilateX_step;
                    const int8_t* weightX = weightY + 16 * fx;
                    for (int i = 0; i < 4; ++i) {
                        const vint8m1_t src8 = __riscv_vlse8_v_i8m1(srcX + i, srcWidthStrideBytes, vl);
                        const vint32m4_t src32 = __riscv_vsext_vf4_i32m4(src8, vl);
                        const vfloat32m4_t srcF = __riscv_vfcvt_f_x_v_f32m4(src32, vl);
                        acc0 = __riscv_vfmacc_vf_f32m4(acc0, static_cast<float>(weightX[4 * i + 0]), srcF, vl);
                        acc1 = __riscv_vfmacc_vf_f32m4(acc1, static_cast<float>(weightX[4 * i + 1]), srcF, vl);
                        acc2 = __riscv_vfmacc_vf_f32m4(acc2, static_cast<float>(weightX[4 * i + 2]), srcF, vl);
                        acc3 = __riscv_vfmacc_vf_f32m4(acc3, static_cast<float>(weightX[4 * i + 3]), srcF, vl);
                    }
                }
            }
        }

        float* dstDx = dst + dx * 4;
        acc0 = __riscv_vfmul_vf_f32m4(acc0, alpha[0], vl);
        acc1 = __riscv_vfmul_vf_f32m4(acc1, alpha[1], vl);
        acc2 = __riscv_vfmul_vf_f32m4(acc2, alpha[2], vl);
        acc3 = __riscv_vfmul_vf_f32m4(acc3, alpha[3], vl);
        __riscv_vsse32_v_f32m4(dstDx + 0, dstWidthStrideBytes, acc0, vl);
        __riscv_vsse32_v_f32m4(dstDx + 1, dstWidthStrideBytes, acc1, vl);
        __riscv_vsse32_v_f32m4(dstDx + 2, dstWidthStrideBytes, acc2, vl);
        __riscv_vsse32_v_f32m4(dstDx + 3, dstWidthStrideBytes, acc3, vl);
        dx += vl;
    }
}
