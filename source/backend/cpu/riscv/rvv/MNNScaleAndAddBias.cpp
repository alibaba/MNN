#include <riscv_vector.h>

void MNNScaleAndAddBias(float *dst, const float *src, const float *bias, const float *alpha, size_t planeNumber, size_t biasNumber) {
    const ptrdiff_t stride = 4 * sizeof(float);

    for (size_t z = 0; z < biasNumber; ++z) {
        float *dstZ = dst + z * planeNumber * 4;
        const float *srcZ = src + z * planeNumber * 4;
        const float *biasZ = bias + 4 * z;
        const float *alphaZ = alpha + 4 * z;
                float b0 = biasZ[0], b1 = biasZ[1], b2 = biasZ[2], b3 = biasZ[3];
        float a0 = alphaZ[0], a1 = alphaZ[1], a2 = alphaZ[2], a3 = alphaZ[3];

        size_t n = planeNumber;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t data = __riscv_vlse32_v_f32m8(srcZ + 0, stride, vl);
            data = __riscv_vfmul_vf_f32m8(data, a0, vl);
            data = __riscv_vfadd_vf_f32m8(data, b0, vl);
            __riscv_vsse32_v_f32m8(dstZ + 0, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(srcZ + 1, stride, vl);
            data = __riscv_vfmul_vf_f32m8(data, a1, vl);
            data = __riscv_vfadd_vf_f32m8(data, b1, vl);
            __riscv_vsse32_v_f32m8(dstZ + 1, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(srcZ + 2, stride, vl);
            data = __riscv_vfmul_vf_f32m8(data, a2, vl);
            data = __riscv_vfadd_vf_f32m8(data, b2, vl);
            __riscv_vsse32_v_f32m8(dstZ + 2, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(srcZ + 3, stride, vl);
            data = __riscv_vfmul_vf_f32m8(data, a3, vl);
            data = __riscv_vfadd_vf_f32m8(data, b3, vl);
            __riscv_vsse32_v_f32m8(dstZ + 3, stride, data, vl);

            srcZ += vl * 4;
            dstZ += vl * 4;
            n -= vl;
        }
    }
}
