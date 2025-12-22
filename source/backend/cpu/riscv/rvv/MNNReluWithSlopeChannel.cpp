#include <riscv_vector.h>

void MNNReluWithSlopeChannel(float *dst, const float *src, 
                              const float *slope, size_t sizeQuad, 
                              size_t depthQuad) {
    const ptrdiff_t stride = 4 * sizeof(float);
    
    for (size_t j = 0; j < depthQuad; ++j) {
        const float *srcZ = src + 4 * j * sizeQuad;
        float *dstZ = dst + 4 * j * sizeQuad;
        float s0 = slope[4*j], s1 = slope[4*j + 1];
        float s2 = slope[4*j + 2], s3 = slope[4*j + 3];
        size_t i = 0;
        while (i < sizeQuad) {
            size_t vl = __riscv_vsetvl_e32m8(sizeQuad - i);
            const float *srcBase = srcZ + 4*i;
            float *dstBase = dstZ + 4*i;
            
            vfloat32m8_t v;
            vbool4_t mask;
            
            v = __riscv_vlse32_v_f32m8(srcBase, stride, vl);
            mask = __riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl);
            v = __riscv_vfmul_vf_f32m8_mu(mask, v, v, s0, vl);
            __riscv_vsse32_v_f32m8(dstBase, stride, v, vl);
            
            v = __riscv_vlse32_v_f32m8(srcBase + 1, stride, vl);
            mask = __riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl);
            v = __riscv_vfmul_vf_f32m8_mu(mask, v, v, s1, vl);
            __riscv_vsse32_v_f32m8(dstBase + 1, stride, v, vl);
            
            v = __riscv_vlse32_v_f32m8(srcBase + 2, stride, vl);
            mask = __riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl);
            v = __riscv_vfmul_vf_f32m8_mu(mask, v, v, s2, vl);
            __riscv_vsse32_v_f32m8(dstBase + 2, stride, v, vl);
            
            v = __riscv_vlse32_v_f32m8(srcBase + 3, stride, vl);
            mask = __riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl);
            v = __riscv_vfmul_vf_f32m8_mu(mask, v, v, s3, vl);
            __riscv_vsse32_v_f32m8(dstBase + 3, stride, v, vl);
            
            i += vl;
        }
    }
}
