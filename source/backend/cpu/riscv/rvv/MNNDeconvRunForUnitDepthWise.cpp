#include <riscv_vector.h>

void MNNDeconvRunForUnitDepthWise(
    const float* dst, float* src, const float* weight, 
    size_t fw, size_t fh,
    size_t weightY_step, size_t dilateX_step, size_t dilateY_step) {    
    const ptrdiff_t wStride = 4 * sizeof(float);
    const ptrdiff_t sStride = dilateX_step * sizeof(float);
    float d0 = dst[0], d1 = dst[1], d2 = dst[2], d3 = dst[3];
    
    for (size_t fy = 0; fy < fh; ++fy) {
        float* srcY = src + fy * dilateY_step;
        const float* weightY = weight + fy * weightY_step;
        
        size_t fx = 0;
        while (fx < fw) {
            size_t vl = __riscv_vsetvl_e32m8(fw - fx);
            
            vfloat32m8_t w = __riscv_vlse32_v_f32m8(weightY + 0 + fx * 4, wStride, vl);
            vfloat32m8_t s = __riscv_vlse32_v_f32m8(srcY + 0 + fx * dilateX_step, sStride, vl);
            s = __riscv_vfmacc_vf_f32m8(s, d0, w, vl);
            __riscv_vsse32_v_f32m8(srcY + 0 + fx * dilateX_step, sStride, s, vl);
            
            w = __riscv_vlse32_v_f32m8(weightY + 1 + fx * 4, wStride, vl);
            s = __riscv_vlse32_v_f32m8(srcY + 1 + fx * dilateX_step, sStride, vl);
            s = __riscv_vfmacc_vf_f32m8(s, d1, w, vl);
            __riscv_vsse32_v_f32m8(srcY + 1 + fx * dilateX_step, sStride, s, vl);
            
            w = __riscv_vlse32_v_f32m8(weightY + 2 + fx * 4, wStride, vl);
            s = __riscv_vlse32_v_f32m8(srcY + 2 + fx * dilateX_step, sStride, vl);
            s = __riscv_vfmacc_vf_f32m8(s, d2, w, vl);
            __riscv_vsse32_v_f32m8(srcY + 2 + fx * dilateX_step, sStride, s, vl);
            
            w = __riscv_vlse32_v_f32m8(weightY + 3 + fx * 4, wStride, vl);
            s = __riscv_vlse32_v_f32m8(srcY + 3 + fx * dilateX_step, sStride, vl);
            s = __riscv_vfmacc_vf_f32m8(s, d3, w, vl);
            __riscv_vsse32_v_f32m8(srcY + 3 + fx * dilateX_step, sStride, s, vl);
            
            fx += vl;
        }
    }
}
