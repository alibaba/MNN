#include <riscv_vector.h>

void MNNConvRunForLineDepthwise(
    float* dst, const float* src, const float* weight, 
    size_t width, size_t src_w_setup,
    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, 
    size_t height, size_t srcHStep, size_t dstHStep, 
    const float* bias, const float* parameters) {
    float minV = parameters[0];
    float maxV = parameters[1];
        ptrdiff_t srcByteStride = src_w_setup * sizeof(float);
    ptrdiff_t dstByteStride = 4 * sizeof(float);
    
    for (size_t y = 0; y < height; ++y) {
        const float* srcY = src + y * srcHStep;
        float* dstY = dst + y * dstHStep;
        size_t dx = 0;
        
        while (dx < width) {
            size_t vl = __riscv_vsetvl_e32m8(width - dx);
            
            for (int c = 0; c < 4; ++c) {
                vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(bias[c], vl);
                const float* srcBase = srcY + dx * src_w_setup + c;
                const float* weightPtr = weight + c;
                
                for (size_t fy = 0; fy < fh; ++fy) {
                    const float* srcFy = srcBase + fy * dilateY_step;
                    
                    for (size_t fx = 0; fx < fw; ++fx) {
                        float w = *weightPtr;
                        weightPtr += 4;
                        const float* srcFx = srcFy + fx * dilateX_step;
                        vfloat32m8_t s = __riscv_vlse32_v_f32m8(srcFx, srcByteStride, vl);
                        acc = __riscv_vfmacc_vf_f32m8(acc, w, s, vl);
                    }
                }
                
                acc = __riscv_vfmax_vf_f32m8(acc, minV, vl);
                acc = __riscv_vfmin_vf_f32m8(acc, maxV, vl);
                float* dstAddr = dstY + dx * 4 + c;
                __riscv_vsse32_v_f32m8(dstAddr, dstByteStride, acc, vl);
            }
            
            dx += vl;
        }
    }
}
