#include <riscv_vector.h>

void CPUBilinearSampleC4(const float* src, float* dst, 
                         const int32_t* position, const float* factor,
                         int8_t* zeroPoint, size_t number) {
    const int pack = 4;
    size_t i = 0;
    
    while (i < number) {
        size_t vl = __riscv_vsetvl_e32m8(number - i);
        vfloat32m8_t vf = __riscv_vle32_v_f32m8(factor + i, vl);
        
        for (int c = 0; c < pack; c++) {
            vuint32m8_t voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 2*i, 8, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c * 4, vl);
            vfloat32m8_t vr = __riscv_vluxei32_v_f32m8(src, voff, vl);
            vfloat32m8_t vsf = __riscv_vfrsub_vf_f32m8(vf, 1.0f, vl);
            vr = __riscv_vfmul_vv_f32m8(vr, vsf, vl);
            
            voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 2*i + 1, 8, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c * 4, vl);
            vsf = __riscv_vluxei32_v_f32m8(src, voff, vl);
            vr = __riscv_vfmacc_vv_f32m8(vr, vf, vsf, vl);
            __riscv_vsse32_v_f32m8(dst + i * pack + c, 16, vr, vl);
        }
        
        i += vl;
    }
}
