#include <riscv_vector.h>

void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, 
                             const int32_t* position, const float* factor,
                             int8_t* zeroPoint, size_t number) {
    int16_t offset = (int16_t)(*zeroPoint);
    const int pack = 8;
    size_t i = 0;
    
    while (i < number) {
        size_t vl = __riscv_vsetvl_e32m8(number - i);
        vfloat32m8_t vf = __riscv_vle32_v_f32m8(factor + i, vl);
        vint16m4_t vdf = __riscv_vnsra_wx_i16m4(
            __riscv_vfcvt_rtz_x_f_v_i32m8(
                __riscv_vfmul_vf_f32m8(vf, 128.0f, vl), vl), 0, vl);
        vint16m4_t vsf = __riscv_vnsra_wx_i16m4(
            __riscv_vfcvt_rtz_x_f_v_i32m8(
                __riscv_vfmul_vf_f32m8(
                    __riscv_vfrsub_vf_f32m8(vf, 1.0f, vl), 128.0f, vl), vl), 0, vl);
        
        for (int c = 0; c < pack; c++) {
            vuint32m8_t voff = __riscv_vadd_vx_u32m8(
                __riscv_vsll_vx_u32m8(
                    __riscv_vreinterpret_v_i32m8_u32m8(
                        __riscv_vlse32_v_i32m8(position + 2*i, 8, vl)), 3, vl), 
                c, vl);
            
            vint16m4_t va = __riscv_vsub_vx_i16m4(
                __riscv_vsext_vf2_i16m4(
                    __riscv_vluxei32_v_i8m2(src, voff, vl), vl), offset, vl);
            
            vint32m8_t vr = __riscv_vwmul_vv_i32m8(va, vsf, vl);
                        voff = __riscv_vadd_vx_u32m8(
                __riscv_vsll_vx_u32m8(
                    __riscv_vreinterpret_v_i32m8_u32m8(
                        __riscv_vlse32_v_i32m8(position + 2*i + 1, 8, vl)), 3, vl), 
                c, vl);
            
            vint16m4_t vb = __riscv_vsub_vx_i16m4(
                __riscv_vsext_vf2_i16m4(
                    __riscv_vluxei32_v_i8m2(src, voff, vl), vl), offset, vl);
            vr = __riscv_vwmacc_vv_i32m8(vr, vb, vdf, vl);
            __riscv_vsse16_v_i16m4(dst + i * pack + c, 16, 
                __riscv_vnsra_wx_i16m4(vr, 0, vl), vl);
        }
        
        i += vl;
    }
}
