#include <riscv_vector.h>

void MNNCubicSampleC16(const int8_t* src, float* dst, 
                        int32_t* position, const float* factor,
                        int8_t* zeroPoint, size_t number) {
    const int pack = 16;
    int8_t zp = *zeroPoint;
    size_t i = 0;
    
    while (i < number) {
        size_t vl = __riscv_vsetvl_e32m8(number - i);
        vfloat32m8_t vt = __riscv_vle32_v_f32m8(factor + i, vl);
        
        for (int c = 0; c < pack; c++) {
            vuint32m8_t voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 4*i + 0, 16, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c, vl);
            
            vint8m2_t vtmp_i8 = __riscv_vluxei32_v_i8m2(src, voff, vl);
            vint16m4_t vtmp_i16 = __riscv_vwsub_vx_i16m4(vtmp_i8, zp, vl);
            vfloat32m8_t vtmp = __riscv_vfcvt_f_x_v_f32m8(
                __riscv_vwcvt_x_x_v_i32m8(vtmp_i16, vl), vl);
            
            vfloat32m8_t va = __riscv_vfmul_vf_f32m8(vtmp, -0.75f, vl);
            vfloat32m8_t vb = __riscv_vfmul_vf_f32m8(vtmp, 1.5f, vl);
            vfloat32m8_t vc = vtmp;
            
            voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 4*i + 1, 16, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c, vl);
            
            vtmp_i8 = __riscv_vluxei32_v_i8m2(src, voff, vl);
            vtmp_i16 = __riscv_vwsub_vx_i16m4(vtmp_i8, zp, vl);
            vfloat32m8_t vB = __riscv_vfcvt_f_x_v_f32m8(
                __riscv_vwcvt_x_x_v_i32m8(vtmp_i16, vl), vl);
            
            va = __riscv_vfmacc_vf_f32m8(va, 1.25f, vB, vl);
            vb = __riscv_vfmacc_vf_f32m8(vb, -2.25f, vB, vl);
            
            voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 4*i + 2, 16, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c, vl);
            
            vtmp_i8 = __riscv_vluxei32_v_i8m2(src, voff, vl);
            vtmp_i16 = __riscv_vwsub_vx_i16m4(vtmp_i8, zp, vl);
            vtmp = __riscv_vfcvt_f_x_v_f32m8(
                __riscv_vwcvt_x_x_v_i32m8(vtmp_i16, vl), vl);
            
            va = __riscv_vfmacc_vf_f32m8(va, -1.25f, vtmp, vl);
            vb = __riscv_vfmacc_vf_f32m8(vb, 1.5f, vtmp, vl);
            vc = __riscv_vfsub_vv_f32m8(vtmp, vc, vl);
            vc = __riscv_vfmul_vf_f32m8(vc, 0.75f, vl);
            
            voff = __riscv_vsll_vx_u32m8(
                __riscv_vreinterpret_v_i32m8_u32m8(
                    __riscv_vlse32_v_i32m8(position + 4*i + 3, 16, vl)), 4, vl);
            voff = __riscv_vadd_vx_u32m8(voff, c, vl);
            
            vtmp_i8 = __riscv_vluxei32_v_i8m2(src, voff, vl);
            vtmp_i16 = __riscv_vwsub_vx_i16m4(vtmp_i8, zp, vl);
            vtmp = __riscv_vfcvt_f_x_v_f32m8(
                __riscv_vwcvt_x_x_v_i32m8(vtmp_i16, vl), vl);
            
            va = __riscv_vfmacc_vf_f32m8(va, 0.75f, vtmp, vl);
            vb = __riscv_vfmacc_vf_f32m8(vb, -0.75f, vtmp, vl);
            
            va = __riscv_vfmadd_vv_f32m8(va, vt, vb, vl);
            va = __riscv_vfmadd_vv_f32m8(va, vt, vc, vl);
            va = __riscv_vfmadd_vv_f32m8(va, vt, vB, vl);
            
            __riscv_vsse32_v_f32m8(dst + i * pack + c, pack * sizeof(float), va, vl);
        }
        
        i += vl;
    }
}
