#include <riscv_vector.h>

void MNNC3ToHSV(const unsigned char* source, unsigned char* dest, 
                    size_t count, bool bgr, bool full) {
    const float hrange = full ? 256.0f : 180.0f;
    const float hscale = hrange / 6.0f;
    const int hrangeI = full ? 256 : 180;
    size_t i = 0;
    
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m2(count - i);
        
        vuint8m2_t vrU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 0, 3, vl);
        vuint8m2_t vgU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 1, 3, vl);
        vuint8m2_t vbU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 2, 3, vl);
        if (bgr) {
            vuint8m2_t tmp = vrU8;
            vrU8 = vbU8;
            vbU8 = tmp;
        }
        
        vuint8m2_t vmaxU8 = __riscv_vmaxu_vv_u8m2(
            __riscv_vmaxu_vv_u8m2(vrU8, vgU8, vl), vbU8, vl);
        vuint8m2_t vminU8 = __riscv_vminu_vv_u8m2(
            __riscv_vminu_vv_u8m2(vrU8, vgU8, vl), vbU8, vl);
        vuint8m2_t vdiffU8 = __riscv_vsub_vv_u8m2(vmaxU8, vminU8, vl);
        
        vint16m4_t vr = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vrU8, vl));
        vint16m4_t vg = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vgU8, vl));
        vint16m4_t vb = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vbU8, vl));
        vint16m4_t vdiff = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vdiffU8, vl));
        
        vbool4_t maskR = __riscv_vmseq_vv_u8m2_b4(vmaxU8, vrU8, vl);
        vbool4_t maskG = __riscv_vmseq_vv_u8m2_b4(vmaxU8, vgU8, vl);
        vbool4_t maskDiffZero = __riscv_vmseq_vx_u8m2_b4(vdiffU8, 0, vl);
        vbool4_t maskVZero = __riscv_vmseq_vx_u8m2_b4(vmaxU8, 0, vl);
        
        vint16m4_t sum16 = __riscv_vadd_vv_i16m4(
            __riscv_vsub_vv_i16m4(vr, vg, vl),
            __riscv_vsll_vx_i16m4(vdiff, 2, vl), vl);
        vint16m4_t temp16 = __riscv_vadd_vv_i16m4(
            __riscv_vsub_vv_i16m4(vb, vr, vl),
            __riscv_vsll_vx_i16m4(vdiff, 1, vl), vl);
        sum16 = __riscv_vmerge_vvm_i16m4(sum16, temp16, maskG, vl);
        sum16 = __riscv_vmerge_vvm_i16m4(sum16, __riscv_vsub_vv_i16m4(vg, vb, vl), maskR, vl);
        
        vfloat32m8_t sumF = __riscv_vfcvt_f_x_v_f32m8(__riscv_vsext_vf2_i32m8(sum16, vl), vl);
        vfloat32m8_t diffF = __riscv_vfcvt_f_xu_v_f32m8(__riscv_vzext_vf4_u32m8(vdiffU8, vl), vl);
        
        sumF = __riscv_vfmul_vf_f32m8(sumF, hscale, vl);
        sumF = __riscv_vfdiv_vv_f32m8(sumF, __riscv_vfmax_vf_f32m8(diffF, 1.0f, vl), vl);
        sumF = __riscv_vfmerge_vfm_f32m8(sumF, 0.0f, maskDiffZero, vl);
        
        sumF = __riscv_vfadd_vf_f32m8(sumF, 0.5f, vl);
        vint32m8_t sum = __riscv_vfcvt_rtz_x_f_v_i32m8(sumF, vl);
        
        vbool4_t isNegFrac  = __riscv_vmflt_vf_f32m8_b4(sumF, 0.0f, vl);
        vfloat32m8_t sumBack = __riscv_vfcvt_f_x_v_f32m8(sum, vl);
        vbool4_t notInt = __riscv_vmfne_vv_f32m8_b4(sumF, sumBack, vl);
        vbool4_t floorAdjust = __riscv_vmand_mm_b4(isNegFrac , notInt, vl);
        sum = __riscv_vsub_vx_i32m8_mu(floorAdjust, sum, sum, 1, vl);
        
        vbool4_t hNeg = __riscv_vmslt_vx_i32m8_b4(sum, 0, vl);
        sum = __riscv_vadd_vx_i32m8_mu(hNeg, sum, sum, hrangeI, vl);
        
        sum = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(sum, 0, vl), hrangeI - 1, vl);
        sum16 = __riscv_vnsra_wx_i16m4(sum, 0, vl);
        vuint8m2_t result = __riscv_vnsrl_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(sum16), 0, vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 0, 3, result, vl);
        
        sumF = __riscv_vfcvt_f_xu_v_f32m8(__riscv_vzext_vf4_u32m8(vmaxU8, vl), vl);
        sumF = __riscv_vfdiv_vv_f32m8(
            __riscv_vfmul_vf_f32m8(diffF, 255.0f, vl),
            __riscv_vfmax_vf_f32m8(sumF, 1.0f, vl), vl);
        sumF = __riscv_vfmerge_vfm_f32m8(sumF, 0.0f, maskVZero, vl);
        
        sumF = __riscv_vfadd_vf_f32m8(sumF, 0.5f, vl);
        sum = __riscv_vfcvt_rtz_x_f_v_i32m8(sumF, vl);
        
        sum = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(sum, 0, vl), 255, vl);
        sum16 = __riscv_vnsra_wx_i16m4(sum, 0, vl);
        result = __riscv_vnsrl_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(sum16), 0, vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 1, 3, result, vl);
        
        __riscv_vsse8_v_u8m2(dest + 3 * i + 2, 3, vmaxU8, vl);
        
        i += vl;
    }
}
