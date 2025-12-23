#include <riscv_vector.h>
#include <algorithm>

void MNNC3ToXYZ(const unsigned char* source, unsigned char* dest, 
                    size_t count, bool bgr) {
    static const int coeffs[] = {
        1689,    1465,    739,
        871,     2929,    296,
        79,      488,     3892
    };
    
    int r0 = 0, r1 = 3, r2 = 6, b0 = 2, b1 = 5, b2 = 8;
    if (bgr) {
        std::swap(r0, b0);
        std::swap(r1, b1);
        std::swap(r2, b2);
    }
    
    int16_t C0 = coeffs[r0], C1 = coeffs[1], C2 = coeffs[b0],
            C3 = coeffs[r1], C4 = coeffs[4], C5 = coeffs[b1],
            C6 = coeffs[r2], C7 = coeffs[7], C8 = coeffs[b2];
    
    size_t i = 0;
    const int32_t rounding = 1 << 11;
    
    while (i < count) {
        size_t vl = __riscv_vsetvl_e8m2(count - i);
        vuint8m2_t vrU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 0, 3, vl);
        vuint8m2_t vgU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 1, 3, vl);
        vuint8m2_t vbU8 = __riscv_vlse8_v_u8m2(source + 3 * i + 2, 3, vl);
        
        vint16m4_t vr = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vrU8, vl));
        vint16m4_t vg = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vgU8, vl));
        vint16m4_t vb = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(vbU8, vl));
        
        vint32m8_t sum = __riscv_vwmul_vx_i32m8(vr, C0, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C1, vg, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C2, vb, vl);
        sum = __riscv_vadd_vx_i32m8(sum, rounding, vl);
        sum = __riscv_vsra_vx_i32m8(sum, 12, vl);
        sum = __riscv_vmax_vx_i32m8(sum, 0, vl);
        sum = __riscv_vmin_vx_i32m8(sum, 255, vl);
        vint16m4_t sum16 = __riscv_vnsra_wx_i16m4(sum, 0, vl);
        vuint8m2_t result = __riscv_vnsrl_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(sum16), 0, vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 0, 3, result, vl);
        
        sum = __riscv_vwmul_vx_i32m8(vr, C3, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C4, vg, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C5, vb, vl);
        sum = __riscv_vadd_vx_i32m8(sum, rounding, vl);
        sum = __riscv_vsra_vx_i32m8(sum, 12, vl);
        sum = __riscv_vmax_vx_i32m8(sum, 0, vl);
        sum = __riscv_vmin_vx_i32m8(sum, 255, vl);
        sum16 = __riscv_vnsra_wx_i16m4(sum, 0, vl);
        result = __riscv_vnsrl_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(sum16), 0, vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 1, 3, result, vl);
        
        sum = __riscv_vwmul_vx_i32m8(vr, C6, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C7, vg, vl);
        sum = __riscv_vwmacc_vx_i32m8(sum, C8, vb, vl);
        sum = __riscv_vadd_vx_i32m8(sum, rounding, vl);
        sum = __riscv_vsra_vx_i32m8(sum, 12, vl);
        sum = __riscv_vmax_vx_i32m8(sum, 0, vl);
        sum = __riscv_vmin_vx_i32m8(sum, 255, vl);
        sum16 = __riscv_vnsra_wx_i16m4(sum, 0, vl);
        result = __riscv_vnsrl_wx_u8m2(__riscv_vreinterpret_v_i16m4_u16m4(sum16), 0, vl);
        __riscv_vsse8_v_u8m2(dest + 3 * i + 2, 3, result, vl);
        
        i += vl;
    }
}
