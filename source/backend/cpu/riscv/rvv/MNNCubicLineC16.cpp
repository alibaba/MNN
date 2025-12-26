#include <riscv_vector.h>

void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, 
                     const float* C, const float* D, float* t, 
                     int8_t* zeroPoint, size_t number, 
                     ssize_t minValue, ssize_t maxValue) {
    const float f = *t;
    const float t2 = f * f, t3 = t2 * f;
    const float b0 = 1.0f - 2.25f * t2 + 1.25f * t3;
    const float t1 = 1.0f - f, t1_2 = t1 * t1;
    const float c0 = 1.0f - 2.25f * t1_2 + 1.25f * t1_2 * t1;
    const float ta = 1.0f + f, ta2 = ta * ta;
    const float a0 = 3.0f - 6.0f * ta + 3.75f * ta2 - 0.75f * ta2 * ta;
    const float td = 2.0f - f, td2 = td * td;
    const float d0 = 3.0f - 6.0f * td + 3.75f * td2 - 0.75f * td2 * td;    
    const int offset = *zeroPoint;
    const int minVal = (int)minValue;
    const int maxVal = (int)maxValue;
    const size_t total = number << 4;
    size_t i = 0;
    
    while (i < total) {
        size_t vl = __riscv_vsetvl_e32m8(total - i);
        vfloat32m8_t v, acc;
        
        v   = __riscv_vle32_v_f32m8(A + i, vl);
        acc = __riscv_vfmul_vf_f32m8(v, a0, vl);
        
        v   = __riscv_vle32_v_f32m8(B + i, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, b0, v, vl);
        
        v   = __riscv_vle32_v_f32m8(C + i, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, c0, v, vl);
        
        v   = __riscv_vle32_v_f32m8(D + i, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, d0, v, vl);
        
        vfloat32m8_t half = __riscv_vfmv_v_f_f32m8(0.5f, vl);
        vfloat32m8_t signHalf = __riscv_vfsgnj_vv_f32m8(half, acc, vl);
        acc = __riscv_vfadd_vv_f32m8(acc, signHalf, vl);
        
        vint32m8_t vint = __riscv_vfcvt_rtz_x_f_v_i32m8(acc, vl);
        vint = __riscv_vadd_vx_i32m8(vint, offset, vl);
        vint = __riscv_vmax_vx_i32m8(vint, minVal, vl);
        vint = __riscv_vmin_vx_i32m8(vint, maxVal, vl);
        
        vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vint, vl);
        vint8m2_t  vi8  = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
        __riscv_vse8_v_i8m2(dst + i, vi8, vl);
        
        i += vl;
    }
}
