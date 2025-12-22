#include <riscv_vector.h>

void MNNCubicLineC4(float* dst, const float* A, const float* B, 
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
    const size_t total = number << 2;
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
        
        __riscv_vse32_v_f32m8(dst + i, acc, vl);
        i += vl;
    }
}
