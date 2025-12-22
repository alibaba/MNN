#include <riscv_vector.h>

void CPUBilinearLineC4(float* dst, const float* A, const float* B, 
                       const float* t, int8_t* zeroPoint, size_t number) {
    float tf = *t;
    float sf = 1.0f - tf;
    size_t total = number << 2;
    
    size_t i = 0;
    while (i < total) {
        size_t vl = __riscv_vsetvl_e32m8(total - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(A + i, vl);
        vfloat32m8_t result = __riscv_vfmul_vf_f32m8(v, sf, vl);
        v = __riscv_vle32_v_f32m8(B + i, vl);
        result = __riscv_vfmacc_vf_f32m8(result, tf, v, vl);
        __riscv_vse32_v_f32m8(dst + i, result, vl);
        i += vl;
    }
}
