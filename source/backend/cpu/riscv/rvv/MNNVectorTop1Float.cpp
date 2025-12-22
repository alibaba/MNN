#include <riscv_vector.h>
#include <cfloat>

#define UNIT 4

void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    size_t n = inputCountUnit * UNIT;
    float maxV = -FLT_MAX;
    int32_t maxIdx = 0;
    size_t vl;
    
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t data = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m1_t scalar = __riscv_vfmv_s_f_f32m1(maxV, vl);
        vfloat32m1_t result = __riscv_vfredmax_vs_f32m8_f32m1(data, scalar, vl);
        maxV = __riscv_vfmv_f_s_f32m1_f32(result);
        i += vl;
    }
    
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t data = __riscv_vle32_v_f32m8(input + i, vl);
        vbool4_t mask = __riscv_vmfeq_vf_f32m8_b4(data, maxV, vl);
        long first = __riscv_vfirst_m_b4(mask, vl);

        if (first >= 0) {
            maxIdx = i + first;
            break;
        }
        
        i += vl;
    }
    
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}
