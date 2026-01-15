#include <climits>
#include <riscv_vector.h>

#define UNIT 4

void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    size_t n = inputCountUnit * UNIT;
    int32_t maxV = INT32_MIN;
    int32_t maxIdx = 0;
    size_t vl;
    
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vint32m8_t data = __riscv_vle32_v_i32m8(input + i, vl);
        vint32m1_t scalar = __riscv_vmv_s_x_i32m1(maxV, vl);
        vint32m1_t result = __riscv_vredmax_vs_i32m8_i32m1(data, scalar, vl);
        maxV = __riscv_vmv_x_s_i32m1_i32(result);
        i += vl;
    }
    
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vint32m8_t data = __riscv_vle32_v_i32m8(input + i, vl);
        vbool4_t mask = __riscv_vmseq_vx_i32m8_b4(data, maxV, vl);
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
