#include <riscv_vector.h>

void MNNStrassenMergeCFunction(float *c11, float *c12, float *c21, float *c22, 
                                float *xAddr, size_t cStride, size_t eSub, size_t hSub) {
    for (int y = 0; y < hSub; ++y) {
        float *c11Y = c11 + y * cStride;
        float *c12Y = c12 + y * cStride;
        float *c22Y = c22 + y * cStride;
        float *c21Y = c21 + y * cStride;
        float *xY = xAddr + y * eSub * 4;        
        size_t totalElements = eSub * 4;
        size_t p = 0;

        while (p < totalElements) {
            size_t vl = __riscv_vsetvl_e32m8(totalElements - p);
            vfloat32m8_t t = __riscv_vle32_v_f32m8(xY + p, vl);
            vfloat32m8_t tmp = __riscv_vle32_v_f32m8(c12Y + p, vl);
            t = __riscv_vfadd_vv_f32m8(t, tmp, vl);
            vfloat32m8_t c22v = __riscv_vle32_v_f32m8(c22Y + p, vl);

            tmp = __riscv_vle32_v_f32m8(c11Y + p, vl);
            tmp = __riscv_vfadd_vv_f32m8(tmp, c22v, vl);
            tmp = __riscv_vfadd_vv_f32m8(tmp, t, vl);
            __riscv_vse32_v_f32m8(c12Y + p, tmp, vl);
            
            tmp = __riscv_vle32_v_f32m8(c21Y + p, vl);
            tmp = __riscv_vfadd_vv_f32m8(t, tmp, vl);
            __riscv_vse32_v_f32m8(c21Y + p, tmp, vl);
            
            c22v = __riscv_vfadd_vv_f32m8(c22v, tmp, vl);
            __riscv_vse32_v_f32m8(c22Y + p, c22v, vl);
            
            p += vl;
        }
    }
}
