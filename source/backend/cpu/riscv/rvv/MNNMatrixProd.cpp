#include <riscv_vector.h>

void MNNMatrixProd0(float* C, const float* A, const float* B,
                       size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    size_t total = widthC4 * 4;

    for (int y = 0; y < height; ++y) {
        const float* a_ptr = A + aStride * y;
        const float* b_ptr = B + bStride * y;
        float* c_ptr = C + cStride * y;
        size_t n = total;

        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t va = __riscv_vle32_v_f32m8(a_ptr, vl);
            vfloat32m8_t vb = __riscv_vle32_v_f32m8(b_ptr, vl);
            vfloat32m8_t vc = __riscv_vfmul_vv_f32m8(va, vb, vl);
            __riscv_vse32_v_f32m8(c_ptr, vc, vl);

            a_ptr += vl;
            b_ptr += vl;
            c_ptr += vl;
            n -= vl;
        }
    }
}
