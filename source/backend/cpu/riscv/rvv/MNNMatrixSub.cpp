#include <riscv_vector.h>

void MNNMatrixSub(float *C, const float *A, const float *B,
                     size_t widthC4, size_t cStride, size_t aStride,
                     size_t bStride, size_t height) {
    size_t total = widthC4 * 4;
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;

        size_t n = total;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
            vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);
            vfloat32m8_t vc = __riscv_vfsub_vv_f32m8(va, vb, vl);
            __riscv_vse32_v_f32m8(c, vc, vl);

            a += vl;
            b += vl;
            c += vl;
            n -= vl;
        }
    }
}
