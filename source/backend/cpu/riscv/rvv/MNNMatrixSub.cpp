#include <riscv_vector.h>
#include <cstddef>

void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    const size_t total = widthC4 * 4;
    for (size_t y = 0; y < height; ++y) {
        const float* a = A + aStride * y;
        const float* b = B + bStride * y;
        float* c = C + cStride * y;

        for (size_t x = 0; x < total;) {
            const size_t vl = __riscv_vsetvl_e32m8(total - x);
            const vfloat32m8_t va = __riscv_vle32_v_f32m8(a + x, vl);
            const vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + x, vl);
            __riscv_vse32_v_f32m8(c + x, __riscv_vfsub_vv_f32m8(va, vb, vl), vl);
            x += vl;
        }
    }
}
