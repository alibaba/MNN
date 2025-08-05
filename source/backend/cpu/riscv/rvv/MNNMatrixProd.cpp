#include <riscv_vector.h>

void MNNMatrixProd(float *C, const float *A, const float *B,
                   size_t widthC4, size_t cStride, size_t aStride,
                   size_t bStride, size_t height)
{
    size_t total = widthC4 * 4;
    const int UNROLL_FACTOR = 4;

    for (int y = 0; y < height; ++y)
    {
        const float *a = A + aStride * y;
        const float *b = B + bStride * y;
        float *c = C + cStride * y;
        size_t n = total;

        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m2(n);
            size_t vlX4 = vl * UNROLL_FACTOR;
            if (n < vlX4)
            {
                break;
            }

            vfloat32m2_t va0 = __riscv_vle32_v_f32m2(a + 0 * vl, vl);
            vfloat32m2_t vb0 = __riscv_vle32_v_f32m2(b + 0 * vl, vl);

            vfloat32m2_t va1 = __riscv_vle32_v_f32m2(a + 1 * vl, vl);
            vfloat32m2_t vb1 = __riscv_vle32_v_f32m2(b + 1 * vl, vl);

            vfloat32m2_t va2 = __riscv_vle32_v_f32m2(a + 2 * vl, vl);
            vfloat32m2_t vb2 = __riscv_vle32_v_f32m2(b + 2 * vl, vl);

            vfloat32m2_t va3 = __riscv_vle32_v_f32m2(a + 3 * vl, vl);
            vfloat32m2_t vb3 = __riscv_vle32_v_f32m2(b + 3 * vl, vl);

            vfloat32m2_t vc0 = __riscv_vfmul_vv_f32m2(va0, vb0, vl);
            vfloat32m2_t vc1 = __riscv_vfmul_vv_f32m2(va1, vb1, vl);
            vfloat32m2_t vc2 = __riscv_vfmul_vv_f32m2(va2, vb2, vl);
            vfloat32m2_t vc3 = __riscv_vfmul_vv_f32m2(va3, vb3, vl);

            __riscv_vse32_v_f32m2(c + 0 * vl, vc0, vl);
            __riscv_vse32_v_f32m2(c + 1 * vl, vc1, vl);
            __riscv_vse32_v_f32m2(c + 2 * vl, vc2, vl);
            __riscv_vse32_v_f32m2(c + 3 * vl, vc3, vl);

            a += vlX4;
            b += vlX4;
            c += vlX4;
            n -= vlX4;
        }

        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m2(n);
            vfloat32m2_t va = __riscv_vle32_v_f32m2(a, vl);
            vfloat32m2_t vb = __riscv_vle32_v_f32m2(b, vl);
            vfloat32m2_t vc = __riscv_vfmul_vv_f32m2(va, vb, vl);
            __riscv_vse32_v_f32m2(c, vc, vl);

            a += vl;
            b += vl;
            c += vl;
            n -= vl;
        }
    }
}
