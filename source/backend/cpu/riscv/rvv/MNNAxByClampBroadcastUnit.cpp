#include <riscv_vector.h>

void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    float beta = parameters[1];
    float minF = parameters[2];
    float maxF = parameters[3];
    const ptrdiff_t stride = 4 * sizeof(float);

    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 4 * y;
        auto c = C + cStride * y;
        float b0Beta = b[0] * beta;
        float b1Beta = b[1] * beta;
        float b2Beta = b[2] * beta;
        float b3Beta = b[3] * beta;
        size_t w = width;

        while (w > 0) {
            size_t vl = __riscv_vsetvl_e32m8(w);

            vfloat32m8_t data = __riscv_vlse32_v_f32m8(a + 0, stride, vl);
                        data = __riscv_vfadd_vf_f32m8(data, b0Beta, vl);
                        data = __riscv_vfmax_vf_f32m8(data, minF, vl);
            data = __riscv_vfmin_vf_f32m8(data, maxF, vl);
                        __riscv_vsse32_v_f32m8(c + 0, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(a + 1, stride, vl);
            data = __riscv_vfadd_vf_f32m8(data, b1Beta, vl);
            data = __riscv_vfmax_vf_f32m8(data, minF, vl);
            data = __riscv_vfmin_vf_f32m8(data, maxF, vl);
            __riscv_vsse32_v_f32m8(c + 1, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(a + 2, stride, vl);
            data = __riscv_vfadd_vf_f32m8(data, b2Beta, vl);
            data = __riscv_vfmax_vf_f32m8(data, minF, vl);
            data = __riscv_vfmin_vf_f32m8(data, maxF, vl);
            __riscv_vsse32_v_f32m8(c + 2, stride, data, vl);

            data = __riscv_vlse32_v_f32m8(a + 3, stride, vl);
            data = __riscv_vfadd_vf_f32m8(data, b3Beta, vl);
            data = __riscv_vfmax_vf_f32m8(data, minF, vl);
            data = __riscv_vfmin_vf_f32m8(data, maxF, vl);
            __riscv_vsse32_v_f32m8(c + 3, stride, data, vl);

            a += 4 * vl;
            c += 4 * vl;
            w -= vl;
        }
    }
}

