#include <riscv_vector.h>
#include <algorithm>
#include <limits>
#include <stddef.h>
void MNNPackedMatMulRemainFP32_RVV(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                   const float* postParameters, const float* bias, const float* k, const float* b) {
    if (eSize == 0)
        return;

    size_t aStride = parameter[0] / sizeof(float);
    size_t l = parameter[1];
    size_t h = parameter[2];
    size_t cStride = parameter[3] / sizeof(float);
    size_t bExtraStride = parameter[5] / sizeof(float);
    size_t bStride = bExtraStride + l * 4;

    size_t hC4 = UP_DIV(h, 4);

    float minValue = -std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::max();
    if (postParameters != nullptr) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }

    size_t vl = __riscv_vsetvl_e32m4(eSize);
    MNN_ASSERT(vl >= eSize);

    for (size_t y = 0; y < hC4; ++y) {
        float* c_base = C + y * cStride;
        const float* b_base = B + y * bStride;
        const float* bias_y = bias ? bias + 4 * y : nullptr;

        vfloat32m4_t acc0, acc1, acc2, acc3;
        if (bias_y) {
            acc0 = __riscv_vfmv_v_f_f32m4(bias_y[0], vl);
            acc1 = __riscv_vfmv_v_f_f32m4(bias_y[1], vl);
            acc2 = __riscv_vfmv_v_f_f32m4(bias_y[2], vl);
            acc3 = __riscv_vfmv_v_f_f32m4(bias_y[3], vl);
        } else {
            acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }

        for (size_t z = 0; z < l; ++z) {
            vfloat32m4_t a_vec = __riscv_vle32_v_f32m4(A + z * aStride, vl);
            const float* w_ptr = b_base + z * 4;

            acc0 = __riscv_vfmacc_vf_f32m4(acc0, w_ptr[0], a_vec, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, w_ptr[1], a_vec, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, w_ptr[2], a_vec, vl);
            acc3 = __riscv_vfmacc_vf_f32m4(acc3, w_ptr[3], a_vec, vl);
        }

        acc0 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc0, minValue, vl), maxValue, vl);
        acc1 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc1, minValue, vl), maxValue, vl);
        acc2 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc2, minValue, vl), maxValue, vl);
        acc3 = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(acc3, minValue, vl), maxValue, vl);

        ptrdiff_t stride = 4 * sizeof(float);

        __riscv_vsse32_v_f32m4(c_base + 0, stride, acc0, vl);
        __riscv_vsse32_v_f32m4(c_base + 1, stride, acc1, vl);
        __riscv_vsse32_v_f32m4(c_base + 2, stride, acc2, vl);
        __riscv_vsse32_v_f32m4(c_base + 3, stride, acc3, vl);
    }
}
