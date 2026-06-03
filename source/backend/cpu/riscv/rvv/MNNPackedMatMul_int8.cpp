#include <riscv_vector.h>
#include <limits>
#include <cstddef>
#include <cstdint>

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

static void MNNPackedMatMulRemain_int8_RVVImpl(float* C, const float* A, const float* fB, size_t eSize,
                                               const size_t* parameter, const float* postParameters, const float* bias,
                                               int aStride, const float* k, const float* b) {
    const int8_t* B = reinterpret_cast<const int8_t*>(fB);
    const size_t h = parameter[2];
    const size_t l = parameter[1];
    const size_t cStride = parameter[3] / sizeof(float);
    const size_t bStride = parameter[5] + 4 * l;
    const size_t hC4 = UP_DIV(h, 4);

    float minValue = -std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::max();
    if (postParameters != nullptr) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const int blockId = static_cast<int>(parameter[6]);
    const ptrdiff_t cXStrideBytes = 4 * sizeof(float);

    for (size_t y = 0; y < hC4; ++y) {
        const int8_t* weight = B + y * bStride;
        const float* alpha = k + y * 4;
        const float* qbias = b + y * 4;
        for (size_t x = 0; x < eSize;) {
            const size_t vl = __riscv_vsetvl_e32m4(eSize - x);
            float* dst = C + y * cStride + 4 * x;
            vfloat32m4_t sum0;
            vfloat32m4_t sum1;
            vfloat32m4_t sum2;
            vfloat32m4_t sum3;
            if (blockId > 0) {
                sum0 = __riscv_vlse32_v_f32m4(dst + 0, cXStrideBytes, vl);
                sum1 = __riscv_vlse32_v_f32m4(dst + 1, cXStrideBytes, vl);
                sum2 = __riscv_vlse32_v_f32m4(dst + 2, cXStrideBytes, vl);
                sum3 = __riscv_vlse32_v_f32m4(dst + 3, cXStrideBytes, vl);
            } else {
                sum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                sum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                sum2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                sum3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            }

            if (bias != nullptr && postParameters != nullptr) {
                const float* biasY = bias + y * 4;
                sum0 = __riscv_vfadd_vf_f32m4(sum0, biasY[0], vl);
                sum1 = __riscv_vfadd_vf_f32m4(sum1, biasY[1], vl);
                sum2 = __riscv_vfadd_vf_f32m4(sum2, biasY[2], vl);
                sum3 = __riscv_vfadd_vf_f32m4(sum3, biasY[3], vl);
            }

            for (size_t z = 0; z < l; ++z) {
                const int8_t* weightZ = weight + 4 * z;
                const vfloat32m4_t a = __riscv_vle32_v_f32m4(A + z * aStride + x, vl);
                const float w0 = static_cast<float>(weightZ[0]) * alpha[0] + qbias[0];
                const float w1 = static_cast<float>(weightZ[1]) * alpha[1] + qbias[1];
                const float w2 = static_cast<float>(weightZ[2]) * alpha[2] + qbias[2];
                const float w3 = static_cast<float>(weightZ[3]) * alpha[3] + qbias[3];
                sum0 = __riscv_vfmacc_vf_f32m4(sum0, w0, a, vl);
                sum1 = __riscv_vfmacc_vf_f32m4(sum1, w1, a, vl);
                sum2 = __riscv_vfmacc_vf_f32m4(sum2, w2, a, vl);
                sum3 = __riscv_vfmacc_vf_f32m4(sum3, w3, a, vl);
            }

            sum0 = __riscv_vfmax_vf_f32m4(__riscv_vfmin_vf_f32m4(sum0, maxValue, vl), minValue, vl);
            sum1 = __riscv_vfmax_vf_f32m4(__riscv_vfmin_vf_f32m4(sum1, maxValue, vl), minValue, vl);
            sum2 = __riscv_vfmax_vf_f32m4(__riscv_vfmin_vf_f32m4(sum2, maxValue, vl), minValue, vl);
            sum3 = __riscv_vfmax_vf_f32m4(__riscv_vfmin_vf_f32m4(sum3, maxValue, vl), minValue, vl);
            __riscv_vsse32_v_f32m4(dst + 0, cXStrideBytes, sum0, vl);
            __riscv_vsse32_v_f32m4(dst + 1, cXStrideBytes, sum1, vl);
            __riscv_vsse32_v_f32m4(dst + 2, cXStrideBytes, sum2, vl);
            __riscv_vsse32_v_f32m4(dst + 3, cXStrideBytes, sum3, vl);
            x += vl;
        }
    }
}

void MNNPackedMatMul_int8_RVV(float* C, const float* A, const float* B, const size_t* parameter,
                              const float* postParameters, const float* bias, const float* k, const float* b) {
    MNNPackedMatMulRemain_int8_RVVImpl(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}

void MNNPackedMatMulRemain_int8_RVV(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, const float* k, const float* b) {
    const int aStride = static_cast<int>(parameter[0] / sizeof(float));
    MNNPackedMatMulRemain_int8_RVVImpl(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
