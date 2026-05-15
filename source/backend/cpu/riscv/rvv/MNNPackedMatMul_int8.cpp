#include <riscv_vector.h>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <algorithm>

#define UP_DIV(x, y) (((x) + (y) - 1) / (y))

void _MNNPackedMatMulRemain_int8(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter,
                                 const float* postParameters, const float* bias, int aStride, const float* k,
                                 const float* b) {
    const int8_t* B = reinterpret_cast<const int8_t*>(fB);
    size_t h = parameter[2];
    size_t l = parameter[1];
    size_t cStride = parameter[3] / sizeof(float);
    int32_t bExtraStride = static_cast<int32_t>(parameter[5]);
    size_t bStride = bExtraStride + 4 * l;
    size_t hC4 = UP_DIV(h, 4);

    float minValue = -std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::max();
    if (postParameters != nullptr) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    int blockId = parameter[6];
    const size_t VL = 4;

    vfloat32m1_t min_v = __riscv_vfmv_v_f_f32m1(minValue, VL);
    vfloat32m1_t max_v = __riscv_vfmv_v_f_f32m1(maxValue, VL);

    for (size_t x = 0; x < eSize; x++) {
        float* dst = C + 4 * x;
        const float* src = A + x;

        for (size_t y = 0; y < hC4; y++) {
            float* dstY = dst + y * cStride;
            const int8_t* weight = B + y * bStride;
            const float* alpha = k + y * 4;
            const float* qbias = b + y * 4;

            vfloat32m1_t sum_v = __riscv_vfmv_v_f_f32m1(0.0f, VL);
            if (blockId > 0) {
                sum_v = __riscv_vle32_v_f32m1(dstY, VL);
            }

            if (bias != nullptr && postParameters != nullptr) {
                vfloat32m1_t bias_v = __riscv_vle32_v_f32m1(bias + 4 * y, VL);
                sum_v = __riscv_vfadd_vv_f32m1(sum_v, bias_v, VL);
            }

            for (size_t z = 0; z < l; z++) {
                float a_val = src[z * aStride];
                vfloat32m1_t a_v = __riscv_vfmv_v_f_f32m1(a_val, VL);

                // 完全和标量逻辑一致，无精度误差
                float w_buf[4] = {
                    (float)weight[z * 4 + 0] * alpha[0] + qbias[0], (float)weight[z * 4 + 1] * alpha[1] + qbias[1],
                    (float)weight[z * 4 + 2] * alpha[2] + qbias[2], (float)weight[z * 4 + 3] * alpha[3] + qbias[3]};

                // 正确加载向量（GCC15.1 标准）
                vfloat32m1_t w_v = __riscv_vle32_v_f32m1(w_buf, VL);

                // 正确 FMA 指令（带 VL）
                sum_v = __riscv_vfmadd_vv_f32m1(w_v, a_v, sum_v, VL);
            }

            sum_v = __riscv_vfmin_vv_f32m1(sum_v, max_v, VL);
            sum_v = __riscv_vfmax_vv_f32m1(sum_v, min_v, VL);
            __riscv_vse32_v_f32m1(dstY, sum_v, VL);
        }
    }
}

void MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_int8(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}

void MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                const float* postParameters, const float* bias, const float* k, const float* b) {
    const int aStride = static_cast<int>(parameter[0] / sizeof(float));
    _MNNPackedMatMulRemain_int8(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
