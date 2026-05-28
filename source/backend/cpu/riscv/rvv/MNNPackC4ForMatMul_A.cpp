#include <riscv_vector.h>
#include <algorithm>
#include <stdint.h>
#include <limits>
#include <stddef.h>
#include <string.h>
#include "backend/cpu/CPURuntime.hpp"

static inline int _rvvChannelPack() {
    auto cpuInfo = MNNGetCPUInfo();
    if (nullptr == cpuInfo || cpuInfo->channel_pack <= 0) {
        return 4;
    }
    return cpuInfo->channel_pack;
}

void MNNGetMatMulPackMode_RVV(int* eP, int* lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = _rvvChannelPack();
}

void MNNPackC4ForMatMul_A_RVV(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int pack = _rvvChannelPack();
    const int number = info[0];
    const int eReal = info[1];
    const int eDest = info[2];
    const int offset = info[3];
    const ptrdiff_t sourceStride = static_cast<ptrdiff_t>(pack * offset * sizeof(float));

    for (int n = 0; n < number; ++n) {
        const int e = el[4 * n + 0];
        const int l = el[4 * n + 1];
        const int eOffset = el[4 * n + 2];
        const int lOffset = el[4 * n + 3];
        auto destBase = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int x = 0; x < l; ++x) {
            const int xC = x / pack;
            const int xR = x % pack;
            const float* sourceBase = source + xC * eReal * pack + xR;
            float* destColumn = destBase + x * eDest;
            int y = 0;
            while (y < e) {
                size_t vl = __riscv_vsetvl_e32m8(e - y);
                auto value = __riscv_vlse32_v_f32m8(sourceBase + y * pack * offset, sourceStride, vl);
                __riscv_vse32_v_f32m8(destColumn + y, value, vl);
                y += static_cast<int>(vl);
            }
        }
    }
}

void MNNPackForMatMul_B_RVV(float* dest, const float* source, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    const int pack = _rvvChannelPack();
    const size_t l = kernelsize * ic;
    const size_t hPack = (h + pack - 1) / pack;
    memset(dest, 0, hPack * pack * l * sizeof(float));
    if (transpose) {
        int offset[] = {
            static_cast<int>(l),
            static_cast<int>(l),
        };
        extern void MNNPackCUnit_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
        MNNPackCUnit_RVV(dest, source, l, h, offset);
        return;
    }
    for (size_t y = 0; y < h; ++y) {
        const size_t yC = y / pack;
        const size_t yR = y % pack;
        float* destY = dest + yC * pack * l + yR;
        const float* sourceY = source + y;
        for (size_t x = 0; x < l; ++x) {
            destY[x * pack] = sourceY[x * h];
        }
    }
}

static void _MNNPackedMatMulRemain_RVV(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                       const float* postParameters, const float* bias, int aStride) {
    const int pack = _rvvChannelPack();
    const size_t h = parameter[2];
    const size_t l = parameter[1];
    const size_t cStride = parameter[3] / sizeof(float);
    const size_t bExtraStride = parameter[5] / sizeof(float);
    const size_t bStride = bExtraStride + l * pack;
    const size_t hC = (h + pack - 1) / pack;
    for (size_t y = 0; y < hC; ++y) {
        memset(C + y * cStride, 0, eSize * pack * sizeof(float));
    }

    float minValue = -std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }

    for (size_t yC = 0; yC < hC; ++yC) {
        const size_t yBase = yC * pack;
        const size_t realPack = std::min(static_cast<size_t>(pack), h - yBase);
        const size_t vl = __riscv_vsetvl_e32m8(realPack);
        const float* weightBase = B + yC * bStride;
        for (size_t x = 0; x < eSize; ++x) {
            vfloat32m8_t summer;
            if (nullptr != bias) {
                summer = __riscv_vle32_v_f32m8(bias + yBase, vl);
            } else {
                summer = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            }
            const float* src = A + x;
            for (size_t z = 0; z < l; ++z) {
                auto weight = __riscv_vle32_v_f32m8(weightBase + z * pack, vl);
                summer = __riscv_vfmacc_vf_f32m8(summer, src[z * aStride], weight, vl);
            }
            summer = __riscv_vfmax_vf_f32m8(summer, minValue, vl);
            summer = __riscv_vfmin_vf_f32m8(summer, maxValue, vl);
            __riscv_vse32_v_f32m8(C + yC * cStride + x * pack, summer, vl);
        }
    }
}

void MNNPackedMatMul_RVV(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters,
                         const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_RVV(C, A, B, 16, parameter, postParameters, bias, 16);
}

void MNNPackedMatMulRemain_RVV(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    const int aStride = static_cast<int>(parameter[0] / sizeof(float));
    _MNNPackedMatMulRemain_RVV(C, A, B, eSize, parameter, postParameters, bias, aStride);
}
