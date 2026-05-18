#include <riscv_vector.h>
#include <cfloat>
#include <cmath>
#include <cstddef>

#ifndef ALIMIN
#define ALIMIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline void MNNCountMaxMinValue_RVV(const float* src, float* minVal, float* maxVal, size_t size) {
    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;
    size_t offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m8(size - offset);
        const vfloat32m8_t value = __riscv_vle32_v_f32m8(src + offset, vl);
        const vfloat32m1_t minReduce = __riscv_vfredmin_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMin, 1), vl);
        const vfloat32m1_t maxReduce = __riscv_vfredmax_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMax, 1), vl);
        localMin = __riscv_vfmv_f_s_f32m1_f32(minReduce);
        localMax = __riscv_vfmv_f_s_f32m1_f32(maxReduce);
        offset += vl;
    }
    *minVal = localMin;
    *maxVal = localMax;
}

void MNNAsyQuantInfo_FP32_RVV(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax,
                              const float* src, const size_t* info) {
    const size_t blockNum = info[0];
    const size_t plane = info[1];
    const size_t innerSide = info[2];
    const size_t DST_XUNIT = info[3];
    const size_t kernelsize = info[5];
    const size_t blockLU = info[6];
    const size_t stride0 = blockNum * blockLU * plane * innerSide;
    const size_t stride1 = blockLU * plane * innerSide;
    const size_t planeStride = plane * innerSide;

    if (info[7] == 1) {
        float maxval = 0.0f;
        float minval = 0.0f;
        MNNCountMaxMinValue_RVV(src, &minval, &maxval, kernelsize * stride0);
        if (info[8] == 1 && (maxval - minval) > 1e-7f) {
            if (minval > 0.0f) {
                minval = 0.0f;
            } else if (maxval < 0.0f) {
                maxval = 0.0f;
            }
        }
        const float range = maxval - minval;
        if (range <= 1e-7f) {
            scale[0] = 1.0f;
            qscale[0] = 1.0f;
            qbias[0] = -maxval;
            bias[0] = maxval;
        } else {
            qscale[0] = 255.0f / range;
            scale[0] = range / 255.0f;
            qbias[0] = -minval * 255.0f / range - 128.0f;
            bias[0] = minval + 128.0f * range / 255.0f;
        }
        return;
    }

    for (size_t i = 0; i < plane; ++i) {
        for (size_t bk = 0; bk < blockNum; ++bk) {
            const float* base = src + i * innerSide + bk * stride1;
            float localMin = FLT_MAX;
            float localMax = -FLT_MAX;
            for (size_t n = 0; n < kernelsize; ++n) {
                const float* kernelBase = base + n * stride0;
                for (size_t k = 0; k < blockLU; ++k) {
                    const float* row = kernelBase + k * planeStride;
                    size_t j = 0;
                    while (j < innerSide) {
                        const size_t vl = __riscv_vsetvl_e32m8(innerSide - j);
                        const vfloat32m8_t value = __riscv_vle32_v_f32m8(row + j, vl);
                        const vfloat32m1_t minReduce =
                            __riscv_vfredmin_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMin, 1), vl);
                        const vfloat32m1_t maxReduce =
                            __riscv_vfredmax_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMax, 1), vl);
                        localMin = __riscv_vfmv_f_s_f32m1_f32(minReduce);
                        localMax = __riscv_vfmv_f_s_f32m1_f32(maxReduce);
                        j += vl;
                    }
                }
            }
            const size_t qIndex = i + bk * plane;
            dstMin[qIndex] = localMin;
            dstMax[qIndex] = localMax;
        }
    }

    for (size_t i = 0; i < plane; ++i) {
        const size_t step = ALIMIN(DST_XUNIT, plane - (i / DST_XUNIT) * DST_XUNIT);
        const size_t scaleBase = (i / DST_XUNIT) * DST_XUNIT * blockNum + (i % DST_XUNIT);
        for (size_t k = 0; k < blockNum; ++k) {
            const size_t scaleIndex = scaleBase + k * step;
            const size_t qIndex = i + k * plane;
            const float maxval = dstMax[qIndex];
            const float minval = dstMin[qIndex];
            const float range = maxval - minval;
            if (std::fabs(range) < 1e-7f) {
                qscale[qIndex] = 0.0f;
                qbias[qIndex] = 0.0f;
                scale[scaleIndex] = 0.0f;
                bias[scaleIndex] = maxval;
            } else {
                qscale[qIndex] = 255.0f / range;
                qbias[qIndex] = std::round(-minval * 255.0f / range) - 128.0f;
                scale[scaleIndex] = range / 255.0f;
                bias[scaleIndex] = minval + (128.0f / 255.0f) * range;
            }
        }
    }
}
