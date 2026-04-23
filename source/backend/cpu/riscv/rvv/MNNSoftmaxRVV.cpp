//
//  MNNSoftmaxRVV.cpp
//  MNN
//
//  Created by ISCAS on 2025/12/05.
//  Copyright (c) 2025, ISCAS.
//
#include <riscv_vector.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include "../../compute/CommonOptFunction.h"

void MNNSoftmaxRVV(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale,
                   int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask) {
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float xLimit = 87.0f;
    const int reduceSizeInner = pack > 1 ? pack : reduceSize;
    const int reduceSizeOuter = pack > 1 ? UP_DIV(reduceSize, pack) : 1;
    const int stride0 = pack > 1 ? outside * reduceSizeInner : reduceSizeInner;

    for (int k = 0; k < outside; ++k) {
        if (mask && kvSeqOffset > k + validOffset) {
            if (updateScale != nullptr) {
                updateScale[k] = 1.0f;
            }
            for (int j = 0; j < reduceSizeOuter; ++j) {
                std::memset(softmaxDst + j * stride0 + k * reduceSizeInner, 0, reduceSizeInner * sizeof(float));
            }
            continue;
        }

        int validReduceSize = reduceSize;
        if (mask) {
            validReduceSize = k + validOffset + 1 - kvSeqOffset;
            if (validReduceSize > reduceSize) {
                validReduceSize = reduceSize;
            }
            if (validReduceSize < 0) {
                validReduceSize = 0;
            }
        }

        const float oldMax = runningMax != nullptr ? runningMax[k] : -FLT_MAX;
        float newMax = -FLT_MAX;

        for (int j = 0; j < reduceSizeOuter; ++j) {
            int validCount = validReduceSize - j * reduceSizeInner;
            if (validCount <= 0) {
                break;
            }
            if (validCount > reduceSizeInner) {
                validCount = reduceSizeInner;
            }
            const float* srcPtr = softmaxSrc + j * stride0 + k * reduceSizeInner;
            vfloat32m1_t maxVec = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);
            int count = validCount;
            while (count > 0) {
                size_t vl = __riscv_vsetvl_e32m8(count);
                vfloat32m8_t value = __riscv_vle32_v_f32m8(srcPtr, vl);
                maxVec = __riscv_vfredmax_vs_f32m8_f32m1(value, maxVec, vl);
                srcPtr += vl;
                count -= static_cast<int>(vl);
            }
            const float blockMax = __riscv_vfmv_f_s_f32m1_f32(maxVec);
            if (newMax < blockMax) {
                newMax = blockMax;
            }
        }

        const float finalMax = oldMax > newMax ? oldMax : newMax;
        float sum = 0.0f;

        for (int j = 0; j < reduceSizeOuter; ++j) {
            int validCount = validReduceSize - j * reduceSizeInner;
            if (validCount <= 0) {
                break;
            }
            if (validCount > reduceSizeInner) {
                validCount = reduceSizeInner;
            }
            const float* srcPtr = softmaxSrc + j * stride0 + k * reduceSizeInner;
            float* dstPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
            vfloat32m1_t sumVec = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            int count = validCount;
            while (count > 0) {
                size_t vl = __riscv_vsetvl_e32m8(count);
                vfloat32m8_t value = __riscv_vle32_v_f32m8(srcPtr, vl);
                value = __riscv_vfsub_vf_f32m8(value, finalMax, vl);
                value = __riscv_vfmax_vf_f32m8(value, -xLimit, vl);
                value = __riscv_vfmin_vf_f32m8(value, xLimit, vl);

                vint32m8_t exponent = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfdiv_vf_f32m8(value, ln2, vl), vl);
                vfloat32m8_t basic = __riscv_vreinterpret_v_i32m8_f32m8(
                    __riscv_vsll_vx_i32m8(__riscv_vadd_vx_i32m8(exponent, 127, vl), 23, vl));

                vfloat32m8_t remain = __riscv_vfnmsub_vf_f32m8(__riscv_vfcvt_f_x_v_f32m8(exponent, vl), ln2, value, vl);
                vfloat32m8_t poly = __riscv_vfmv_v_f_f32m8(1.0f / 120.0f, vl);
                poly = __riscv_vfmul_vv_f32m8(poly, remain, vl);
                poly = __riscv_vfadd_vf_f32m8(poly, 1.0f / 24.0f, vl);
                poly = __riscv_vfmul_vv_f32m8(poly, remain, vl);
                poly = __riscv_vfadd_vf_f32m8(poly, 1.0f / 6.0f, vl);
                poly = __riscv_vfmul_vv_f32m8(poly, remain, vl);
                poly = __riscv_vfadd_vf_f32m8(poly, 0.5f, vl);
                poly = __riscv_vfmul_vv_f32m8(poly, remain, vl);
                poly = __riscv_vfadd_vf_f32m8(poly, 1.0f, vl);
                poly = __riscv_vfmul_vv_f32m8(poly, remain, vl);
                poly = __riscv_vfadd_vf_f32m8(poly, 1.0f, vl);

                value = __riscv_vfmul_vv_f32m8(basic, poly, vl);
                __riscv_vse32_v_f32m8(dstPtr, value, vl);
                sumVec = __riscv_vfredosum_vs_f32m8_f32m1(value, sumVec, vl);

                srcPtr += vl;
                dstPtr += vl;
                count -= static_cast<int>(vl);
            }
            sum += __riscv_vfmv_f_s_f32m1_f32(sumVec);
        }

        if (runningMax != nullptr && runningSum != nullptr && updateScale != nullptr) {
            const float scaleForSum = expf(oldMax - finalMax);
            runningSum[k] = runningSum[k] * scaleForSum + sum;
            runningMax[k] = finalMax;
            updateScale[k] = scaleForSum;
        } else {
            if (runningMax != nullptr && runningSum != nullptr) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            const float scale = 1.0f / (sum + 1e-20f);
            for (int j = 0; j < reduceSizeOuter; ++j) {
                int validCount = validReduceSize - j * reduceSizeInner;
                if (validCount <= 0) {
                    break;
                }
                if (validCount > reduceSizeInner) {
                    validCount = reduceSizeInner;
                }
                float* dstPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                int count = validCount;
                while (count > 0) {
                    size_t vl = __riscv_vsetvl_e32m8(count);
                    vfloat32m8_t value = __riscv_vle32_v_f32m8(dstPtr, vl);
                    value = __riscv_vfmul_vf_f32m8(value, scale, vl);
                    __riscv_vse32_v_f32m8(dstPtr, value, vl);
                    dstPtr += vl;
                    count -= static_cast<int>(vl);
                }
            }
        }

        if (pack > 1) {
            for (int j = 0; j < reduceSizeOuter; ++j) {
                int validCount = validReduceSize - j * reduceSizeInner;
                float* dstPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                if (validCount <= 0) {
                    std::memset(dstPtr, 0, reduceSizeInner * sizeof(float));
                    continue;
                }
                if (validCount > reduceSizeInner) {
                    validCount = reduceSizeInner;
                }
                if (validCount < reduceSizeInner) {
                    std::memset(dstPtr + validCount, 0, (reduceSizeInner - validCount) * sizeof(float));
                }
            }
        } else {
            std::memset(softmaxDst + k * reduceSizeInner + validReduceSize, 0,
                        (reduceSize - validReduceSize) * sizeof(float));
        }
    }
}
