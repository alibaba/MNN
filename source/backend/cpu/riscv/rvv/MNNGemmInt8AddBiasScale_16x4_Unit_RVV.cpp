//
//  MNNGemmInt8AddBiasScale_16x4_Unit_RVV.cpp
//  MNN
//
//  Created by ISCAS on 2026/04/02.
//  Copyright (c) 2026, ISCAS.
//
#include <riscv_vector.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../compute/Int8FunctionsOpt.h"

void MNNGemmInt8AddBiasScale_16x4_Unit_RVV(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                           size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                           size_t realCount) {
    const int bytes = post->useInt8 == 1 ? 1 : 4;
    const int weightStepY = GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT;
    const int weightStepZ = src_depth_quad * weightStepY + 4 * 2 * GEMM_INT8_UNIT;
    const ptrdiff_t srcStride = GEMM_INT8_SRC_UNIT;
    const ptrdiff_t dstStride = GEMM_INT8_UNIT * sizeof(float);
    float fp32min = 0.0f;
    float fp32max = 0.0f;

    if (post->useInt8 == 0 && post->fp32minmax != nullptr) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    auto biasPtr = post->biasFloat;
    auto accumBuffer = post->accumBuffer;
    auto blockNum = post->blockNum;
    std::vector<int32_t> accCache(realCount);

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dstZ = dst + dz * dst_step;
        auto accumZ = accumBuffer;
        const auto biasDz = biasPtr + dz * GEMM_INT8_UNIT;

        for (int bk = 0; bk < blockNum; ++bk) {
            const auto weightDz = weight + dz * blockNum * weightStepZ + bk * weightStepZ;
            const auto scaleDz = reinterpret_cast<const float*>(weightDz + src_depth_quad * weightStepY);
            const auto weightBiasDz = scaleDz + GEMM_INT8_UNIT;
            const auto srcSumPtr = post->srcKernelSum + bk * realCount;
            const auto inputScalePtr = post->inputBias ? post->inputScale + bk * realCount : post->inputScale;
            const auto inputBiasPtr = post->inputBias ? post->inputBias + bk * realCount : nullptr;
            const auto weightKernelSum =
                post->inputBias ? post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT
                                : nullptr;

            size_t w = 0;
            while (w < realCount) {
                const size_t vl = __riscv_vsetvl_e8m2(realCount - w);
                for (int c = 0; c < GEMM_INT8_UNIT; ++c) {
                    auto acc = __riscv_vmv_v_x_i32m8(0, vl);

                    for (int sz = 0; sz < src_depth_quad; ++sz) {
                        const auto weightSz = weightDz + weightStepY * sz + c * GEMM_INT8_SRC_UNIT;
                        const auto srcSz = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount +
                                           sz * realCount * GEMM_INT8_SRC_UNIT + w * GEMM_INT8_SRC_UNIT;

                        for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                            auto src8 = __riscv_vlse8_v_i8m2(srcSz + i, srcStride, vl);
                            auto src16 = __riscv_vsext_vf2_i16m4(src8, vl);
                            acc = __riscv_vwmacc_vx_i32m8(acc, static_cast<int16_t>(weightSz[i]), src16, vl);
                        }
                    }

                    if (post->useInt8 == 0) {
                        auto value = __riscv_vfcvt_f_x_v_f32m8(acc, vl);
                        value = __riscv_vfmul_vf_f32m8(value, scaleDz[c], vl);
                        if (inputScalePtr != nullptr) {
                            auto inputScaleVec = __riscv_vle32_v_f32m8(inputScalePtr + w, vl);
                            value = __riscv_vfmul_vv_f32m8(value, inputScaleVec, vl);
                        }
                        auto srcSumVec = __riscv_vle32_v_f32m8(srcSumPtr + w, vl);
                        value = __riscv_vfmacc_vf_f32m8(value, weightBiasDz[c], srcSumVec, vl);
                        if (inputBiasPtr != nullptr) {
                            auto inputBiasVec = __riscv_vle32_v_f32m8(inputBiasPtr + w, vl);
                            value = __riscv_vfmacc_vf_f32m8(value, weightKernelSum[c], inputBiasVec, vl);
                        }
                        if (bk > 0) {
                            auto old = __riscv_vlse32_v_f32m8(accumZ + w * GEMM_INT8_UNIT + c, dstStride, vl);
                            value = __riscv_vfadd_vv_f32m8(value, old, vl);
                        }
                        if (bk == blockNum - 1) {
                            if (biasPtr != nullptr) {
                                value = __riscv_vfadd_vf_f32m8(value, biasDz[c], vl);
                            }
                            if (post->fp32minmax != nullptr) {
                                value = __riscv_vfmax_vf_f32m8(value, fp32min, vl);
                                value = __riscv_vfmin_vf_f32m8(value, fp32max, vl);
                            }
                            __riscv_vsse32_v_f32m8(reinterpret_cast<float*>(dstZ + w * GEMM_INT8_UNIT * bytes) + c,
                                                   dstStride, value, vl);
                        } else {
                            __riscv_vsse32_v_f32m8(accumZ + w * GEMM_INT8_UNIT + c, dstStride, value, vl);
                        }
                        continue;
                    }

                    __riscv_vse32_v_i32m8(accCache.data() + w, acc, vl);
                    for (size_t lane = 0; lane < vl; ++lane) {
                        const size_t index = w + lane;
                        auto dstX = dstZ + index * GEMM_INT8_UNIT * bytes;
                        float value = accCache[index] * scaleDz[c] + srcSumPtr[index] * weightBiasDz[c];
                        if (inputScalePtr != nullptr) {
                            value = accCache[index] * scaleDz[c] * inputScalePtr[index] +
                                    srcSumPtr[index] * weightBiasDz[c];
                        }
                        if (inputBiasPtr != nullptr) {
                            value += inputBiasPtr[index] * weightKernelSum[c];
                        }
                        value += biasDz[c];
                        value = std::max(value, static_cast<float>(post->minValue));
                        value = std::min(value, static_cast<float>(post->maxValue));
                        dstX[c] = static_cast<int8_t>(roundf(value));
                    }
                }
                w += vl;
            }
        }
    }
}
