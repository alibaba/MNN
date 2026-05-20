//
//  CommonOptFunction.h
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <cstring>
#include <cmath>
#include "../../compute/Int8FunctionsOpt.h"

 void MNNGemmInt8AddBiasScale_16x4_Unit_RVV(
    int8_t* dst,
    const int8_t* src,
    const int8_t* weight,
    size_t src_depth_quad,
    size_t dst_step,
    size_t dst_depth_quad,
    const QuanPostTreatParameters* post,
    size_t realCount) {

    const int bytes = (post->useInt8 == 1) ? 1 : 4;

    float fp32min = 0.f, fp32max = 0.f;
    if (post->useInt8 == 0 && post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    const int weight_step_Z =
        src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT)
        + 4 * 2 * GEMM_INT8_UNIT;

    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);

    float* biasPtr = (float*)post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z = dst + dz * dst_step;

        for (int bk = 0; bk < blockNum; ++bk) {

            const auto weight_dz =
                weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;

            const float* scale_dz =
                reinterpret_cast<const float*>(
                    weight_dz + src_depth_quad * weight_step_Y);

            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;

            const auto srcSumPtr = post->srcKernelSum + bk * realCount;

            const auto inputScalePtr =
                post->inputBias ? post->inputScale + bk * realCount
                                : post->inputScale;

            for (int w = 0; w < realCount; ++w) {

                const auto src_x =
                    src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount
                    + w * GEMM_INT8_SRC_UNIT;

                auto dst_x = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x = accumbuff + w * GEMM_INT8_UNIT;

                int32_t acc[4] = {0, 0, 0, 0};

                // ===============================
                // RVV 核心：int8 GEMM 累加
                // ===============================
                for (int sz = 0; sz < src_depth_quad; ++sz) {

                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z =
                        src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                    size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);

                    // load src
                    vint8m1_t vsrc =
                        __riscv_vle8_v_i8m1(src_z, vl);

                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {

                        const auto weight_j =
                            weight_sz + j * GEMM_INT8_SRC_UNIT;

                        vint8m1_t vw =
                            __riscv_vle8_v_i8m1(weight_j, vl);

                        // widen mul → int16
                        vint16m2_t prod =
                            __riscv_vwmul_vv_i16m2(vsrc, vw, vl);

                        // reduce → int32
                        vint32m1_t sum =
                            __riscv_vwredsum_vs_i16m2_i32m1(
                                prod,
                                __riscv_vmv_v_x_i32m1(0, 1),
                                vl);

                        acc[j] += __riscv_vmv_x_s_i32m1_i32(sum);
                    }
                }

                // ===============================
                // 后处理（严格按标量逻辑）
                // ===============================
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {

                    float value = acc[j] * scale_dz[j]
                                  + srcSumPtr[w] * weightBias_dz[j];

                    if (post->inputScale) {
                        value = acc[j] * scale_dz[j] * inputScalePtr[w]
                                + srcSumPtr[w] * weightBias_dz[j];
                    }

                    if (post->inputBias) {
                        auto weightKernelSum =
                            post->weightKernelSum
                            + dz * (blockNum * GEMM_INT8_UNIT)
                            + bk * GEMM_INT8_UNIT;

                        value += (post->inputBias[bk * realCount + w]
                                  * weightKernelSum[j]);
                    }

                    if (post->useInt8 == 0) {
                        if (bk > 0) {
                            value += ((float*)accum_x)[j];
                        }

                        if (bk == blockNum - 1) {
                            if (biasPtr) {
                                value += bias_dz[j];
                            }

                            if (post->fp32minmax) {
                                value = std::min(
                                    std::max(fp32min, value),
                                    fp32max);
                            }

                            ((float*)dst_x)[j] = value;
                        } else {
                            ((float*)accum_x)[j] = value;
                        }
                    } else {
                        value += bias_dz[j];

                        value = std::max(value, (float)post->minValue);
                        value = std::min(value, (float)post->maxValue);

                        dst_x[j] = (int8_t)roundf(value);
                    }
                }
            }
        }
    }
}
