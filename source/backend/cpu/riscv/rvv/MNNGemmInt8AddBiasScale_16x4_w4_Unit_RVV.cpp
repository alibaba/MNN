#include <riscv_vector.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "../../compute/Int8FunctionsOpt.h"

void MNNGemmInt8AddBiasScale_16x4_w4_Unit_RVV(
    int8_t* dst,
    const int8_t* src,
    const int8_t* weight,
    size_t src_depth_quad,
    size_t dst_step,
    size_t dst_depth_quad,
    const QuanPostTreatParameters* post,
    size_t realCount) {

    const int bytes = 4; // w4 版本通常输出 float
    float fp32min = 0.f, fp32max = 0.f;
    if (post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    // 4-bit 权重步长计算
    const int weight_step_Y = 0.5 * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    const int weight_step_Z = weight_step_Y * src_depth_quad + 4 * 2 * GEMM_INT8_UNIT;

    float* biasPtr = (float*)post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z = dst + dz * dst_step;

        for (int bk = 0; bk < blockNum; ++bk) {
            const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
            const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;
            const auto srcSumPtr = post->srcKernelSum + bk * realCount;
            const auto inputScalePtr = post->inputBias ? post->inputScale + bk * realCount : post->inputScale;

            for (int w = 0; w < realCount; ++w) {
                const auto src_x = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount + w * GEMM_INT8_SRC_UNIT;
                auto dst_x = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x = accumbuff + w * GEMM_INT8_UNIT;

                int32_t acc[4] = {0, 0, 0, 0};

                // ==========================================
                // RVV 核心：4-bit 权重解码与点积累加
                // ==========================================
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = (const uint8_t*)(weight_dz + weight_step_Y * sz);
                    const auto src_z = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                    // 1. 加载 src (int8)
                    size_t vl_src = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
                    vint8m1_t vsrc = __riscv_vle8_v_i8m1(src_z, vl_src);

                    // 2. 解码 4-bit 权重
                    // 每一个 weight_sz 含有 4个输出通道 * 4个输入通道 的 4-bit 权重
                    // 总共 16个权重 = 8字节
                    // 这里我们为了匹配标量逻辑，手动处理每个输出通道 j
                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        // 每个 j 对应的权重占 4 * 4bit = 2 字节
                        uint8_t w_packed[2]; 
                        w_packed[0] = weight_sz[j * 2 + 0];
                        w_packed[1] = weight_sz[j * 2 + 1];

                        // 将 2 字节扩展为 4 字节 (4个 int8)
                        int8_t w_unpacked[4];
                        w_unpacked[0] = (int8_t)(w_packed[0] >> 4);
                        w_unpacked[1] = (int8_t)(w_packed[0] & 0xf);
                        w_unpacked[2] = (int8_t)(w_packed[1] >> 4);
                        w_unpacked[3] = (int8_t)(w_packed[1] & 0xf);

                        // 加载解压后的权重
                        vint8m1_t vw = __riscv_vle8_v_i8m1(w_unpacked, vl_src);

                        // 3. Widening 乘法与归约累加 (int8*int8 -> int16 -> int32)
                        vint16m2_t prod = __riscv_vwmul_vv_i16m2(vsrc, vw, vl_src);
                        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m2_i32m1(
                            prod, 
                            __riscv_vmv_v_x_i32m1(0, 1), 
                            vl_src
                        );
                        acc[j] += __riscv_vmv_x_s_i32m1_i32(v_sum);
                    }
                }

                // ==========================================
                // 后处理：严格按标量逻辑执行
                // ==========================================
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    float value = acc[j] * scale_dz[j] + srcSumPtr[w] * weightBias_dz[j];

                    if (post->inputScale) {
                        value = acc[j] * scale_dz[j] * inputScalePtr[w] + srcSumPtr[w] * weightBias_dz[j];
                    }
                    if (post->inputBias) {
                        auto weightKernelSum = post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;
                        value += (post->inputBias[bk * realCount + w] * weightKernelSum[j]);
                    }

                    if (bk > 0) {
                        value += ((float*)accum_x)[j];
                    }

                    if (bk == blockNum - 1) {
                        if (biasPtr) {
                            value += bias_dz[j];
                        }
                        if (post->fp32minmax) {
                            value = std::min(std::max(fp32min, value), fp32max);
                        }
                        ((float*)dst_x)[j] = value;
                    } else {
                        ((float*)accum_x)[j] = value;
                    }
                }
            }
        }
    }
}
