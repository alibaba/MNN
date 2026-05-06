#include "../../compute/Int8FunctionsOpt.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#ifndef ALIMIN
#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#endif

void MNNSumByAxisLForMatmul_A_RVV(float* dest, int8_t* source, const float* scale, ssize_t realDstCount,
                                  SumByAxisParams sumParams) {
    int8_t* srcInt8 = source;
    auto scalePtr = scale;
    auto blockNum = sumParams.blockNum;
    auto EP = sumParams.DST_XUNIT;
    auto LP = sumParams.SRC_UNIT;
    auto col_buffer_unit_size = sumParams.unitColBufferSize;
    auto oneScale = sumParams.oneScale;
    auto LU = sumParams.LU;
    auto valid = sumParams.valid;
    auto kernelxy = sumParams.kernelxy;
    auto blockSizeQuad = LU / blockNum;
    auto inputBlockQuant = sumParams.inputBlock;

    auto lastL = (valid != 0) ? valid : LP;
    float singlescale = scale[0];

    do {
        int step = ALIMIN(EP, realDstCount);
        int scaleOffset = inputBlockQuant ? (step * blockNum) : step;

        for (int k = 0; k < blockNum; ++k) {
            const auto src_x = srcInt8 + k * (step * LP * blockSizeQuad * kernelxy);
            for (int w = 0; w < step; ++w) {
                float dequantScale = singlescale;
                if (oneScale == 0 && inputBlockQuant) {
                    dequantScale = scalePtr[w + k * step];
                } else if (oneScale == 0) {
                    dequantScale = scalePtr[w];
                }

                int sumint32 = 0;
                const auto src_y = src_x + w * LP;

                for (int j = 0; j < kernelxy; ++j) {
                    for (int i = 0; i < blockSizeQuad; ++i) {
                        auto sumsize = (i == blockSizeQuad - 1) ? lastL : LP;
                        const auto src_z = src_y + j * (blockSizeQuad * step * LP) + i * step * LP;

                        size_t vl;
                        for (int x = 0; x < sumsize; x += vl) {
                            vl = __riscv_vsetvl_e8m1(sumsize - x);
                            vint8m1_t v_data = __riscv_vle8_v_i8m1(src_z + x, vl);
                            vint32m4_t v_data_32 = __riscv_vwadd_vx_i32m4(__riscv_vwadd_vx_i16m2(v_data, 0, vl), 0, vl);
                            // Reduction
                            vint32m1_t v_acc = __riscv_vmv_s_x_i32m1(0, vl);
                            v_acc = __riscv_vredsum_vs_i32m4_i32m1(v_data_32, v_acc, vl);
                            // sumint32
                            sumint32 += __riscv_vmv_x_s_i32m1_i32(v_acc);
                        }
                    }
                }
                dest[w + k * step] = dequantScale * static_cast<float>(sumint32);
            }
        }
        scalePtr += scaleOffset;
        dest += (step * blockNum);
        realDstCount -= step;
        srcInt8 += col_buffer_unit_size;
    } while (realDstCount > 0);
}