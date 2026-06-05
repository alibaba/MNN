#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>
#include <sys/types.h>
#include "../../compute/CommonOptFunction.h"

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

    auto lastL = valid ? valid : LP;
    float singlescale = scale[0];
    const size_t vlmax = __riscv_vsetvlmax_e32m4();

    do {
        int step = ALIMIN(EP, realDstCount);
        int scaleOffset = inputBlockQuant ? (step * blockNum) : step;

        for (int k = 0; k < blockNum; ++k) {
            const auto src_x = srcInt8 + k * (step * LP * blockSizeQuad * kernelxy);

            for (int w = 0; w < step; w += 2) {
                int w0 = w;
                int w1 = w + 1;
                bool has_w1 = (w1 < step);

                float scale0, scale1;

                if (oneScale) {
                    scale0 = scale1 = singlescale;
                } else if (inputBlockQuant) {
                    scale0 = scalePtr[w0 + k * step];
                    if (has_w1)
                        scale1 = scalePtr[w1 + k * step];
                } else {
                    scale0 = scalePtr[w0];
                    if (has_w1)
                        scale1 = scalePtr[w1];
                }

                const auto src_y0 = src_x + w0 * LP;
                const auto src_y1 = has_w1 ? (src_x + w1 * LP) : nullptr;

                vint32m4_t vacc0 = __riscv_vmv_v_x_i32m4(0, vlmax);
                vint32m4_t vacc1 = __riscv_vmv_v_x_i32m4(0, vlmax);

                for (int j = 0; j < kernelxy; ++j) {
                    for (int i = 0; i < blockSizeQuad; ++i) {
                        int sumsize = (i == blockSizeQuad - 1) ? lastL : LP;

                        const auto base = j * (blockSizeQuad * step * LP) + i * step * LP;

                        const auto src_z0 = src_y0 + base;
                        const auto src_z1 = has_w1 ? (src_y1 + base) : nullptr;

                        size_t x = 0;

                        while (x < sumsize) {
                            size_t vl = __riscv_vsetvl_e8m1(sumsize - x);

                            // w0
                            vint8m1_t v8_0 = __riscv_vle8_v_i8m1(src_z0 + x, vl);

                            vint16m2_t v16_0 = __riscv_vwcvt_x_x_v_i16m2(v8_0, vl);

                            vint32m4_t v32_0 = __riscv_vwcvt_x_x_v_i32m4(v16_0, vl);

                            vacc0 = __riscv_vadd_vv_i32m4(vacc0, v32_0, vl);

                            // w1
                            if (has_w1) {
                                vint8m1_t v8_1 = __riscv_vle8_v_i8m1(src_z1 + x, vl);

                                vint16m2_t v16_1 = __riscv_vwcvt_x_x_v_i16m2(v8_1, vl);

                                vint32m4_t v32_1 = __riscv_vwcvt_x_x_v_i32m4(v16_1, vl);

                                vacc1 = __riscv_vadd_vv_i32m4(vacc1, v32_1, vl);
                            }

                            x += vl;
                        }
                    }
                }
                // Reduce full accumulator width.

                vint32m1_t vzero = __riscv_vmv_s_x_i32m1(0, vlmax);

                vint32m1_t r0 = __riscv_vredsum_vs_i32m4_i32m1(vacc0, vzero, vlmax);

                int32_t sum0 = __riscv_vmv_x_s_i32m1_i32(r0);

                dest[w0 + k * step] = scale0 * (float)sum0;

                if (has_w1) {
                    vint32m1_t r1 = __riscv_vredsum_vs_i32m4_i32m1(vacc1, vzero, vlmax);

                    int32_t sum1 = __riscv_vmv_x_s_i32m1_i32(r1);

                    dest[w1 + k * step] = scale1 * (float)sum1;
                }
            }
        }

        scalePtr += scaleOffset;
        dest += (step * blockNum);
        realDstCount -= step;
        srcInt8 += col_buffer_unit_size;

    } while (realDstCount > 0);
}
