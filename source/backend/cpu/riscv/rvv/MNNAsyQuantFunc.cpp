#include <riscv_vector.h>

void MNNAsyQuantFunc_RVV(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
    auto blockNum = info[0];
    auto EP = info[1];
    auto LP = info[2];
    auto kernelsize = info[5];
    auto blockLU = info[6];

    auto stride0 = blockNum * blockLU * EP * LP;
    auto stride1 = blockLU * EP * LP;

    const int int8_min = -128;
    const int int8_max = 127;
    const size_t RV_RNU = 0x4;

    const int EP_TILE = 4;

    for (int n = 0; n < kernelsize; ++n) {
        for (int bk = 0; bk < blockNum; ++bk) {
            const float* scalePtr = qscale + bk * EP;
            const float* biasPtr = qbias + bk * EP;

            for (int k = 0; k < blockLU; ++k) {
                int base = n * stride0 + bk * stride1 + k * EP * LP;
                const float* srcBase = src + base;
                int8_t* dstBase = dst + base;

                for (int i = 0; i < EP; i += EP_TILE) {
                    int tile = (i + EP_TILE <= EP) ? EP_TILE : (EP - i);

                    __builtin_prefetch(srcBase + (i + EP_TILE) * LP, 0, 1);

                    for (int t = 0; t < tile; ++t) {
                        float scaleVal = scalePtr[i + t];
                        float biasVal = biasPtr[i + t];

                        const float* srcZ = srcBase + (i + t) * LP;
                        int8_t* dstZ = dstBase + (i + t) * LP;

                        int j = 0;

                        while (j < LP) {
                            size_t vl = __riscv_vsetvl_e32m4(LP - j);

                            vfloat32m4_t v = __riscv_vle32_v_f32m4(srcZ + j, vl);

                            v = __riscv_vfmul_vf_f32m4(v, scaleVal, vl);
                            v = __riscv_vfadd_vf_f32m4(v, biasVal, vl);

                            vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4_rm(v, RV_RNU, vl);

                            vi = __riscv_vmax_vx_i32m4(vi, int8_min, vl);
                            vi = __riscv_vmin_vx_i32m4(vi, int8_max, vl);

                            vint16m2_t v16 = __riscv_vncvt_x_x_w_i16m2(vi, vl);
                            vint8m1_t v8 = __riscv_vncvt_x_x_w_i8m1(v16, vl);

                            __riscv_vse8_v_i8m1(dstZ + j, v8, vl);

                            j += vl;
                        }
                    }
                }
            }
        }
    }
}
