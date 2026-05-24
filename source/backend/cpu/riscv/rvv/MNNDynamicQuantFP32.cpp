#include <riscv_vector.h>

void MNNDynamicQuantFP32_RVV(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize,
                             int pack, const float* bias) {
    size_t stride = pack * realSize;
    const int int8_min = -128;
    const int int8_max = 127;
    const size_t RV_RNU = 0x4;
    for (size_t i = 0; i < realSize; ++i) {
        float scaleVal = scale[i];
        float biasVal = (bias != nullptr) ? bias[i] : 0.0f;
        for (size_t c = 0; c < src_depth_quad; ++c) {
            const float* srcZ = src + c * stride + i * pack;
            int8_t* dstZ = dst + c * stride + i * pack;
            size_t n = pack;
            size_t vl;
            for (; n > 0; n -= vl, srcZ += vl, dstZ += vl) {
                vl = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v = __riscv_vle32_v_f32m4(srcZ, vl);
                vfloat32m4_t vbias = __riscv_vfmv_v_f_f32m4(biasVal, vl);
                v = __riscv_vfmadd_vf_f32m4(v, scaleVal, vbias, vl);
                vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4_rm(v, RV_RNU, vl);
                vi = __riscv_vmax_vx_i32m4(vi, int8_min, vl);
                vi = __riscv_vmin_vx_i32m4(vi, int8_max, vl);
                vint16m2_t v16 = __riscv_vncvt_x_x_w_i16m2(vi, vl);
                vint8m1_t v8 = __riscv_vncvt_x_x_w_i8m1(v16, vl);
                __riscv_vse8_v_i8m1(dstZ, v8, vl);
            }
        }
    }
}
