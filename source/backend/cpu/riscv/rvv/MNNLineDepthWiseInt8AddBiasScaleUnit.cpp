//
//  MNNLineDepthWiseInt8AddBiasScaleUnit.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>

void MNNLineDepthWiseInt8AddBiasScaleUnit_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                              const QuanPostTreatParameters* parameters, size_t width,
                                              size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                              size_t dilateY_step, int8_t* idxOrder) {
    const size_t vl = __riscv_vsetvl_e8m1(16);
    const int offset = 0;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    const int8_t* weightPtr = weight;

    const float* bias_z = parameters->bias;
    const float* scale_z = parameters->scale;
    const int32_t max_val = parameters->maxValue;
    const int32_t min_val = parameters->minValue;

    vint32m4_t v_min = __riscv_vmv_v_x_i32m4(min_val + offset, vl);
    vint32m4_t v_max = __riscv_vmv_v_x_i32m4(max_val + offset, vl);

    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        int8_t* dst_x = dstPtr + dx * 16;
        const int8_t* src_z = srcPtr + src_w_step * dx;

        vint32m4_t vec_sum = __riscv_vmv_v_x_i32m4(0, vl);

        for (fy = 0; fy < fh; ++fy) {
            const int8_t* src_y = src_z + fy * dilateY_step;
            const int8_t* weight_y = weightPtr + fy * fw * 16;

            for (fx = 0; fx < fw; ++fx) {
                const int8_t* src_x = src_y + fx * dilateX_step;
                const int8_t* weight_x = weight_y + 16 * fx;

                vint8m1_t vec_src = __riscv_vle8_v_i8m1(src_x, vl);
                vint8m1_t vec_weight = __riscv_vle8_v_i8m1(weight_x, vl);

                vint16m2_t s_src = __riscv_vsext_vf2_i16m2(vec_src, vl);
                vint16m2_t s_wgt = __riscv_vsext_vf2_i16m2(vec_weight, vl);
                vint16m2_t vec_mul = __riscv_vmul_vv_i16m2(s_src, s_wgt, vl);

                vint32m4_t vec_mul_32 = __riscv_vsext_vf2_i32m4(vec_mul, vl);
                vec_sum = __riscv_vadd_vv_i32m4(vec_sum, vec_mul_32, vl);
            }
        }

        vfloat32m4_t vec_bias = __riscv_vle32_v_f32m4(bias_z, vl);
        vfloat32m4_t f_sum = __riscv_vfcvt_f_x_v_f32m4(vec_sum, vl);
        f_sum = __riscv_vfadd_vv_f32m4(f_sum, vec_bias, vl);

        vfloat32m4_t vec_scale = __riscv_vle32_v_f32m4(scale_z, vl);
        f_sum = __riscv_vfmul_vv_f32m4(f_sum, vec_scale, vl);

        vint32m4_t v_res = __riscv_vfcvt_x_f_v_i32m4(f_sum, vl);
        v_res = __riscv_vmax_vv_i32m4(v_res, v_min, vl);
        v_res = __riscv_vmin_vv_i32m4(v_res, v_max, vl);

        vint16m2_t v_res_16 = __riscv_vncvt_x_x_w_i16m2(v_res, vl);
        vint8m1_t v_out = __riscv_vncvt_x_x_w_i8m1(v_res_16, vl);

        __riscv_vse8_v_i8m1(dst_x, v_out, vl);
    }
}
