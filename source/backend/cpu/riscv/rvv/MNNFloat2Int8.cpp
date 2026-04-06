//
//  MNNFloat2Int8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <stdint.h>

void MNNFloat2Int8_RVV(const float* src, int8_t* dst, size_t sizeQuad,
                       const float* scalep, ssize_t minValue, ssize_t maxValue,
                       const float* zeroPoint, ssize_t quanParamVec)
{
    float scale[4] = {scalep[0], scalep[0], scalep[0], scalep[0]};
    float zero[4]  = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};

    if (quanParamVec & 1) {
        scale[0]=scalep[0]; scale[1]=scalep[1];
        scale[2]=scalep[2]; scale[3]=scalep[3];
    }
    if (quanParamVec & 2) {
        zero[0]=zeroPoint[0]; zero[1]=zeroPoint[1];
        zero[2]=zeroPoint[2]; zero[3]=zeroPoint[3];
    }

    const float minf = (float)minValue;
    const float maxf = (float)maxValue;
    const size_t total = sizeQuad * 4;

    // ---- 循环外预构造周期向量 ----
    // 取一次 vl，用于构造 scale/zero 的周期模板
    // 固定用 e32m1 构造模板，再在主循环里按需使用
    size_t vl_template = __riscv_vsetvlmax_e32m2();
    // 保证 vl_template 是 4 的倍数（RISC-V Vector 规范保证 VLEN>=128，vl>=4）

    // 用 iota + 取模 构造 channel index，然后 gather scale/zero
    // index: [0,1,2,3,0,1,2,3,...]
    vuint32m2_t v_idx = __riscv_vid_v_u32m2(vl_template);           // [0,1,2,...,vl-1]
    vuint32m2_t v_ch  = __riscv_vremu_vx_u32m2(v_idx, 4, vl_template); // [0,1,2,3,0,1,2,3,...]

    // gather：从 scale[4] 和 zero[4] 按 channel index 取值
    vfloat32m2_t v_scale_tpl = __riscv_vloxei32_v_f32m2(scale, 
                                    __riscv_vsll_vx_u32m2(v_ch, 2, vl_template), 
                                    vl_template);
    vfloat32m2_t v_zero_tpl  = __riscv_vloxei32_v_f32m2(zero,
                                    __riscv_vsll_vx_u32m2(v_ch, 2, vl_template),
                                    vl_template);

    vfloat32m2_t v_min = __riscv_vfmv_v_f_f32m2(minf, vl_template);
    vfloat32m2_t v_max = __riscv_vfmv_v_f_f32m2(maxf, vl_template);

    // ---- 主循环：纯向量运算，无标量介入 ----
    size_t i = 0;
    while (i < total) {
        size_t vl = __riscv_vsetvl_e32m2(total - i);

        vfloat32m2_t v_src = __riscv_vle32_v_f32m2(src + i, vl);

        // scale 和 zero 直接复用模板（vl <= vl_template，前 vl 个元素正好对齐）
        vfloat32m2_t v_mul   = __riscv_vfmul_vv_f32m2(v_src, v_scale_tpl, vl);
        vfloat32m2_t v_add   = __riscv_vfadd_vv_f32m2(v_mul, v_zero_tpl, vl);
        vfloat32m2_t v_clamp = __riscv_vfmin_vv_f32m2(
                                   __riscv_vfmax_vv_f32m2(v_add, v_min, vl),
                                   v_max, vl);

        // 浮点→int8，两级窄化
        vint32m2_t v_i32 = __riscv_vfcvt_x_f_v_i32m2(v_clamp, vl);
        vint16m1_t v_i16 = __riscv_vncvt_x_x_w_i16m1(v_i32, vl);
        vint8mf2_t v_i8  = __riscv_vncvt_x_x_w_i8mf2(v_i16, vl);

        __riscv_vse8_v_i8mf2(dst + i, v_i8, vl);

        i += vl;
    }
}
