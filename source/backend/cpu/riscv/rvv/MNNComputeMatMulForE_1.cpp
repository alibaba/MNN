// Copyright © 2026, Alibaba Group Holding Limited
#include "MNNRvvMatMulFunctions.hpp"
#include <riscv_vector.h>

void MNNComputeMatMulForE_1_RVV(const float* A, const float* B, float* C, const float* bias, const MatMulParam* param,
                                size_t tId) {
    if (param->BTranspose) {
        MNNComputeMatMulForE_1(A, B, C, bias, param, tId);
        return;
    }
    // Raw A[K], B[K,H], C[H]. No packing or reduction across vector lanes.
    // CPUMatMul's E=1 scheduler does not initialize ATranspose; this path must not read it.
    const size_t h = param->h;
    const size_t k = param->l;
    const size_t lanes = __riscv_vsetvlmax_e32m4();
    const size_t step = lanes * param->numberThread;
    // Whole vector blocks have one owner, including the final partial block.
    for (size_t y = tId * lanes; y < h; y += step) {
        const size_t vl = __riscv_vsetvl_e32m4(h - y < lanes ? h - y : lanes);
        auto sum = bias ? __riscv_vle32_v_f32m4(bias + y, vl) : __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (size_t z = 0; z < k; ++z) {
            auto weight = __riscv_vle32_v_f32m4(B + z * h + y, vl);
            auto product = __riscv_vfmul_vf_f32m4(weight, A[z], vl);
            sum = __riscv_vfadd_vv_f32m4(sum, product, vl);
        }
        __riscv_vse32_v_f32m4(C + y, sum, vl);
    }
}
