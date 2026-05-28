#include <riscv_vector.h>

extern "C" {

void MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    size_t count = countC8 * 8;
    float xLimit = 87.0f;

    float inputScale = offset[0];
    float outputBias = offset[1];
    float inputBias = offset[2];
    float summer = offset[3];
    float param = parameters[0];    // ln(2)
    float invParam = parameters[1]; // 1/ln(2)
    float p4 = parameters[4];
    float p5 = parameters[5];
    float p6 = parameters[6];
    float p7 = parameters[7];

    size_t n = count;
    const float* srcPtr = source;
    float* dstPtr = dest;
    vfloat32m1_t vSum = __riscv_vfmv_s_f_f32m1(summer, 1);

    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);

        vfloat32m4_t vX = __riscv_vle32_v_f32m4(srcPtr, vl);
        vX = __riscv_vfmul_vf_f32m4(vX, inputScale, vl);
        vX = __riscv_vfadd_vf_f32m4(vX, inputBias, vl);
        vX = __riscv_vfmax_vf_f32m4(vX, -xLimit, vl);
        vX = __riscv_vfmin_vf_f32m4(vX, xLimit, vl);

        // div = int(x / ln(2))
        vfloat32m4_t vDiv = __riscv_vfmul_vf_f32m4(vX, invParam, vl);
        vint32m4_t vDivI = __riscv_vfcvt_x_f_v_i32m4(vDiv, vl);

        // Start float conversion early — its latency overlaps with integer ops below
        vfloat32m4_t vDivF = __riscv_vfcvt_f_x_v_f32m4(vDivI, vl);

        // expBasic = 2^div via IEEE 754 (integer ops, fast, runs while vDivF completes)
        vfloat32m4_t vExpBasic =
            __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(vDivI, 127, vl), 23, vl));

        // xRemain = x - div * ln(2)
        vfloat32m4_t vRemain = __riscv_vfnmsub_vf_f32m4(vDivF, param, vX, vl);
        vfloat32m4_t vT = __riscv_vfmul_vf_f32m4(vRemain, 0.25f, vl);

        // Taylor: ((((p7*t + p6)*t + p5)*t + p4)*t + 1)*t + 1
        vfloat32m4_t vPoly = __riscv_vfmv_v_f_f32m4(p7, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vT, vl);
        vPoly = __riscv_vfadd_vf_f32m4(vPoly, p6, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vT, vl);
        vPoly = __riscv_vfadd_vf_f32m4(vPoly, p5, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vT, vl);
        vPoly = __riscv_vfadd_vf_f32m4(vPoly, p4, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vT, vl);
        vPoly = __riscv_vfadd_vf_f32m4(vPoly, 1.0f, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vT, vl);
        vPoly = __riscv_vfadd_vf_f32m4(vPoly, 1.0f, vl);

        // (exp(x/4))^4: square twice
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vPoly, vl);
        vPoly = __riscv_vfmul_vv_f32m4(vPoly, vPoly, vl);

        // result = expBasic * expRemain + outputBias
        vfloat32m4_t vResult = __riscv_vfmul_vv_f32m4(vExpBasic, vPoly, vl);
        vResult = __riscv_vfadd_vf_f32m4(vResult, outputBias, vl);
        __riscv_vse32_v_f32m4(dstPtr, vResult, vl);

        vSum = __riscv_vfredusum_vs_f32m4_f32m1(vResult, vSum, vl);

        srcPtr += vl;
        dstPtr += vl;
        n -= vl;
    }

    offset[3] = __riscv_vfmv_f_s_f32m1_f32(vSum);
}

} // extern "C"
