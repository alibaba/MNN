#include <riscv_vector.h>
#include <cmath>

extern "C" {

void MNNGeluCommon(float* dst, const float* src, size_t size) {
    const float p0 = 0.044715f;
    const float p1 = 0.79788458f; // sqrt(2/pi)
    const float c0 = 378.0f;
    const float c1 = 17325.0f;
    const float c2 = 135135.0f;
    const float c3 = 28.0f;
    const float c4 = 3150.0f;
    const float c5 = 62370.0f;

    size_t n = size;
    const float* srcPtr = src;
    float* dstPtr = dst;

    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vX = __riscv_vle32_v_f32m4(srcPtr, vl);

        // temp = x + 0.044715 * x^3 = x * (1 + 0.044715 * x^2)
        vfloat32m4_t vX2 = __riscv_vfmul_vv_f32m4(vX, vX, vl);
        vfloat32m4_t vTemp = __riscv_vfmul_vf_f32m4(vX2, p0, vl);
        vTemp = __riscv_vfadd_vf_f32m4(vTemp, 1.0f, vl);
        vTemp = __riscv_vfmul_vv_f32m4(vTemp, vX, vl);

        // value = 0.79788458 * temp, clamp to [-5, 5]
        vfloat32m4_t vVal = __riscv_vfmul_vf_f32m4(vTemp, p1, vl);
        vVal = __riscv_vfmax_vf_f32m4(vVal, -5.0f, vl);
        vVal = __riscv_vfmin_vf_f32m4(vVal, 5.0f, vl);

        // x2 = value * value
        vfloat32m4_t vV2 = __riscv_vfmul_vv_f32m4(vVal, vVal, vl);

        // numerator: a = value * (135135 + x2 * (17325 + x2 * (378 + x2)))
        vfloat32m4_t vA = __riscv_vfadd_vf_f32m4(vV2, c0, vl);
        vA = __riscv_vfmul_vv_f32m4(vA, vV2, vl);
        vA = __riscv_vfadd_vf_f32m4(vA, c1, vl);
        vA = __riscv_vfmul_vv_f32m4(vA, vV2, vl);
        vA = __riscv_vfadd_vf_f32m4(vA, c2, vl);
        vA = __riscv_vfmul_vv_f32m4(vA, vVal, vl);

        // denominator: b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
        vfloat32m4_t vB = __riscv_vfmul_vf_f32m4(vV2, c3, vl);
        vB = __riscv_vfadd_vf_f32m4(vB, c4, vl);
        vB = __riscv_vfmul_vv_f32m4(vB, vV2, vl);
        vB = __riscv_vfadd_vf_f32m4(vB, c5, vl);
        vB = __riscv_vfmul_vv_f32m4(vB, vV2, vl);
        vB = __riscv_vfadd_vf_f32m4(vB, c2, vl);

        // tanh_approx = a / b (vfrec7 + 2x Newton-Raphson)
        vfloat32m4_t vRcp = __riscv_vfrec7_v_f32m4(vB, vl);
        vfloat32m4_t vErr = __riscv_vfmul_vv_f32m4(vB, vRcp, vl);
        vErr = __riscv_vfrsub_vf_f32m4(vErr, 2.0f, vl);
        vRcp = __riscv_vfmul_vv_f32m4(vRcp, vErr, vl);
        vErr = __riscv_vfmul_vv_f32m4(vB, vRcp, vl);
        vErr = __riscv_vfrsub_vf_f32m4(vErr, 2.0f, vl);
        vRcp = __riscv_vfmul_vv_f32m4(vRcp, vErr, vl);
        vfloat32m4_t vTanh = __riscv_vfmul_vv_f32m4(vA, vRcp, vl);

        // clamp tanh to [-1, 1]
        vTanh = __riscv_vfmax_vf_f32m4(vTanh, -1.0f, vl);
        vTanh = __riscv_vfmin_vf_f32m4(vTanh, 1.0f, vl);

        // result = 0.5 * x * (1 + tanh)
        vTanh = __riscv_vfadd_vf_f32m4(vTanh, 1.0f, vl);
        vfloat32m4_t vResult = __riscv_vfmul_vv_f32m4(vX, vTanh, vl);
        vResult = __riscv_vfmul_vf_f32m4(vResult, 0.5f, vl);
        __riscv_vse32_v_f32m4(dstPtr, vResult, vl);

        srcPtr += vl;
        dstPtr += vl;
        n -= vl;
    }
}

void MNNGeluStandardCommon(float* dst, const float* src, size_t size) {
    // Standard GeLU using erf: dst = 0.5 * x * (1 + erf(x / sqrt(2)))
    // Keep scalar — erf has no simple polynomial vectorization
    for (int i = 0; i < size; i++) {
        dst[i] = (erf(src[i] * 0.7071067932881648f) + 1.0f) * src[i] * 0.5f;
    }
}

} // extern "C"
